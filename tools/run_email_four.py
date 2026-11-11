"""Plan/check/run four datasets with the same author-email protocol."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASETS = ('cmnist_gsc', 'spawrious_gsc', 'spawrious_gci', 'spawrious_gai')
REMAINING_DATASETS = ('waterbirds_dist', 'spawrious_4', 'spawrious_gci_100')
ALL_DATASETS = DATASETS + REMAINING_DATASETS
DEFAULT_DATASETS = DATASETS
BATCH_PREFIX = 'email-four'
CONFIG_SUFFIX = ''
TRAIN_ENTRY = 'train_email_four.py'
CLIENT_COUNTS = dict(zip(ALL_DATASETS, (24, 24, 24, 25, 30, 25, 100)))


def active_clients(dataset):
    return 12 if dataset == 'spawrious_gci_100' else 9


def resolved_config(name):
    from hydra import compose, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
    from src.config_params import Config
    ConfigStore.instance().store(group='job', name='federated_training', node=Config)
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base=None):
        return OmegaConf.to_container(compose(config_name=name), resolve=True, throw_on_missing=True)


def verify(directory, rounds, method, num_clients=24, selected_clients=9):
    for name in ('config.yaml', 'final/torchmodel.pt', 'client_weights.csv', 'schedule.jsonl',
                 'data_manifest.json', 'data_manifest.sha256', 'initial_weights.sha256',
                 'final_test_metrics.json', 'unified_protocol.json'):
        if not (directory / name).is_file():
            raise RuntimeError(f'Missing artifact: {directory / name}')
    rows = [[float(x) for x in line.split(',')] for line in
            (directory / 'client_weights.csv').read_text().splitlines()]
    counts = ([num_clients, num_clients] + [selected_clients] * (rounds - 2) if method == 'feddiverse'
              else [num_clients] + [selected_clients] * (rounds - 1))
    if len(rows) != rounds or any(len(r) != num_clients or sum(r) != n or
            any(x not in (0, 1) for x in r) for r, n in zip(rows, counts)):
        raise RuntimeError('Incorrect participation schedule')
    events = [json.loads(x) for x in (directory / 'schedule.jsonl').read_text().splitlines()]
    updates = [e for e in events if e['phase'] == 'aggregated']
    no_update = [e['round'] for e in updates if not e['model_update']]
    if len(updates) != rounds or no_update != ([2] if method == 'feddiverse' else []):
        raise RuntimeError('Incorrect DHT/update schedule')
    return json.loads((directory / 'final_test_metrics.json').read_text())


def summarize(rows):
    result = []
    for dataset in ALL_DATASETS:
        for method in ('random', 'feddiverse'):
            group = [r for r in rows if r['dataset'] == dataset and r['method'] == method]
            if group:
                values = [r['worst_group'] for r in group]
                result.append(dict(dataset=dataset, method=method, n=len(group),
                    seeds=[r['seed'] for r in group], mean=statistics.mean(values),
                    std_population=statistics.pstdev(values)))
    return result


def run_summary(job, metrics, directory, stage, elapsed):
    updates = job['rounds'] - int(job['method'] == 'feddiverse')
    lines = ['', '=' * 64, 'FINAL TRAINING SUMMARY', 'Status: COMPLETED AND VERIFIED',
             f"Dataset: {job['dataset']} | Method: {job['method']} | Seed: {job['seed']}",
             f"Stage: {stage} | Clients: {job['num_clients']} | Selected per update after round 1: {job.get('selected_clients', 9)}",
             f"Model updates: {updates} | Flower events: {job['rounds']}",
             f"Elapsed: {elapsed / 60:.2f} minutes",
             f"Final test accuracy: {metrics['test_accuracy']:.4f}%",
             f"Final worst-group accuracy (WGA): {metrics['worst_group']:.4f}%"]
    keys = sorted((k for k in metrics if re.fullmatch(r'y\d+g\d+', k)),
                  key=lambda k: tuple(map(int, re.findall(r'\d+', k))))
    groups = {k: metrics[k] for k in keys}
    for name, accuracy in groups.items():
        lines.append(f'  {name}: {accuracy:.4f}%')
    if groups:
        worst = min(groups.values())
        names = [k for k, value in groups.items() if abs(value - worst) < 1e-8]
        lines.append('Worst group(s): ' + ', '.join(names))
    lines += [f'Checkpoint: {directory / "final/torchmodel.pt"}',
              'Evaluation: final checkpoint; intermediate best scores are not selected.']
    if stage == 'smoke':
        lines.append('Smoke test only: reduced training budget; do not assess method effectiveness.')
    else:
        lines.append('Method effectiveness is assessed against the matched control and across seeds.')
    return '\n'.join(lines + ['=' * 64, ''])


def pair_summary(rows, stage):
    methods = {r['method']: r for r in rows}
    if not {'random', 'feddiverse'} <= methods.keys():
        return None
    random, fed = methods['random'], methods['feddiverse']
    delta = fed['worst_group'] - random['worst_group']
    accuracy_delta = fed['test_accuracy'] - random['test_accuracy']
    conclusion = ('higher' if delta > 1e-8 else 'lower' if delta < -1e-8 else 'equal')
    return '\n'.join(['', '=' * 64, 'MATCHED PAIR SUMMARY',
        f"Dataset: {fed['dataset']} | Seed: {fed['seed']}",
        'Data manifest and initial model hashes: MATCHED',
        f"Random WGA: {random['worst_group']:.4f}%",
        f"FedDiverse WGA: {fed['worst_group']:.4f}%",
        f'FedDiverse - Random WGA: {delta:+.4f} percentage points',
        f'FedDiverse - Random test accuracy: {accuracy_delta:+.4f} percentage points',
        ('Smoke test only: no effectiveness conclusion.' if stage == 'smoke' else
         f'Observed final WGA in this pair: FedDiverse is {conclusion} than Random; '
         'this single seed does not establish a stable advantage.'), '=' * 64, ''])


def append_log(path, text):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(text)


def load_append_rows(output, jobs):
    """Read an existing formal batch and reject overwriting any run/log."""
    if not output.is_dir() or not (output / 'plan.json').is_file():
        raise RuntimeError(f'Existing batch/plan not found: {output}')
    plan = json.loads((output / 'plan.json').read_text())
    if any(j['rounds'] != (201 if j['method'] == 'feddiverse' else 200) for j in plan['jobs']):
        raise RuntimeError('Cannot append formal runs to a smoke batch')
    rows = json.loads((output / 'summary.json').read_text())
    keys = [(r['dataset'], r['seed'], r['method']) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError('Existing summary has duplicate runs')
    for job in jobs:
        log = output / f"{job['dataset']}_{job['method']}_seed{job['seed']}.log"
        if (job['dataset'], job['seed'], job['method']) in keys or log.exists():
            raise RuntimeError(f'Run or log already exists; refusing to overwrite: {log}')
    return rows


def save_extension_plan(output, jobs, environment, commit, stamp):
    original = json.loads((output / 'plan.json').read_text())
    for key in ('pretrained_sha256', 'versions'):
        if original['environment'][key] != environment[key]:
            raise RuntimeError(f'Existing batch environment differs: {key}')
    # Preserve the original plan and pre-extension summaries byte for byte.
    for name in ('summary.json', 'aggregate.json'):
        source = output / name
        if source.is_file():
            with (output / f'{source.stem}_before_append_{stamp}.json').open('xb') as stream:
                stream.write(source.read_bytes())
    with (output / f'extension_plan_{stamp}.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(jobs=jobs, environment=environment, commit=commit), stream, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('plan', 'check', 'preflight', 'smoke', 'formal', 'summary'), default='plan')
    p.add_argument('--seeds', default=None)
    p.add_argument('--datasets', default=','.join(DEFAULT_DATASETS))
    p.add_argument('--gpu', default='0')
    p.add_argument('--batch', type=Path, help='Existing formal batch to append to, or to summarize')
    args = p.parse_args()
    os.chdir(ROOT)
    if args.stage == 'summary':
        if args.batch is None:
            p.error('--batch is required for summary')
        print(json.dumps(summarize(json.loads((args.batch / 'summary.json').read_text())), indent=2))
        return
    if args.batch is not None and args.stage == 'smoke':
        p.error('Smoke runs must remain separate from the formal batch; omit --batch for smoke')
    seeds = [int(x) for x in (args.seeds or ('42' if args.stage == 'smoke' else '42,3407')).split(',')]
    datasets = args.datasets.split(',')
    if len(seeds) != len(set(seeds)) or not seeds or any(s < 0 or s >= 2**32 for s in seeds):
        p.error('Seeds must be unique integers in [0, 2**32)')
    if not datasets or len(datasets) != len(set(datasets)) or not set(datasets) <= set(ALL_DATASETS):
        p.error('Unknown or duplicate dataset')
    jobs = []
    for dataset in datasets:
        for seed in seeds:
            for method in ('random', 'feddiverse'):
                rounds = (2 if method == 'random' else 3) if args.stage == 'smoke' else (200 if method == 'random' else 201)
                config = f'email_{dataset}_{method}{CONFIG_SUFFIX}'
                command = [sys.executable, TRAIN_ENTRY, f'--config-name={config}',
                           f'seed={seed}', f'server_opt.rounds={rounds}']
                if args.stage == 'smoke':
                    command += ['client_opt.num_steps=2', 'client_opt.biased_trainer_steps=2',
                                'client_opt.left_right_trainer_steps=2']
                jobs.append(dict(dataset=dataset, seed=seed, method=method, rounds=rounds,
                                 config=config, command=command,
                                 num_clients=CLIENT_COUNTS[dataset], selected_clients=active_clients(dataset)))
    existing = load_append_rows(args.batch, jobs) if args.batch is not None else []
    if args.stage == 'plan':
        print(json.dumps(dict(protocol='email-four-v1', jobs=jobs,
                              append_to=str(args.batch) if args.batch else None,
                              existing_runs=len(existing)), indent=2))
        return
    for dataset in datasets:
        fed = resolved_config(f'email_{dataset}_feddiverse{CONFIG_SUFFIX}')
        rnd = resolved_config(f'email_{dataset}_random{CONFIG_SUFFIX}')
        for key in ('client_opt', 'model_options', 'dataset_options', 'machine', 'seed'):
            if fed[key] != rnd[key]:
                raise RuntimeError(f'Unmatched pair: {dataset}/{key}')
        differences = {k for k in fed['server_opt'] if fed['server_opt'][k] != rnd['server_opt'][k]}
        if differences != {'rounds', 'selection_method', 'client_info'}:
            raise RuntimeError(f'Unexpected method difference: {differences}')
        if (fed['dataset_options']['num_clients'], fed['server_opt']['rounds'],
            rnd['server_opt']['rounds'], fed['server_opt']['paper_faithful_pretrain']) != (
                CLIENT_COUNTS[dataset], 201, 200, False):
            raise RuntimeError('Incorrect email protocol settings')
        if fed['server_opt']['num_active_clients'] != active_clients(dataset):
            raise RuntimeError('Wrong per-round client budget')
    print('CONFIG CHECK PASSED', flush=True)
    if args.stage == 'check':
        return
    os.environ.update(AUTHOR_SEED_MODE='global', CUDA_VISIBLE_DEVICES=args.gpu)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training started')
    weights = ROOT / 'pretrained_ckpt/pt_mobilenetv2_100-4065ukrp/pretrained_imagenet.pth'
    if not weights.is_file():
        raise RuntimeError('Missing pretrained weights')
    if any(d.startswith('spawrious') for d in datasets):
        for name in ('0', '1'):
            if not (ROOT / 'datasets/spawrious224' / name).is_dir():
                raise RuntimeError('Missing Spawrious dataset')
    if 'waterbirds_dist' in datasets:
        from wilds import get_dataset
        # Validate local metadata and official splits; do not start a large download implicitly.
        get_dataset(dataset='waterbirds', download=False, root_dir=str(ROOT / 'datasets'))
    import train_email_four  # Check actual runtime imports before launching training.
    environment = dict(gpu=torch.cuda.get_device_name(0), python=sys.version,
        pretrained_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        versions={n: importlib.metadata.version(n) for n in
                  ('torch', 'torchvision', 'flwr', 'ray', 'numpy', 'pandas', 'hydra-core', 'timm')})
    print(json.dumps(environment, indent=2), flush=True)
    if args.stage == 'preflight':
        print('PREFLIGHT PASSED; no training started')
        return
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = args.batch or ROOT / 'run_logs' / (BATCH_PREFIX + '-' + args.stage + '-' + stamp)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if args.batch is not None:
        save_extension_plan(output, jobs, environment, commit, stamp)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'plan.json').write_text(json.dumps(dict(jobs=jobs, environment=environment, commit=commit), indent=2))
    summary, pairs = list(existing), {}
    for row in existing:
        key = (row['dataset'], row['seed'])
        if key in pairs and pairs[key] != row['hashes']:
            raise RuntimeError(f'Existing pair hashes differ: {key}')
        pairs[key] = row['hashes']
    for job in jobs:
        before = set((ROOT / 'checkpoints').glob('*/config.yaml'))
        env = dict(os.environ, PYTHONHASHSEED=str(job['seed']), AUTHOR_PROTOCOL='email-four-v1')
        log = output / f"{job['dataset']}_{job['method']}_seed{job['seed']}.log"
        print('START', job['dataset'], job['method'], job['seed'], 'log=', log, flush=True)
        started = time.monotonic()
        with log.open('x', encoding='utf-8') as stream:
            result = subprocess.run(job['command'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            append_log(log, f'\nFINAL TRAINING SUMMARY\nStatus: FAILED\n'
                f"Dataset: {job['dataset']} | Method: {job['method']} | Seed: {job['seed']}\n"
                f'Exit code: {result.returncode}\nNo verified final result; subsequent runs stopped.\n')
            raise RuntimeError(f'Run failed ({result.returncode}); stopped. Inspect {log}')
        try:
            created = set((ROOT / 'checkpoints').glob('*/config.yaml')) - before
            if len(created) != 1:
                raise RuntimeError('Expected one new run directory')
            directory = created.pop().parent
            metrics = verify(directory, job['rounds'], job['method'], job['num_clients'], job['selected_clients'])
            hashes = {n: (directory / n).read_text().strip() for n in
                      ('data_manifest.sha256', 'initial_weights.sha256')}
            key = (job['dataset'], job['seed'])
            if key in pairs and pairs[key] != hashes:
                raise RuntimeError(f'Pair data/initialization differs; stopped: {key}, {directory}')
        except Exception as error:
            append_log(log, f'\nFINAL TRAINING SUMMARY\nStatus: VERIFICATION FAILED\n'
                f"Dataset: {job['dataset']} | Method: {job['method']} | Seed: {job['seed']}\n"
                f'Reason: {error}\nMetrics are not accepted as a verified result; subsequent runs stopped.\n')
            raise
        pairs[key] = hashes
        summary.append(dict(dataset=job['dataset'], method=job['method'], seed=job['seed'],
                            checkpoint=str(directory), log=str(log), hashes=hashes, **metrics))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        (output / 'aggregate.json').write_text(json.dumps(summarize(summary), indent=2))
        footer = run_summary(job, metrics, directory, args.stage, time.monotonic() - started)
        append_log(log, footer)
        (directory / 'run_summary.txt').write_text(footer, encoding='utf-8')
        print(footer, flush=True)
        matched = [r for r in summary if (r['dataset'], r['seed']) == key]
        comparison = pair_summary(matched, args.stage)
        if comparison:
            for row in matched:
                append_log(Path(row['log']), comparison)
            print(comparison, flush=True)
    print('COMPLETED', len(summary), output, flush=True)


if __name__ == '__main__':
    main()
