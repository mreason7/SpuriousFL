"""Plan/check/run four datasets with the same author-email protocol."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASETS = ('cmnist_gsc', 'spawrious_gsc', 'spawrious_gci', 'spawrious_gai')


def resolved_config(name):
    from hydra import compose, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
    from src.config_params import Config
    ConfigStore.instance().store(group='job', name='federated_training', node=Config)
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base=None):
        return OmegaConf.to_container(compose(config_name=name), resolve=True, throw_on_missing=True)


def verify(directory, rounds, method):
    for name in ('config.yaml', 'final/torchmodel.pt', 'client_weights.csv', 'schedule.jsonl',
                 'data_manifest.json', 'data_manifest.sha256', 'initial_weights.sha256',
                 'final_test_metrics.json', 'unified_protocol.json'):
        if not (directory / name).is_file():
            raise RuntimeError(f'Missing artifact: {directory / name}')
    rows = [[float(x) for x in line.split(',')] for line in
            (directory / 'client_weights.csv').read_text().splitlines()]
    counts = ([24, 24] + [9] * (rounds - 2) if method == 'feddiverse'
              else [24] + [9] * (rounds - 1))
    if len(rows) != rounds or any(len(r) != 24 or sum(r) != n or
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
    for dataset in DATASETS:
        for method in ('random', 'feddiverse'):
            group = [r for r in rows if r['dataset'] == dataset and r['method'] == method]
            if group:
                values = [r['worst_group'] for r in group]
                result.append(dict(dataset=dataset, method=method, n=len(group),
                    seeds=[r['seed'] for r in group], mean=statistics.mean(values),
                    std_population=statistics.pstdev(values)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('plan', 'check', 'preflight', 'smoke', 'formal', 'summary'), default='plan')
    p.add_argument('--seeds', default=None)
    p.add_argument('--datasets', default=','.join(DATASETS))
    p.add_argument('--gpu', default='0')
    p.add_argument('--batch', type=Path, help='Existing batch folder for summary')
    args = p.parse_args()
    os.chdir(ROOT)
    if args.stage == 'summary':
        if args.batch is None:
            p.error('--batch is required for summary')
        print(json.dumps(summarize(json.loads((args.batch / 'summary.json').read_text())), indent=2))
        return
    seeds = [int(x) for x in (args.seeds or ('42' if args.stage == 'smoke' else '42,3407')).split(',')]
    datasets = args.datasets.split(',')
    if len(seeds) != len(set(seeds)) or not seeds or any(s < 0 or s >= 2**32 for s in seeds):
        p.error('Seeds must be unique integers in [0, 2**32)')
    if len(datasets) != len(set(datasets)) or not set(datasets) <= set(DATASETS):
        p.error('Unknown or duplicate dataset')
    jobs = []
    for dataset in datasets:
        for seed in seeds:
            for method in ('random', 'feddiverse'):
                rounds = (2 if method == 'random' else 3) if args.stage == 'smoke' else (200 if method == 'random' else 201)
                config = f'email_{dataset}_{method}'
                command = [sys.executable, 'train_email_four.py', f'--config-name={config}',
                           f'seed={seed}', f'server_opt.rounds={rounds}']
                if args.stage == 'smoke':
                    command += ['client_opt.num_steps=2', 'client_opt.biased_trainer_steps=2',
                                'client_opt.left_right_trainer_steps=2']
                jobs.append(dict(dataset=dataset, seed=seed, method=method, rounds=rounds,
                                 config=config, command=command))
    if args.stage == 'plan':
        print(json.dumps(dict(protocol='email-four-v1', jobs=jobs), indent=2))
        return
    for dataset in datasets:
        fed = resolved_config(f'email_{dataset}_feddiverse')
        rnd = resolved_config(f'email_{dataset}_random')
        for key in ('client_opt', 'model_options', 'dataset_options', 'machine', 'seed'):
            if fed[key] != rnd[key]:
                raise RuntimeError(f'Unmatched pair: {dataset}/{key}')
        differences = {k for k in fed['server_opt'] if fed['server_opt'][k] != rnd['server_opt'][k]}
        if differences != {'rounds', 'selection_method', 'client_info'}:
            raise RuntimeError(f'Unexpected method difference: {differences}')
        if (fed['dataset_options']['num_clients'], fed['server_opt']['rounds'],
            rnd['server_opt']['rounds'], fed['server_opt']['paper_faithful_pretrain']) != (24, 201, 200, False):
            raise RuntimeError('Incorrect email protocol settings')
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
    import train_email_four  # Check actual runtime imports before launching training.
    environment = dict(gpu=torch.cuda.get_device_name(0), python=sys.version,
        pretrained_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        versions={n: importlib.metadata.version(n) for n in
                  ('torch', 'torchvision', 'flwr', 'ray', 'numpy', 'pandas', 'hydra-core', 'timm')})
    print(json.dumps(environment, indent=2), flush=True)
    if args.stage == 'preflight':
        print('PREFLIGHT PASSED; no training started')
        return
    output = ROOT / 'run_logs' / ('email-four-' + args.stage + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    (output / 'plan.json').write_text(json.dumps(dict(jobs=jobs, environment=environment, commit=commit), indent=2))
    summary, pairs = [], {}
    for job in jobs:
        before = set((ROOT / 'checkpoints').glob('*/config.yaml'))
        env = dict(os.environ, PYTHONHASHSEED=str(job['seed']), AUTHOR_PROTOCOL='email-four-v1')
        log = output / f"{job['dataset']}_{job['method']}_seed{job['seed']}.log"
        print('START', job['dataset'], job['method'], job['seed'], 'log=', log, flush=True)
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(job['command'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Run failed ({result.returncode}); stopped. Inspect {log}')
        created = set((ROOT / 'checkpoints').glob('*/config.yaml')) - before
        if len(created) != 1:
            raise RuntimeError('Expected one new run directory')
        directory = created.pop().parent
        metrics = verify(directory, job['rounds'], job['method'])
        hashes = {n: (directory / n).read_text().strip() for n in
                  ('data_manifest.sha256', 'initial_weights.sha256')}
        key = (job['dataset'], job['seed'])
        if key in pairs and pairs[key] != hashes:
            raise RuntimeError(f'Pair data/initialization differs; stopped: {key}, {directory}')
        pairs[key] = hashes
        summary.append(dict(dataset=job['dataset'], method=job['method'], seed=job['seed'],
                            checkpoint=str(directory), log=str(log), hashes=hashes, **metrics))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        (output / 'aggregate.json').write_text(json.dumps(summarize(summary), indent=2))
        print('VERIFIED', job['dataset'], job['method'], job['seed'], metrics, flush=True)
    print('COMPLETED', len(summary), output, flush=True)


if __name__ == '__main__':
    main()
