"""Run the three missing estimated-DHT controls; never retrain old arms.

Use the original server checkpoints as immutable pairing references.
The default stage only prints the plan. GPU smoke is still required.
"""
import argparse
import copy
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILE = 'research_complementarity_gsc_estimated_dht'
REFERENCES = {
    42: dict(Random='20261112-090127', DHT='20261112-095107', Joint='20261112-104044'),
    43: dict(Random='20261112-113021', DHT='20261112-122003', Joint='20261112-130942'),
    44: dict(Random='20261112-135939', DHT='20261112-145003', Joint='20261112-153945'),
}


def config_at(directory):
    import yaml
    return yaml.safe_load((directory / 'config.yaml').read_text(encoding='utf-8'))


def identity(directory):
    values = tuple((directory / name).read_text().strip() for name in
                   ('data_manifest.sha256', 'initial_weights.sha256'))
    if any(len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in values):
        raise ValueError(f'Invalid identity hashes: {directory}')
    return values


def controls(conf):
    value = {k: copy.deepcopy(conf[k]) for k in
             ('seed', 'dataset_options', 'model_options', 'client_opt', 'server_opt', 'research', 'machine')}
    value['research'].pop('run_tag', None)
    value['server_opt'].pop('client_info', None)
    return value


def verify_config(conf, reference, seed, smoke=False):
    opt, research = conf['server_opt'], conf['research']
    if conf['seed'] != seed or opt['client_info'] != 'triplets_Npredicted':
        raise ValueError('Seed or estimated-DHT information source mismatch')
    if opt['selection_method'] != 'triplets_stochasticmatrix' or opt['selector_mode'] != 'guarded':
        raise ValueError('Controlled DHT selector mismatch')
    if not research['enabled'] or research['training_seed'] != seed+100 or research['validation_seed'] != seed+1000:
        raise ValueError('Controlled seed protocol mismatch')
    if reference['server_opt']['client_info'] != 'triplets':
        raise ValueError('Pairing reference is not the true-DHT control')
    left, right = controls(conf), controls(reference)
    if smoke:
        for value in (left, right):
            value['server_opt'].pop('rounds')
            for key in ('num_steps', 'biased_trainer_steps', 'left_right_trainer_steps'):
                value['client_opt'].pop(key)
    if left != right:
        differing = [k for k in left if left[k] != right[k]]
        raise ValueError(f'Controls differ from previous DHT run: {differing}')


def metrics(directory, rounds=201):
    for name in ('final/torchmodel.pt', 'client_info.json', 'data_manifest.json',
                 'final_validation_metrics.json', 'final_test_metrics.json'):
        if not (directory / name).is_file():
            raise ValueError(f'Missing {directory / name}')
    identity(directory)
    with (directory / 'client_weights.csv').open() as stream:
        weights = [[float(v) for v in row] for row in csv.reader(stream)]
    if len(weights) != rounds or any(len(row) != 24 or any(v not in (0, 1) for v in row) for row in weights):
        raise ValueError('Invalid participation table')
    if any(sum(row) != 24 for row in weights[:2]) or any(sum(row) != 9 for row in weights[2:]):
        raise ValueError('Participation schedule mismatch')
    selected = weights[2:]
    info = json.loads((directory / 'client_info.json').read_text())
    counts = [[float(info[str(i)][f'interaction_matrix_{g}']) for g in range(4)] for i in range(24)]
    if any(any(not math.isfinite(v) or v < 0 for v in row) or sum(row) <= 0 for row in counts):
        raise ValueError('Invalid true training counts')
    p = [[v/sum(row) for v in row] for row in counts]
    joints = [[sum(w[i]*p[i][g] for i in range(24))/9 for g in range(4)] for w in selected]
    times = [sum(w[i] for w in selected) for i in range(24)]
    shares = [t/sum(times) for t in times]
    test = json.loads((directory / 'final_test_metrics.json').read_text())
    val = json.loads((directory / 'final_validation_metrics.json').read_text())
    group_values = [float(test[f'y{y}g{g}']) for y in range(2) for g in range(2)]
    if not all(math.isfinite(v) for v in group_values):
        raise ValueError('Nonfinite group accuracies')
    group_wga = min(group_values)
    if not math.isclose(group_wga, float(test['worst_group']), abs_tol=1e-6):
        raise ValueError('Test WGA differs from group accuracies')
    result = dict(validation_WGA=float(val['worst_group']), test_WGA=float(test['worst_group']),
                  test_ACC=float(test['test_accuracy']),
                  min_group_share_pct=100*statistics.mean(min(q) for q in joints),
                  joint_error=statistics.mean(sum((v-.25)**2 for v in q) for q in joints),
                  effective_clients=math.exp(-sum(s*math.log(s) for s in shares if s)),
                  top9_selection_share_pct=100*sum(sorted(shares)[-9:]))
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite summary metrics')
    return result


def summarize(batch, root=ROOT):
    data = json.loads((batch / 'summary.json').read_text())
    rows = data['runs']
    if data['stage'] != 'formal' or {r['seed'] for r in rows} != set(REFERENCES) or len(rows) != 3:
        raise ValueError('Summary requires exactly the three formal seeds')
    combined = []
    for run in rows:
        seed = run['seed']
        estimated = root / run['checkpoint']
        ref = root / 'checkpoints' / REFERENCES[seed]['DHT']
        verify_config(config_at(estimated), config_at(ref), seed)
        expected_identity = identity(ref)
        for arm, name in REFERENCES[seed].items():
            directory = root / 'checkpoints' / name
            if identity(directory) != expected_identity:
                raise ValueError(f'Previous arms are not paired for seed {seed}')
            combined.append(dict(seed=seed, arm=arm, checkpoint=str(directory.relative_to(root)), **metrics(directory)))
        if identity(estimated) != expected_identity:
            raise ValueError(f'Estimated run is not paired for seed {seed}')
        combined.append(dict(seed=seed, arm='EstimatedDHT', checkpoint=run['checkpoint'], **metrics(estimated)))
    with (batch / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    lines = ['Artifact/identity checks passed; test WGA in %, differences in percentage points.',
             'seed | Random | EstimatedDHT | TrueDHT | Joint | Est-TrueDHT | Joint-Est']
    for seed in REFERENCES:
        scores = {r['arm']: r['test_WGA'] for r in combined if r['seed'] == seed}
        r, e, d, j = [scores[a] for a in ('Random', 'EstimatedDHT', 'DHT', 'Joint')]
        lines.append(f'{seed} | {r:.4f} | {e:.4f} | {d:.4f} | {j:.4f} | {e-d:+.4f} | {j-e:+.4f}')
    metric_names = [k for k in combined[0] if k not in ('seed', 'arm', 'checkpoint')]
    for arm in ('Random', 'EstimatedDHT', 'DHT', 'Joint'):
        lines.append('\n' + arm + ' (mean +/- population std):')
        for metric in metric_names:
            values = [r[metric] for r in combined if r['arm'] == arm]
            lines.append(f'  {metric}: {statistics.mean(values):.6f} +/- {statistics.pstdev(values):.6f}')
    lines.append('Coverage uses true counts ONLY for post-training diagnostics; the estimated selector reads estimated SC/AI/CI.')
    text = '\n'.join(lines)+'\n'
    (batch / 'comparison.txt').write_text(text, encoding='utf-8')
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'smoke', 'formal', 'summary'), default='plan')
    parser.add_argument('--batch-dir', type=Path, help='Existing formal batch for summary')
    args = parser.parse_args()
    if args.stage == 'summary':
        if args.batch_dir is None:
            parser.error('--batch-dir is required for summary')
        summarize(args.batch_dir.resolve())
        return
    from tools.run_email_four import resolved_config
    identifier = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = ROOT / 'run_logs' / 'estimated-dht-gsc' / (args.stage+'-'+identifier)
    seeds = [42] if args.stage == 'smoke' else list(REFERENCES)
    jobs = []
    for seed in seeds:
        tag = f'estimated-dht-gsc-{identifier}-{args.stage}-{seed}'
        command = [sys.executable, '-u', 'flower_train.py', '--config-name='+PROFILE,
                   f'seed={seed}', f'research.training_seed={seed+100}',
                   f'research.validation_seed={seed+1000}', 'research.run_tag='+tag]
        if args.stage == 'smoke':
            command += ['server_opt.rounds=3', 'client_opt.num_steps=2',
                        'client_opt.biased_trainer_steps=2', 'client_opt.left_right_trainer_steps=2']
        jobs.append(dict(seed=seed, tag=tag, command=command))
        print('COMMAND:', ' '.join(command), flush=True)
    if args.stage == 'plan':
        print('Plan only. Formal stage adds exactly three runs; no old arm is retrained.')
        return
    # Check every pairing reference before starting any training.
    base = resolved_config(PROFILE)
    for seed in seeds:
        conf = copy.deepcopy(base)
        conf['seed'] = seed
        conf['research'].update(training_seed=seed+100, validation_seed=seed+1000)
        verify_config(conf, config_at(ROOT/'checkpoints'/REFERENCES[seed]['DHT']), seed)
    if args.stage == 'formal':
        for path in (ROOT/'checkpoints').glob('*/config.yaml'):
            cfg = config_at(path.parent)
            tag = str((cfg.get('research') or {}).get('run_tag', ''))
            if tag.startswith('estimated-dht-gsc-') and '-formal-' in tag:
                raise SystemExit(f'STOP: an estimated formal run already exists at {path.parent}; inspect it, do not automatically rerun.')
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(stage=args.stage, jobs=jobs, runs=[])
    summary_path = output/'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True), encoding='utf-8')
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='0', RAY_DEDUP_LOGS='0')
    for job in jobs:
        started = time.monotonic()
        logfile = output/f"seed{job['seed']}.log"
        with logfile.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors='replace')
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
        if code:
            raise SystemExit(f'STOP: exit code {code}; inspect {logfile}; no automatic retry.')
        matches = [p.parent for p in (ROOT/'checkpoints').glob('*/config.yaml')
                   if (config_at(p.parent).get('research') or {}).get('run_tag') == job['tag']]
        if len(matches) != 1:
            raise RuntimeError('Expected exactly one checkpoint for this run tag')
        directory = matches[0]
        reference = ROOT/'checkpoints'/REFERENCES[job['seed']]['DHT']
        verify_config(config_at(directory), config_at(reference), job['seed'], smoke=args.stage=='smoke')
        if identity(directory) != identity(reference):
            raise RuntimeError('Data membership or initialization differs from the paired true-DHT run')
        log = logfile.read_text(encoding='utf-8')
        if 'DHT collection round 2 completed without a model update' not in log:
            raise RuntimeError('Missing metadata-only round evidence')
        estimation_records = sum('True: [' in line and ' Pred: [' in line for line in log.splitlines())
        if not estimation_records:
            raise RuntimeError('Missing evidence that matrix estimation was executed')
        result = metrics(directory, rounds=3 if args.stage=='smoke' else 201)
        summary['runs'].append(dict(seed=job['seed'], checkpoint=str(directory.relative_to(ROOT)),
                                    elapsed_seconds=time.monotonic()-started, process_exit_code=code,
                                    identity_verified=True, metadata_only_log_verified=True,
                                    matrix_estimation_log_records=estimation_records, **result))
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print('VERIFIED:', directory, flush=True)
    print('Batch:', output, flush=True)
    if args.stage == 'formal':
        summarize(output)


if __name__ == '__main__':
    main()
