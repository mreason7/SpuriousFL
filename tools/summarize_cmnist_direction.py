"""Audit all nine formal runs without loading models or selecting checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

METHODS = {'joint_direction_oracle': 'direction', 'triplets_stochasticmatrix': 'dht', 'random': 'random'}


def model_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('results/cmnist_direction_formal'))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    errors, rows, identities, seen = [], [], {}, set()

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    for config_path in sorted((root / 'checkpoints').glob('*/config.yaml')):
        conf = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        research = conf.get('research') or {}
        if not str(research.get('run_tag', '')).startswith('cmnist-direction-formal-'):
            continue
        directory = config_path.parent
        try:
            seed = int(conf['seed'])
            opt = conf['server_opt']
            arm = METHODS[opt['selection_method']]
            key = seed, arm
            require(key not in seen, f'duplicate run for {key}; do not select one automatically')
            seen.add(key)
            require(seed in (42, 43, 44), 'unexpected data seed')
            require(research['enabled'] and research['training_seed'] == seed + 100 and
                    research['validation_seed'] == seed + 1000, 'seed protocol mismatch')
            require(opt['rounds'] == 201 and opt['num_active_clients'] == 9 and
                    opt['weight_clients'] == 'same' and opt['paper_faithful_pretrain'] and
                    opt['pretrain_rounds'] == 1 and opt['dht_collection_rounds'] == 1,
                    'schedule/aggregation configuration mismatch')
            require(conf['dataset_options']['num_clients'] == 24 and
                    conf['dataset_options']['name'] == 'CMNIST', 'dataset mismatch')
            for name in ('final/torchmodel.pt', 'client_info.json', 'data_manifest.json',
                         'data_manifest.sha256', 'initial_weights.sha256',
                         'final_validation_metrics.json', 'final_test_metrics.json', 'client_weights.csv'):
                require((directory / name).is_file(), f'missing {name}')
            identity = tuple((directory / name).read_text().strip()
                             for name in ('data_manifest.sha256', 'initial_weights.sha256'))
            require(all(identity), 'empty identity hashes')
            require(seed not in identities or identities[seed] == identity, 'data/initialization hashes differ across arms')
            identities[seed] = identity
            weights = np.loadtxt(directory / 'client_weights.csv', delimiter=',')
            require(weights.shape == (201, 24), f'participation shape {weights.shape}')
            require(np.all((weights == 0) | (weights == 1)), 'nonbinary participation')
            require(np.all(weights[:2] == 1) and np.all(weights[2:].sum(1) == 9), 'participation schedule mismatch')
            info = json.loads((directory / 'client_info.json').read_text())
            counts = np.array([[info[str(i)][f'interaction_matrix_{j}'] for j in range(4)] for i in range(24)], dtype=float)
            require(np.isfinite(counts).all() and (counts >= 0).all() and (counts.sum(1) > 0).all(), 'invalid true counts')
            probabilities = counts / counts.sum(1, keepdims=True)
            joints = weights[2:] @ probabilities / 9
            objectives = np.square(joints - .25).sum(1)
            if arm == 'direction':
                traces = [json.loads(line) for line in (directory / 'joint_selection_trace.jsonl').read_text().splitlines()]
                require([t['round'] for t in traces] == list(range(3, 202)), 'direction trace rounds incomplete/duplicated')
                for index, trace in enumerate(traces):
                    ids = list(map(int, trace['selected_ids']))
                    require(len(ids) == len(set(ids)) == 9 and all(0 <= i < 24 for i in ids), 'invalid selected IDs')
                    require(set(ids) == set(np.flatnonzero(weights[index + 2])), 'trace/participation disagreement')
                    require([trace['available_cids'][p] for p in trace['selected_positions']] == trace['selected_ids'], 'position/ID mismatch')
                    require(trace['source'] == 'oracle_joint' and np.allclose(trace['target'], .25), 'source/target mismatch')
                    require(np.allclose(np.asarray(trace['selected_joint']).ravel(), joints[index]), 'joint reconstruction mismatch')
                    require(np.isclose(trace['objective'], objectives[index]) and
                            trace['objective'] <= trace['greedy_objective'] + 1e-12, 'objective mismatch/increase')
            validation = json.loads((directory / 'final_validation_metrics.json').read_text())
            test = json.loads((directory / 'final_test_metrics.json').read_text())
            row = dict(seed=seed, arm=arm, checkpoint=str(directory.relative_to(root)),
                       validation_wga=float(validation['worst_group']), test_wga=float(test['worst_group']),
                       test_accuracy=float(test['test_accuracy']), mean_joint_objective=float(objectives.mean()),
                       mean_min_group_share=float(joints.min(1).mean()),
                       model_sha256=model_hash(directory / 'final/torchmodel.pt'))
            for group in ('y0g0', 'y0g1', 'y1g0', 'y1g1'):
                row[group] = float(test[group])
            require(all(np.isfinite(row[k]) for k in ('validation_wga', 'test_wga', 'test_accuracy')), 'nonfinite metrics')
            require(np.isclose(row['test_wga'], min(row[g] for g in ('y0g0','y0g1','y1g0','y1g1'))), 'WGA/group disagreement')
            rows.append(row)
        except Exception as exc:
            errors.append(f'{directory.name}: {exc}')
    expected = {(s, a) for s in (42, 43, 44) for a in METHODS.values()}
    if seen != expected:
        errors.append(f'run coverage mismatch; missing={sorted(expected-seen)}, extra={sorted(seen-expected)}')
    report = dict(status='FAIL' if errors else 'ARTIFACT_CHECKS_PASS', errors=errors, runs=rows,
                  limitation='Artifact checks do not establish process exit codes or runtime metadata-only logs. Inspect the nine training logs separately.')
    (output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if rows:
        with (output / 'runs.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if errors:
        print('\n'.join(errors))
        raise SystemExit('Audit failed; see ' + str(output / 'audit.json'))
    lines = ['Artifact checks passed. Standard deviations below use ddof=0.',
             'seed | Random test WGA | DHT test WGA | Direction test WGA | Direction-Random | Direction-DHT']
    for seed in (42, 43, 44):
        by_arm = {r['arm']: r for r in rows if r['seed'] == seed}
        r, h, d = (by_arm[a]['test_wga'] for a in ('random', 'dht', 'direction'))
        lines.append(f'{seed} | {r:.4f} | {h:.4f} | {d:.4f} | {d-r:+.4f} | {d-h:+.4f}')
    for arm in ('random', 'dht', 'direction'):
        group = [r for r in rows if r['arm'] == arm]
        stats = []
        for metric in ('validation_wga', 'test_wga', 'test_accuracy', 'mean_joint_objective', 'mean_min_group_share'):
            values = [r[metric] for r in group]
            stats.append(f'{metric}={np.mean(values):.6f} +/- {np.std(values):.6f}')
        lines.append(arm + ': ' + '; '.join(stats))
    lines.append(report['limitation'])
    summary = '\n'.join(lines) + '\n'
    (output / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)
    print('Saved:', output)


if __name__ == '__main__':
    main()
