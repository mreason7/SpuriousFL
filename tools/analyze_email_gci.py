"""Inspect completed GCI artifacts without loading models or training."""
import argparse
import ast
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def triplet(matrix):
    total = sum(map(sum, matrix))
    joint = [[x / total for x in row] for row in matrix]
    py = [sum(row) for row in joint]
    pa = [sum(joint[y][a] for y in range(2)) for a in range(2)]
    hy, ha = entropy(py), entropy(pa)
    mi = sum(joint[y][a] * math.log(joint[y][a] / (py[y] * pa[a]))
             for y in range(2) for a in range(2) if joint[y][a] > 0)
    return dict(CI=1 - hy / math.log(2), AI=1 - ha / math.log(2),
                SC=2 * mi / (hy + ha) if hy + ha > 0 else 0.0)


def correlation(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    a, b = [v - mx for v in x], [v - my for v in y]
    denominator = math.sqrt(sum(v*v for v in a) * sum(v*v for v in b))
    return sum(i*j for i, j in zip(a, b)) / denominator if denominator > 1e-12 else None


def inspect(root, row):
    directory = Path(row['checkpoint'])
    if not directory.is_absolute():
        directory = root / directory
    manifest = json.loads((directory / 'data_manifest.json').read_text())
    matrices = {int(c['cid']): c['counts'] for c in manifest['clients']}
    if set(matrices) != set(range(24)) or sum(sum(map(sum, m)) for m in matrices.values()) != 4800:
        raise RuntimeError('Expected the 24-client, 4800-sample GCI split')
    kinds = {cid: max(triplet(m), key=triplet(m).get) for cid, m in matrices.items()}
    inverse = [cid for cid, m in matrices.items() if kinds[cid] == 'CI' and sum(m[1]) > sum(m[0])]
    weights = [[float(x) for x in line.split(',')] for line in
               (directory / 'client_weights.csv').read_text().splitlines()]
    method = row['method']
    expected_events = 201 if method == 'feddiverse' else 200
    if len(weights) != expected_events or any(len(v) != 24 for v in weights):
        raise RuntimeError(f'Unexpected participation shape: {directory}')
    # Both methods have exactly 199 selection/update events after the shared first update.
    selections = weights[2:] if method == 'feddiverse' else weights[1:]
    frequency, exposures = Counter(), []
    for event in selections:
        if any(x not in (0, 1) for x in event) or sum(event) != 9:
            raise RuntimeError('Expected nine distinct equally weighted clients')
        selected = [i for i, x in enumerate(event) if x]
        frequency.update(selected)
        p = [sum(matrices[i][y][a] / sum(map(sum, matrices[i])) for i in selected) / 9
             for y in range(2) for a in range(2)]
        exposures.append(dict(class1_share=p[2] + p[3], min_group_share=min(p),
            entropy=entropy(p) / math.log(4),
            type_count=len({kinds[i] for i in selected}),
            **{f'y{y}g{a}': p[2*y+a] for y in range(2) for a in range(2)}))
    total_frequency = sum(frequency.values())
    metrics = json.loads((directory / 'final_test_metrics.json').read_text())
    groups = {k: metrics[k] for k in ('y0g0', 'y0g1', 'y1g0', 'y1g1')}
    worst = min(groups.values())
    report = dict(method=method, seed=row['seed'], checkpoint=str(directory),
        metrics=metrics, worst_groups=[k for k, v in groups.items() if abs(v-worst) < 1e-8],
        selection_updates=len(selections),
        inverse_class_clients={str(cid): frequency[cid] for cid in inverse},
        effective_clients=1 / sum((frequency[i] / total_frequency)**2 for i in matrices),
        never_selected=[i for i in matrices if frequency[i] == 0],
        mean_exposure={k: statistics.mean(v[k] for v in exposures) for k in exposures[0]},
        type_selection_share={k: sum(frequency[i] for i in matrices if kinds[i] == k) / total_frequency
                              for k in ('CI', 'AI', 'SC')},
        frequency={str(i): frequency[i] for i in matrices},
        truth_dht={str(i): triplet(m) for i, m in matrices.items()},
        data_hash=(directory / 'data_manifest.sha256').read_text().strip(),
        initial_hash=(directory / 'initial_weights.sha256').read_text().strip())
    if method == 'feddiverse':
        info = json.loads((directory / 'client_info.json').read_text())
        inferred = {i: {k: float(info[str(i)][k]) for k in ('CI', 'AI', 'SC')} for i in matrices}
        report['dht_estimation'] = {k: dict(
            pearson=correlation([triplet(matrices[i])[k] for i in matrices],
                                [inferred[i][k] for i in matrices]),
            mean_absolute_error=statistics.mean(abs(triplet(matrices[i])[k] - inferred[i][k]) for i in matrices))
            for k in ('CI', 'AI', 'SC')}
        report['predicted_dht'] = inferred
        # Verify scalar invariance for oppositely class-biased clients in this actual manifest.
        positive = [i for i in matrices if kinds[i] == 'CI' and sum(matrices[i][0]) > sum(matrices[i][1])]
        report['class_direction_collision'] = dict(
            class0_biased_clients=positive, class1_biased_clients=inverse,
            true_dht_identical=all(abs(triplet(matrices[i])[k] - triplet(matrices[j])[k]) < 1e-9
                for i in positive for j in inverse for k in ('CI', 'AI', 'SC')))
    log = Path(row['log'])
    if not log.is_absolute():
        log = root / log
    if log.is_file():
        text = log.read_text(errors='replace')
        values = [ast.literal_eval(m) for m in re.findall(r'aggregated eval results (\{[^\n]+\})', text)]
        if values:
            report['evaluation_diagnostic'] = dict(
                parsed_log_records=len(values),
                max_logged_wga=max(float(v['worst_group']) for v in values),
                final_wga=float(metrics['worst_group']),
                note='Diagnostic only; duplicate log handlers may repeat records. Do not select an intermediate checkpoint.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    candidates = sorted((root / 'run_logs').glob('email-four-formal-*'))
    batch = args.batch or (candidates[-1] if candidates else None)
    if batch is None:
        parser.error('No formal batch found; pass --batch')
    rows = json.loads((batch / 'summary.json').read_text())
    chosen = [r for r in rows if r['dataset'] == 'spawrious_gci']
    if len(chosen) != 4 or {(r['seed'], r['method']) for r in chosen} != {
            (s, m) for s in (42, 3407) for m in ('random', 'feddiverse')}:
        raise RuntimeError('Expected all four GCI runs for seeds 42 and 3407')
    reports = [inspect(root, r) for r in chosen]
    for seed in (42, 3407):
        pair = [r for r in reports if r['seed'] == seed]
        if len({r['data_hash'] for r in pair}) != 1 or len({r['initial_hash'] for r in pair}) != 1:
            raise RuntimeError(f'Unmatched data or initialization: seed {seed}')
    output = args.output or batch / 'gci_diagnostic'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'diagnostic.json').write_text(json.dumps(dict(batch=str(batch), runs=reports), indent=2))
    lines = ['GCI diagnosis: actual saved artifacts; 199 selection updates per method.',
             'Descriptive associations only; this analysis does not establish a unique cause.']
    for r in sorted(reports, key=lambda r: (r['seed'], r['method'])):
        e = r['mean_exposure']
        lines += ['', f"seed={r['seed']} method={r['method']}",
            f"ACC={r['metrics']['test_accuracy']:.4f} WGA={r['metrics']['worst_group']:.4f} worst={r['worst_groups']}",
            'Final group accuracies: ' + json.dumps({k: r['metrics'][k] for k in ('y0g0','y0g1','y1g0','y1g1')}),
            f"Inverse class client participation: {r['inverse_class_clients']} / 199",
            f"class1 exposure={e['class1_share']:.4f}; min group exposure={e['min_group_share']:.4f}; normalized entropy={e['entropy']:.4f}",
            'Mean real group exposure: ' + json.dumps({k: e[k] for k in ('y0g0','y0g1','y1g0','y1g1')}),
            f"Effective clients={r['effective_clients']:.3f}; never selected={r['never_selected']}",
            'CI/AI/SC selection shares: ' + json.dumps(r['type_selection_share']),
            'Client frequency: ' + json.dumps(r['frequency'])]
        if 'dht_estimation' in r:
            lines += ['DHT estimation: ' + json.dumps(r['dht_estimation']),
                      'Class-direction collision: ' + json.dumps(r['class_direction_collision'])]
        if 'evaluation_diagnostic' in r:
            lines.append('Evaluation trajectory: ' + json.dumps(r['evaluation_diagnostic']))
    text = '\n'.join(lines) + '\n'
    (output / 'summary.txt').write_text(text)
    print(text)
    print('SAVED', output)


if __name__ == '__main__':
    main()
