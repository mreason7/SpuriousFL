"""Compare the fixed three-seed augmentation condition with the frozen baseline."""
import argparse
import csv
import json
from pathlib import Path
import statistics

SEEDS = (42, 3407, 2026)


def index(rows):
    result = {}
    for row in rows:
        if row['dataset'] != 'waterbirds_dist' or row['seed'] not in SEEDS:
            continue
        key = row['seed'], row['method']
        if key in result:
            raise ValueError(f'Duplicate WaterBirds run: {key}')
        result[key] = row
    expected = {(seed, method) for seed in SEEDS for method in ('random', 'feddiverse')}
    if set(result) != expected:
        raise ValueError('All six formal runs are required; partial results are not a complete comparison.')
    return result


def compare(baseline_rows, augmented_rows):
    baseline, augmented = index(baseline_rows), index(augmented_rows)
    records, effects = [], []
    for seed in SEEDS:
        for method in ('random', 'feddiverse'):
            before, after = baseline[seed, method], augmented[seed, method]
            if before['hashes'] != after['hashes']:
                raise ValueError(f'Data/initial weights differ between conditions: {seed}/{method}')
            records.append(dict(seed=seed, method=method,
                baseline_wga=before['worst_group'], augmented_wga=after['worst_group'],
                wga_change=after['worst_group'] - before['worst_group'],
                baseline_acc=before['test_accuracy'], augmented_acc=after['test_accuracy'],
                acc_change=after['test_accuracy'] - before['test_accuracy']))
        old_gain = baseline[seed, 'feddiverse']['worst_group'] - baseline[seed, 'random']['worst_group']
        new_gain = augmented[seed, 'feddiverse']['worst_group'] - augmented[seed, 'random']['worst_group']
        effects.append(dict(seed=seed, baseline_method_gain=old_gain,
                            augmented_method_gain=new_gain, gain_change=new_gain-old_gain))
    aggregate = []
    for method in ('random', 'feddiverse'):
        group = [r for r in records if r['method'] == method]
        value = dict(method=method, n=3)
        for field in ('baseline_wga', 'augmented_wga', 'wga_change',
                      'baseline_acc', 'augmented_acc', 'acc_change'):
            vals = [r[field] for r in group]
            value[field] = dict(mean=statistics.mean(vals), std_population=statistics.pstdev(vals))
        aggregate.append(value)
    return dict(units='percent; differences in percentage points; std ddof=0',
        seeds=list(SEEDS), per_run=records, aggregate=aggregate, paired_method_effects=effects,
        mean_gain_change=statistics.mean(e['gain_change'] for e in effects))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-summary', type=Path, required=True)
    p.add_argument('--aug-batch', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads((args.aug_batch / 'plan.json').read_text())
    if len(plan['jobs']) != 6 or any(j['rounds'] != (200 if j['method'] == 'random' else 201)
            or j['config'] != f"email_waterbirds_dist_{j['method']}_aug" for j in plan['jobs']):
        raise ValueError('Expected the six-run formal augmentation plan, not smoke or baseline runs.')
    result = compare(json.loads(args.baseline_summary.read_text()),
                     json.loads((args.aug_batch / 'summary.json').read_text()))
    (args.aug_batch / 'augmentation_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.aug_batch / 'augmentation_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['per_run'][0]))
        writer.writeheader()
        writer.writerows(result['per_run'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
