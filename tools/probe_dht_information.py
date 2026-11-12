"""CPU-only information probe using archived TRAINING group counts.

No images, models, training, test metrics, or third-party packages are needed.
Sampled subsets are feasible references, not globally optimal solutions.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    'cmnist_gsc': 'cmnist_gsc_official_rng_v1',
    'spawrious_gsc': 'spawrious_gsc_official_rng_v1',
    'spawrious_gci': 'spawrious_gci_official_rng_v1',
    'spawrious_gai': 'spawrious_gai_official_rng_v1',
}


def probabilities(counts):
    if len(counts) != 4 or any(not math.isfinite(x) or x < 0 for x in counts) or sum(counts) <= 0:
        raise ValueError('Expected four finite nonnegative counts with positive total')
    return tuple(x / sum(counts) for x in counts)


def entropy(values):
    return -sum(x * math.log(x) for x in values if x > 0)


def dht(p):
    rows = (p[0] + p[1], p[2] + p[3])
    cols = (p[0] + p[2], p[1] + p[3])
    hy, ha = entropy(rows), entropy(cols)
    mi = sum(p[2*y+a] * math.log(p[2*y+a] / (rows[y] * cols[a]))
             for y in range(2) for a in range(2) if p[2*y+a] > 0)
    return (1 - hy / math.log(2), 1 - ha / math.log(2),
            2 * mi / (hy + ha) if hy + ha > 0 else 0.0)


def objective(p):
    return sum((x - .25) ** 2 for x in p)


def self_check():
    a = probabilities((90, 10, 10, 90))
    b = probabilities((10, 90, 90, 10))
    assert all(abs(x-y) < 1e-12 for x, y in zip(dht(a), dht(b)))
    assert objective(tuple((x+y)/2 for x, y in zip(a, b))) < 1e-12
    assert objective(a) > .1
    assert probabilities((9, 1, 1, 9)) == a


def load_counts(root, dataset):
    paths = sorted((root / 'checkpoints/local_backup' / DATASETS[dataset]).glob('checkpoints/*/client_info.json'))
    if not paths:
        raise ValueError(f'No archived client_info.json for {dataset}')
    reference, sources = None, []
    for path in paths:
        raw = path.read_bytes()
        info = json.loads(raw)
        ids = sorted(map(int, info))
        counts = [[info[str(i)][f'interaction_matrix_{j}'] for j in range(4)] for i in ids]
        if ids != list(range(len(ids))):
            raise ValueError(f'Noncontiguous client IDs: {path}')
        for row in counts:
            probabilities(row)
        if reference is not None and counts != reference:
            raise ValueError(f'Archived training counts differ within {dataset}; inspect separately: {path}')
        reference = counts
        sources.append(dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest()))
    return reference, sources


def probe(counts, budget, samples, seed):
    p = [probabilities(row) for row in counts]
    triplets = [dht(row) for row in p]
    # Rounding is only for finding candidates; verify actual numeric equality below.
    signatures = [tuple(round(x, 10) for x in row) for row in triplets]
    collisions = []
    for i, j in itertools.combinations(range(len(p)), 2):
        if max(abs(x-y) for x, y in zip(triplets[i], triplets[j])) < 1e-10:
            tv = .5 * sum(abs(x-y) for x, y in zip(p[i], p[j]))
            if tv > 1e-10:
                collisions.append(dict(ids=[i, j], tv=tv, dht=triplets[i],
                                       joint_distributions=[p[i], p[j]]))
    rng = random.Random(seed)
    groups = defaultdict(list)
    trials = []
    for _ in range(samples):
        ids = sorted(rng.sample(range(len(p)), budget))
        joint = tuple(sum(p[i][g] for i in ids) / budget for g in range(4))
        row = dict(ids=ids, joint=joint, objective=objective(joint), min_group_share=min(joint))
        groups[tuple(sorted(signatures[i] for i in ids))].append(row)
        trials.append(row)
    examples = []
    for members in groups.values():
        lo = min(members, key=lambda r: r['objective'])
        hi = max(members, key=lambda r: r['objective'])
        if hi['objective'] - lo['objective'] > 1e-10:
            # Verify the two subsets really expose the same sorted DHT values.
            left = sorted(triplets[i] for i in lo['ids'])
            right = sorted(triplets[i] for i in hi['ids'])
            if all(max(abs(a-b) for a, b in zip(x, y)) < 1e-10 for x, y in zip(left, right)):
                examples.append(dict(gap=hi['objective']-lo['objective'], lower_error=lo, higher_error=hi))
    return dict(clients=len(p), training_samples=sum(map(sum, counts)), budget=budget,
                draws=samples, unique_subsets=len({tuple(r['ids']) for r in trials}),
                equal_dht_different_distribution_pairs=len(collisions),
                largest_pair_collision=max(collisions, key=lambda r: r['tv'], default=None),
                random_mean_objective=statistics.mean(r['objective'] for r in trials),
                random_mean_min_group_share=statistics.mean(r['min_group_share'] for r in trials),
                best_sampled_subset=min(trials, key=lambda r: r['objective']),
                same_dht_subset_counterexample=max(examples, key=lambda r: r['gap'], default=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--budget', type=int, default=9)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.budget < 1:
        parser.error('samples and budget must be positive')
    self_check()
    root = args.root.resolve()
    results = {}
    for name in args.datasets:
        counts, sources = load_counts(root, name)
        if args.budget > len(counts):
            parser.error(f'budget exceeds number of clients in {name}')
        results[name] = dict(**probe(counts, args.budget, args.samples, args.seed), sources=sources)
    output = args.output or root / 'results' / ('dht_information_probe_' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True, exist_ok=False)
    limitations = [
        'Input: archived historical TRAINING group counts; not the newer author-email protocol.',
        'DHT order is CI, AI, SC. Joint order is y0g0, y0g1, y1g0, y1g1.',
        'Joint exposure is the equal-client mean of each normalized client distribution.',
        'Uniform joint target is a prespecified diagnostic; no test labels/metrics are read.',
        'This is a sampled subset search, not a global optimum or a training performance upper bound.',
        'Repeated draws are Monte Carlo samples, not independent training seeds.',
        'A uniform joint distribution may be infeasible for a given dataset or budget.',
        'Coverage is a statistical proxy; actual SGD exposure and WGA require separate training experiments.',
        'No raw images, training checkpoints, test metrics, or historical files were modified.',
    ]
    report = dict(seed=args.seed, self_checks='PASS', datasets=results, limitations=limitations)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    rows, lines = [], ['DHT information probe: structural evidence only; NO new model training.']
    for name, item in results.items():
        example = item['same_dht_subset_counterexample']
        best = item['best_sampled_subset']
        row = dict(dataset=name, clients=item['clients'], equal_dht_different_joint_pairs=item['equal_dht_different_distribution_pairs'],
                   random_mean_error=item['random_mean_objective'], best_sampled_error=best['objective'],
                   random_mean_min_group_share=item['random_mean_min_group_share'], best_subset_min_group_share=best['min_group_share'],
                   same_dht_subset_error_gap=example['gap'] if example else 0.0)
        rows.append(row)
        lines.append(f"{name}: equal-DHT/different-joint pairs={row['equal_dht_different_joint_pairs']}; "
                     f"random mean error={row['random_mean_error']:.6f}; sampled best={row['best_sampled_error']:.6f}; "
                     f"same-DHT subset error gap={row['same_dht_subset_error_gap']:.6f}")
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines.extend(['', *limitations])
    (output / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('Saved:', output)


if __name__ == '__main__':
    main()
