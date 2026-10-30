"""Read audited formal runs; describe coverage and write a hash snapshot.

No training, tuning, checkpoint selection or deletion. The snapshot is an
integrity manifest, not an independent backup of the checkpoint files.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from summarize_cmnist_direction import model_hash


def selection_stats(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (199, 24) or not np.isin(weights, [0, 1]).all() or not (weights.sum(1) == 9).all():
        raise ValueError('Expected 199 selection rounds with 9 of 24 clients')
    counts = weights.sum(0)
    share = counts / counts.sum()
    nonzero = share[share > 0]
    sets = [tuple(np.flatnonzero(row)) for row in weights]
    intersection = (weights[1:] * weights[:-1]).sum(1)
    return dict(clients_ever_selected=int((counts > 0).sum()),
                effective_clients=float(np.exp(-(nonzero * np.log(nonzero)).sum())),
                top9_selection_share=float(np.sort(share)[-9:].sum()),
                min_client_rounds=int(counts.min()), max_client_rounds=int(counts.max()),
                distinct_subsets=len(set(sets)),
                mean_adjacent_jaccard=float(np.mean(intersection / (18 - intersection))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--audit', type=Path, default=Path('results/cmnist_direction_formal/audit.json'))
    parser.add_argument('--output', type=Path, default=Path('results/cmnist_direction_formal/selection_analysis'))
    args = parser.parse_args()
    root = args.root.resolve()
    audit_path = args.audit if args.audit.is_absolute() else root / args.audit
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise SystemExit('Output already exists; use --output with a new path to preserve snapshots')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['status'] != 'ARTIFACT_CHECKS_PASS' or len(audit['runs']) != 9:
        raise SystemExit('Run summarize_cmnist_direction.py successfully first')
    rows, clients, snapshot = [], [], {}
    for run in audit['runs']:
        directory = (root / run['checkpoint']).resolve()
        if root not in directory.parents:
            raise ValueError('Checkpoint must remain inside repository')
        # Capture every file in the run, including final model, and later verify
        # it did not change relative to the source audit.
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                snapshot[str(path.relative_to(root))] = model_hash(path)
        if snapshot[str((directory / 'final/torchmodel.pt').relative_to(root))] != run['model_sha256']:
            raise ValueError('Model changed since audit: ' + str(directory))
        weights = np.loadtxt(directory / 'client_weights.csv', delimiter=',')[2:]
        stats = selection_stats(weights)
        rows.append(dict(seed=run['seed'], arm=run['arm'], **stats))
        info = json.loads((directory / 'client_info.json').read_text())
        counts = np.array([[info[str(i)][f'interaction_matrix_{j}'] for j in range(4)] for i in range(24)], dtype=float)
        p = counts / counts.sum(1, keepdims=True)
        for cid, times in enumerate(weights.sum(0).astype(int)):
            # Association agreement is descriptive, not a marginal-adjusted correlation.
            clients.append(dict(seed=run['seed'], arm=run['arm'], cid=cid,
                                selected_rounds=int(times), round_fraction=float(times / 199),
                                sample_exposures=int(times * counts[cid].sum()),
                                p00=p[cid, 0], p01=p[cid, 1], p10=p[cid, 2], p11=p[cid, 3],
                                class_bias=p[cid, 2]+p[cid, 3]-p[cid, 0]-p[cid, 1],
                                attribute_bias=p[cid, 1]+p[cid, 3]-p[cid, 0]-p[cid, 2],
                                agreement=p[cid, 0]+p[cid, 3]-p[cid, 1]-p[cid, 2]))
    output.mkdir(parents=True)
    for name, data in [('selection_summary.csv', rows), ('client_frequency.csv', clients)]:
        with (output / name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    # Retain full audit evidence so a later regenerated audit cannot replace it silently.
    (output / 'source_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    (output / 'snapshot_sha256.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    lines = ['seed | arm | clients used | effective clients | top9 share | distinct subsets | adjacent Jaccard']
    for row in rows:
        lines.append(f"{row['seed']} | {row['arm']} | {row['clients_ever_selected']} | "
                     f"{row['effective_clients']:.2f} | {row['top9_selection_share']:.3f} | "
                     f"{row['distinct_subsets']} | {row['mean_adjacent_jaccard']:.3f}")
    lines.extend(['Only rounds 3-201 are counted.',
                  'Effective clients: exp(selection-share entropy), at most 24.',
                  'Top9 share: selection slots assigned to the nine most selected clients; uniform reference 9/24=.375.',
                  'Jaccard describes adjacent subset overlap, not sample quality or update conflict.',
                  'Snapshot is a hash manifest, not an independent backup; copy checkpoints and logs separately.',
                  'Process exit codes and metadata-only runtime evidence still require training logs.'])
    summary = '\n'.join(lines) + '\n'
    (output / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)
    print('Saved:', output)


if __name__ == '__main__':
    main()
