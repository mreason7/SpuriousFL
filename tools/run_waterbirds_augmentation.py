"""Independent three-seed WaterBirds augmentation condition; never append to baseline."""
import argparse
import os
from pathlib import Path
import sys

import run_email_four as runner


def configure():
    runner.DEFAULT_DATASETS = ('waterbirds_dist',)
    runner.ALL_DATASETS = ('waterbirds_dist',)
    runner.BATCH_PREFIX = 'waterbirds-augmentation'
    runner.CONFIG_SUFFIX = '_aug'
    runner.TRAIN_ENTRY = 'train_waterbirds_augmentation.py'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--baseline-summary', type=Path)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--stage')
    parser.add_argument('--seeds')
    known, _ = parser.parse_known_args()
    if known.batch and not known.batch.name.startswith('waterbirds-augmentation-formal-'):
        raise SystemExit('Use only a waterbirds-augmentation-formal batch; baseline batches are protected.')
    baseline = known.baseline_summary
    if baseline is None:
        baseline = runner.ROOT / 'run_logs/email-four-formal/summary.json'
        if not baseline.is_file():
            baseline = runner.ROOT / 'results/email_four_20261106/server_export/summary.json'
    baseline = baseline.resolve()
    if not baseline.is_file():
        raise SystemExit('Baseline summary missing; provide --baseline-summary PATH.')
    os.environ['WB_AUG_BASELINE'] = str(baseline)
    # Remove only the wrapper argument; let the established runner validate all others.
    args = sys.argv[1:]
    for i, value in enumerate(args):
        if value == '--baseline-summary':
            del args[i:i + 2]
            break
        if value.startswith('--baseline-summary='):
            del args[i]
            break
    if known.seeds is None:
        args += ['--seeds', '42' if known.stage == 'smoke' else '42,3407,2026']
    sys.argv = [sys.argv[0], *args]
    configure()
    # Validate that augmentation is the only configuration change from each baseline.
    for method in ('random', 'feddiverse'):
        base = runner.resolved_config(f'email_waterbirds_dist_{method}')
        aug = runner.resolved_config(f'email_waterbirds_dist_{method}_aug')
        aug['dataset_options']['aug_crop'] = 0
        aug['dataset_options']['aug_horizontal_flip'] = False
        if aug != base:
            raise RuntimeError('Unexpected settings beyond the two augmentation options.')
    runner.main()


if __name__ == '__main__':
    main()
