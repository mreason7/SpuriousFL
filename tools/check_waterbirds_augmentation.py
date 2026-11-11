"""CPU checks for condition isolation, initialization guards and paired comparisons."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as S
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_email_four import resolved_config
from compare_waterbirds_augmentation import compare


def fixture_rows():
    return [dict(dataset='waterbirds_dist', seed=seed, method=method,
                 worst_group=40 + offset + (6 if method == 'feddiverse' else 0),
                 test_accuracy=80 + offset,
                 hashes={'data_manifest.sha256': f'data-{seed}',
                         'initial_weights.sha256': f'weights-{seed}'})
            for offset, seed in enumerate((42, 3407, 2026))
            for method in ('random', 'feddiverse')]


class AugmentationChecks(unittest.TestCase):
    def test_six_job_plan_and_existing_runner_unchanged(self):
        value = json.loads(subprocess.check_output([sys.executable,
            str(ROOT / 'tools/run_waterbirds_augmentation.py'), '--stage', 'plan'], text=True))
        jobs = value['jobs']
        self.assertEqual(len(jobs), 6)
        self.assertEqual({j['seed'] for j in jobs}, {42, 3407, 2026})
        self.assertTrue(all(j['command'][1] == 'train_waterbirds_augmentation.py' for j in jobs))
        baseline = json.loads(subprocess.check_output([sys.executable,
            str(ROOT / 'tools/run_email_four.py'), '--stage', 'plan'], text=True))
        self.assertEqual(len(baseline['jobs']), 16)
        self.assertTrue(all(j['command'][1] == 'train_email_four.py' for j in baseline['jobs']))

    def test_only_two_options_change_and_methods_match(self):
        configs = {}
        for method in ('random', 'feddiverse'):
            base = resolved_config(f'email_waterbirds_dist_{method}')
            aug = resolved_config(f'email_waterbirds_dist_{method}_aug')
            self.assertEqual(aug['dataset_options']['aug_crop'], 24)
            self.assertIs(aug['dataset_options']['aug_horizontal_flip'], True)
            configs[method] = copy.deepcopy(aug)
            aug['dataset_options'].update(aug_crop=0, aug_horizontal_flip=False)
            self.assertEqual(aug, base)
        for key in ('client_opt', 'dataset_options', 'model_options', 'machine', 'seed'):
            self.assertEqual(configs['random'][key], configs['feddiverse'][key])

    def test_baseline_batch_is_rejected(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/run_waterbirds_augmentation.py'),
            '--stage', 'formal', '--batch', str(ROOT / 'run_logs/email-four-formal')],
            capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('baseline batches are protected', result.stderr)

    def test_initial_guard_rejects_a_different_initialization(self):
        tree = ast.parse((ROOT / 'train_waterbirds_augmentation.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'checked_initial']
        rows = fixture_rows()
        baseline = next(r for r in rows if r['dataset'] == 'waterbirds_dist'
                        and r['seed'] == 42 and r['method'] == 'random')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = out / 'baseline.json'
            summary.write_text(json.dumps([baseline]))
            for name, value in baseline['hashes'].items():
                (out / name).write_text(value)
            scope = dict(original_initial=lambda *a, **kw: None, Path=Path, json=json,
                os=S(environ={'WB_AUG_BASELINE':str(summary)}), __file__=__file__,
                hashlib=__import__('hashlib'))
            exec(compile(tree, 'initial_guard', 'exec'), scope)
            conf = resolved_config('email_waterbirds_dist_random_aug')
            conf['exp_id'] = str(out)
            scope['checked_initial'](S(conf=conf))
            self.assertTrue((out / 'augmentation_control.json').is_file())
            (out / 'initial_weights.sha256').write_text('different')
            with self.assertRaisesRegex(RuntimeError, 'stopped before training'):
                scope['checked_initial'](S(conf=conf))

    def test_comparison_uses_paired_effect_and_rejects_missing_or_mismatched_runs(self):
        rows = fixture_rows()
        augmented = [copy.deepcopy(r) for r in rows if r['dataset'] == 'waterbirds_dist']
        for row in augmented:
            row['worst_group'] += 2 if row['method'] == 'feddiverse' else 1
        value = compare(rows, augmented)
        self.assertAlmostEqual(value['mean_gain_change'], 1)
        with self.assertRaisesRegex(ValueError, 'All six formal'):
            compare(rows, augmented[:-1])
        augmented[0]['hashes']['initial_weights.sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'weights differ'):
            compare(rows, augmented)


if __name__ == '__main__':
    unittest.main()
