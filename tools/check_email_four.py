"""CPU checks for the unified batch; GPU smoke remains required."""
import ast
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_email_four import DATASETS, resolved_config, verify


class UnifiedChecks(unittest.TestCase):
    def test_every_pair_matches_except_method_and_collection_event(self):
        for dataset in DATASETS:
            f = resolved_config(f'email_{dataset}_feddiverse')
            r = resolved_config(f'email_{dataset}_random')
            for key in ('dataset_options', 'client_opt', 'model_options', 'machine', 'seed'):
                self.assertEqual(f[key], r[key])
            self.assertEqual({k for k in f['server_opt'] if f['server_opt'][k] != r['server_opt'][k]},
                             {'rounds', 'client_info', 'selection_method'})
            self.assertFalse(f['research']['enabled'])
            self.assertFalse(f['server_opt']['paper_faithful_pretrain'])
            self.assertEqual(f['server_opt']['optimizer'], 'FedAvgM')

    def test_data_matrices_and_sample_budgets(self):
        tree = ast.parse((ROOT / 'src/datasets/data_splits.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'split_mode_to_matrix']
        scope = {'List': list}
        exec(compile(tree, 'data_splits.py', 'exec'), scope)
        for dataset in DATASETS:
            conf = resolved_config(f'email_{dataset}_feddiverse')
            matrix = scope['split_mode_to_matrix'](conf)
            expected_clients = 25 if dataset == 'spawrious_gai' else 24
            self.assertEqual(len(matrix), expected_clients)
            self.assertEqual(conf['dataset_options']['num_clients'], expected_clients)
            self.assertEqual(sum(sum(map(sum, m)) for m in matrix),
                             4600 if dataset == 'spawrious_gai' else 4800)

    def test_verifier_accepts_25_client_schedule_and_rejects_wrong_count(self):
        import json
        import tempfile
        temporary_root = ROOT / 'results/email_four_checks'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as tmp:
            directory = Path(tmp)
            self.assertTrue(directory.resolve().is_relative_to(temporary_root.resolve()))
            (directory / 'final').mkdir()
            for name in ('config.yaml', 'final/torchmodel.pt', 'data_manifest.json',
                         'data_manifest.sha256', 'initial_weights.sha256', 'unified_protocol.json'):
                (directory / name).touch()
            (directory / 'final_test_metrics.json').write_text(json.dumps({'worst_group': 80.0}))
            for method, counts in [('feddiverse', [25, 25, 9]), ('random', [25, 9, 9])]:
                (directory / 'client_weights.csv').write_text('\n'.join(
                    ','.join(str(int(i < count)) for i in range(25)) for count in counts))
                (directory / 'schedule.jsonl').write_text('\n'.join(json.dumps(dict(
                    round=r, phase='aggregated', model_update=not (method == 'feddiverse' and r == 2)))
                    for r in range(1, 4)))
                self.assertEqual(verify(directory, 3, method, 25)['worst_group'], 80.0)
                with self.assertRaisesRegex(RuntimeError, 'participation'):
                    verify(directory, 3, method, 24)

    def test_shared_seed_is_set_only_once_per_process(self):
        tree = ast.parse((ROOT / 'src/author_protocol.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'seed_process']
        calls = []
        from types import SimpleNamespace as S
        scope = dict(_seeded=set(), os=S(environ={'AUTHOR_SEED_MODE': 'global'}, getpid=lambda: 1),
            random=S(seed=lambda s: calls.append(('python', s))),
            np=S(random=S(seed=lambda s: calls.append(('numpy', s)))),
            torch=S(manual_seed=lambda s: calls.append(('torch', s)),
                    cuda=S(is_available=lambda: True, manual_seed_all=lambda s: calls.append(('cuda', s)))))
        exec(compile(tree, 'author_protocol.py', 'exec'), scope)
        scope['seed_process']({'seed': 3407})
        scope['seed_process']({'seed': 3407})
        self.assertEqual(calls, [(x, 3407) for x in ('python', 'numpy', 'torch', 'cuda')])

    def test_plan_contains_all_sixteen_unique_jobs(self):
        import json
        value = json.loads(subprocess.check_output([sys.executable, str(ROOT / 'tools/run_email_four.py'),
                                                     '--stage', 'plan'], text=True))
        jobs = value['jobs']
        self.assertEqual(len(jobs), 16)
        self.assertEqual({j['seed'] for j in jobs}, {42, 3407})
        self.assertEqual(len({(j['dataset'], j['seed'], j['method']) for j in jobs}), 16)


if __name__ == '__main__':
    unittest.main()
