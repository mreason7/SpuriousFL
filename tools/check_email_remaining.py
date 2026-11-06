"""CPU checks for the three remaining datasets, including multiclass metadata."""
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
from run_email_four import (REMAINING_DATASETS, resolved_config, verify, run_summary,
                           load_append_rows, save_extension_plan, summarize)
from check_author_gai import load_classes, SourceStrategy


class RemainingChecks(unittest.TestCase):
    def test_append_preserves_original_batch_and_rejects_duplicates(self):
        root = ROOT / 'results/email_remaining_checks'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            output = Path(tmp)
            environment = dict(pretrained_sha256='same-weights', versions={'torch': '2.3.0'})
            original = dict(jobs=[dict(rounds=200, method='random')], environment=environment)
            rows = [dict(dataset='cmnist_gsc', seed=42, method='random', worst_group=90)]
            (output / 'plan.json').write_text(json.dumps(original))
            (output / 'summary.json').write_text(json.dumps(rows))
            (output / 'aggregate.json').write_text('[]')
            plan_bytes = (output / 'plan.json').read_bytes()
            jobs = [dict(dataset='waterbirds_dist', seed=42, method='random')]
            self.assertEqual(load_append_rows(output, jobs), rows)
            save_extension_plan(output, jobs, environment, 'new-commit', 'test')
            self.assertEqual((output / 'plan.json').read_bytes(), plan_bytes)
            self.assertEqual(json.loads((output / 'summary_before_append_test.json').read_text()), rows)
            self.assertTrue((output / 'extension_plan_test.json').is_file())
            combined = rows + [dict(dataset='waterbirds_dist', seed=42, method='random', worst_group=50)]
            self.assertEqual(len(summarize(combined)), 2)
            with self.assertRaisesRegex(RuntimeError, 'refusing to overwrite'):
                load_append_rows(output, [dict(dataset='cmnist_gsc', seed=42, method='random')])
            (output / 'waterbirds_dist_random_seed42.log').write_text('keep')
            with self.assertRaisesRegex(RuntimeError, 'refusing to overwrite'):
                load_append_rows(output, jobs)
            self.assertEqual((output / 'waterbirds_dist_random_seed42.log').read_text(), 'keep')

    def test_config_matrices_and_matching_controls(self):
        tree = ast.parse((ROOT / 'src/datasets/data_splits.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'split_mode_to_matrix']
        scope = {'List': list}
        exec(compile(tree, 'splits', 'exec'), scope)
        expected = {'waterbirds_dist': (30, 4795, 2, 9), 'spawrious_4': (25, 8800, 4, 9),
                    'spawrious_gci_100': (100, 15600, 2, 12)}
        for dataset in REMAINING_DATASETS:
            fed = resolved_config(f'email_{dataset}_feddiverse')
            rnd = resolved_config(f'email_{dataset}_random')
            for key in ('client_opt', 'model_options', 'dataset_options', 'seed', 'machine'):
                self.assertEqual(fed[key], rnd[key])
            self.assertEqual({k for k in fed['server_opt'] if fed['server_opt'][k] != rnd['server_opt'][k]},
                             {'rounds', 'selection_method', 'client_info'})
            n, samples, classes, selected = expected[dataset]
            matrix = scope['split_mode_to_matrix'](fed)
            self.assertEqual((len(matrix), sum(sum(map(sum, m)) for m in matrix)), (n, samples))
            self.assertTrue(all(len(m) == classes and all(len(row) == 2 for row in m) for m in matrix))
            self.assertEqual(fed['dataset_options']['num_clients'], n)
            self.assertEqual(fed['server_opt']['num_active_clients'], selected)
            self.assertTrue(fed['server_opt']['multiclass'])

    def test_remaining_plan_has_twelve_runs(self):
        value = json.loads(subprocess.check_output([sys.executable,
            str(ROOT / 'tools/run_email_remaining.py'), '--stage', 'plan'], text=True))
        jobs = value['jobs']
        self.assertEqual(len(jobs), 12)
        self.assertEqual({j['dataset'] for j in jobs}, set(REMAINING_DATASETS))
        self.assertEqual({j['seed'] for j in jobs}, {42, 3407})
        self.assertEqual({j['selected_clients'] for j in jobs if j['num_clients'] == 100}, {12})

    def test_100_client_schedules_use_twelve_and_dht_does_not_update(self):
        scope = load_classes()
        tree = ast.parse((ROOT / 'src/author_random.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        scope['MyStrategy'] = SourceStrategy
        exec(compile(tree, 'random', 'exec'), scope)
        root = ROOT / 'results/email_remaining_checks'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            for name, expected in [('AuthorStrategy', [100, 100, 12]), ('RandomStrategy', [100, 12, 12])]:
                strategy = scope[name]()
                strategy.conf = dict(dataset_options=dict(num_clients=100),
                    server_opt=dict(num_active_clients=12), exp_id=tmp)
                strategy.shared_copt_params, strategy.stored_client_data = {}, {}
                strategy.current_weights, strategy.m_t = [0], [0]
                strategy.log_client_info = lambda rnd: None
                manager = S(sample=lambda num_clients, **kwargs: [S(cid=str(i)) for i in range(num_clients)])
                for rnd, n in enumerate(expected, 1):
                    tasks = strategy.configure_fit(rnd, None, manager)
                    self.assertEqual(len(tasks), n)
                    before = copy.deepcopy((strategy.current_weights, strategy.m_t))
                    results = [(c, S(metrics={'cid': int(c.cid)})) for c, _ in tasks]
                    strategy.aggregate_fit(rnd, results, [])
                    if name == 'AuthorStrategy' and rnd == 2:
                        self.assertEqual((strategy.current_weights, strategy.m_t), before)

    def test_waterbirds_global_sample_ids_and_four_class_manifest(self):
        tree = ast.parse((ROOT / 'train_email_four.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('members', 'split_data', 'digest')]
        import hashlib
        source = S(y_array=[0, 1, 0, 1], metadata_array=[[0], [1], [1], [0]])
        bird = type('WaterBirds', (), {})()
        bird.subset = S(dataset=source, indices=[1, 3])
        scope = dict(Path=Path, json=json, hashlib=hashlib)
        exec(compile(tree, 'metadata', 'exec'), scope)
        self.assertEqual(scope['members'](bird), [['waterbirds/1', 1, 1], ['waterbirds/3', 1, 0]])
        dataset = type('Spawrious', (), {})()
        dataset.subset = [[str(ROOT / f'datasets/train{i}.png'), i//2, i%2] for i in range(8)]
        test = type('Spawrious', (), {})()
        test.subset = [[str(ROOT / 'datasets/test.png'), 0, 0]]
        client = type('Client', (), {'__len__': lambda self: 8})()
        client.indices = list(range(8))
        root = ROOT / 'results/email_remaining_checks'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            scope.update(author=S(original_split=lambda ds, conf: [client]),
                split_mode_to_matrix=lambda conf: [[[1, 1]] * 4], loaded={'test': test})
            conf = dict(exp_id=tmp, dataset_options=dict(num_clients=1,num_targets=4,num_groups=2,name='Spawrious'))
            scope['split_data'](dataset, conf)
            manifest = json.loads((Path(tmp) / 'data_manifest.json').read_text())
            self.assertEqual(manifest['clients'][0]['counts'], [[1, 1]] * 4)

    def test_multiclass_footer_includes_all_eight_groups(self):
        metrics = {f'y{y}g{g}': 90-y-g for y in range(4) for g in range(2)}
        metrics.update(test_accuracy=88, worst_group=86)
        text = run_summary(dict(rounds=201,method='feddiverse',dataset='spawrious_4',seed=42,
            num_clients=25, selected_clients=9), metrics, Path('example'), 'formal', 60)
        self.assertIn('y3g1: 86.0000%', text)
        self.assertIn('Worst group(s): y3g1', text)


if __name__ == '__main__':
    unittest.main()
