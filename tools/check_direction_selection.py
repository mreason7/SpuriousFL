"""CPU checks: python tools/check_direction_selection.py"""
import ast
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.direction_selection import select_joint
from src.baseline_validation import map_positions


class DirectionChecks(unittest.TestCase):
    def test_opposite_correlations_are_complementary(self):
        a = [[90, 10], [10, 90]]
        b = [[10, 90], [90, 10]]
        for seed in range(20):
            np.random.seed(seed)
            selected, info = select_joint([a, a, b], 2)
            self.assertIn(2, selected)
            self.assertAlmostEqual(info['objective'], 0)
            np.testing.assert_allclose(info['selected_joint'], np.full((2, 2), .25))

    def test_normalization_budget_reproducibility_and_swap_improvement(self):
        counts = np.random.default_rng(7).integers(0, 100, (24, 2, 2))
        original = counts.copy()
        np.random.seed(42)
        selected, info = select_joint(counts, 9)
        np.random.seed(42)
        repeated, repeat_info = select_joint(counts * np.arange(1, 25)[:, None, None], 9)
        self.assertEqual(selected, repeated)
        self.assertEqual(info, repeat_info)
        self.assertEqual(len(set(selected)), 9)
        self.assertLessEqual(info['objective'], info['greedy_objective'] + 1e-12)
        p = counts / counts.sum(axis=(1, 2), keepdims=True)
        np.testing.assert_allclose(info['selected_joint'], p[selected].mean(axis=0))
        for outgoing in selected:
            for incoming in set(range(24)) - set(selected):
                alternative = [incoming if i == outgoing else i for i in selected]
                self.assertGreaterEqual(float(((p[alternative].mean(0) - .25) ** 2).sum()), info['objective'] - 1e-12)
        np.testing.assert_array_equal(counts, original)

    def test_ties_all_clients_and_invalid_inputs(self):
        for count in (1, 9, 24):
            selected, _ = select_joint(np.ones((24, 2, 2)), count)
            self.assertEqual(len(set(selected)), count)
        for counts, count in [(np.zeros((3, 2, 2)), 2), (np.ones((3, 2)), 2),
                              (np.full((3, 2, 2), np.nan), 2), (np.full((3, 2, 2), -1), 2),
                              (np.ones((3, 2, 2)), 4), (np.ones((3, 2, 2)), 0),
                              (np.ones((3, 2, 2)), True)]:
            with self.assertRaises(ValueError):
                select_joint(counts, count)
        with self.assertRaises(ValueError):
            select_joint(np.ones((3, 2, 2)), 2, target=[[1, 1], [1, 1]])

    def test_real_manager_uses_true_counts_and_candidate_ids(self):
        tree = ast.parse((ROOT / 'src/flower_manager.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MyManager')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_sample_impl')
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        traces = []
        ns = dict(np=np, random=random, select_joint=select_joint, map_positions=map_positions,
                  research_enabled=lambda conf: True, record_selection=lambda *args: None,
                  record_joint_selection=lambda *args: traces.append(args), print=lambda *args: None)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), 'manager', 'exec'), ns)
        ids = ['7', '11', '3']
        counts = [[90, 10, 10, 90], [90, 10, 10, 90], [10, 90, 90, 10]]
        info = {int(cid): {f'interaction_matrix_{j}': v for j, v in enumerate(row)} for cid, row in zip(ids, counts)}
        conf = dict(server_opt=dict(selection_method='joint_direction_oracle', weight_clients='same', client_info='joint_oracle'),
                    dataset_options=dict(num_targets=2, num_groups=2))
        instance = SimpleNamespace(conf=conf, clients={i: SimpleNamespace(cid=i) for i in ids}, wait_for=lambda n: None)
        result = ns['_sample_impl'](instance, 2, client_info=info, server_round=3)
        self.assertIn('3', [c.cid for c in result])
        self.assertEqual(len(traces), 1)
        self.assertAlmostEqual(traces[0][-1]['objective'], 0)
        conf['server_opt']['client_info'] = 'triplets_Npredicted'
        with self.assertRaises(ValueError):
            ns['_sample_impl'](instance, 2, client_info=info, server_round=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
