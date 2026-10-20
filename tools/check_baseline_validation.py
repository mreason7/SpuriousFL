"""Focused CPU checks. Run with the experiment Python on the server too.

Local checks use actual selector ASTs and NumPy pivot tensors. If torch is
installed, also exercise the actual tensor split/pivot code and data subsets.
"""
import ast
from contextlib import contextmanager
from pathlib import Path
import random
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import baseline_validation as bv


def ast_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace


class BaselineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy = staticmethod(ast_functions("src/optimizers/weighting_strategy.py",
                           ["select_3_clients", "select_noreplacement"], {"np": np})["select_noreplacement"])

    def test_guard_preserves_non_degenerate_legacy_trajectory(self):
        rng = np.random.default_rng(17)
        for k, p in [(24, 9), (25, 9), (100, 12)]:
            for seed in range(20):
                matrix = rng.uniform(.01, 1, (3, k))
                np.random.seed(seed)
                expected = self.legacy(matrix.copy(), p)
                np.random.seed(seed)
                actual = bv.guarded_selection(matrix, p)
                self.assertEqual(list(expected), actual)

    def test_guards_zero_collinear_partial_and_all_clients(self):
        matrices = [np.zeros((3, 8)), np.ones((3, 8)),
                    np.array([[0.] * 8, np.arange(1., 9.), np.arange(8., 0., -1.)])]
        for matrix in matrices:
            original = matrix.copy()
            for count in range(9):
                for seed in range(10):
                    np.random.seed(seed)
                    ids = bv.guarded_selection(matrix, count)
                    self.assertEqual(len(ids), count)
                    self.assertEqual(len(set(ids)), count)
                    self.assertTrue(all(0 <= i < 8 for i in ids))
            np.testing.assert_array_equal(original, matrix)

    def test_guard_rejects_invalid_input(self):
        for matrix, count in [(np.ones((2, 4)), 3), (np.ones((3, 4)), 5),
                              (np.ones((3, 4)), -1), (np.ones((3, 4)), 1.5),
                              (np.full((3, 4), np.nan), 3), (np.full((3, 4), -1.), 3)]:
            with self.assertRaises(ValueError):
                bv.guarded_selection(matrix, count)

    def test_pivot_counterexample_and_legacy_preservation(self):
        splits = {0: (SimpleNamespace(tensors=[None, np.array([0] * 90 + [1] * 10)]), None, 1.),
                  1: (SimpleNamespace(tensors=[None, np.array([0] * 50 + [1] * 50)]), None, 1.)}
        self.assertEqual(bv.choose_pivot(splits), 0)
        self.assertEqual(bv.choose_pivot(splits, "min_count_gap"), 1)
        self.assertEqual(bv.choose_pivot({1: splits[1], 0: splits[1]}, "min_count_gap"), 0)

    def test_mapping_for_shuffled_and_non_contiguous_candidates(self):
        self.assertEqual(bv.map_positions([0, 1, 2], ["2", "0", "4", "1"]), ["0", "1", "2"])
        self.assertEqual(bv.map_positions([0, 1, 2], ["2", "0", "4", "1"], "candidate"), ["2", "0", "4"])
        self.assertEqual(bv.map_positions([0, 2], ["7", "9", "11"], "candidate"), ["7", "11"])

    def test_real_manager_mapping_branch(self):
        tree = ast.parse((ROOT / "src/flower_manager.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MyManager")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_sample_impl")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        namespace = dict(np=np, random=random, select_noreplacement=lambda m, p: [0, 1, 2],
                         guarded_selection=lambda m, p: [0, 1, 2], map_positions=bv.map_positions,
                         record_selection=lambda *a: None, print=lambda *a: None,
                         research_enabled=lambda conf: False)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])),
                     "real_manager", "exec"), namespace)
        ids = ["2", "0", "4", "1", "3"]
        instance = SimpleNamespace(clients={i: SimpleNamespace(cid=i) for i in ids},
                                   wait_for=lambda n: True,
                                   conf={"server_opt": {"selection_method": "triplets_stochasticmatrix",
                                                        "client_id_mapping": "candidate"}})
        info = {int(i): dict(SC=.2, AI=.3, CI=.4) for i in ids}
        result = namespace["_sample_impl"](instance, 3, client_info=info)
        self.assertEqual([x.cid for x in result], ids[:3])

    def test_rng_scopes_repeat_but_not_across_rounds_and_restore(self):
        conf = {"research": {"enabled": True, "training_seed": 142}}
        np.random.seed(7)
        random.seed(7)
        expected_numpy, expected_python = np.random.random(), random.random()
        np.random.seed(7)
        random.seed(7)
        with bv.rng_scope(conf, "fit", 3, 1, use_torch=False):
            first = np.random.random(5)
        with bv.rng_scope(conf, "fit", 3, 1, use_torch=False):
            repeated = np.random.random(5)
        with bv.rng_scope(conf, "fit", 4, 1, use_torch=False):
            other = np.random.random(5)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, other))
        self.assertEqual(np.random.random(), expected_numpy)
        self.assertEqual(random.random(), expected_python)
        with bv.rng_scope({}, "disabled"):
            np.random.random()  # no torch import and no reseeding

    def test_validation_disjoint_grouped_and_reproducible(self):
        rows = [(f"id-{y}-{g}-{i}", y, g) for y in range(2) for g in range(2) for i in range(20)]
        train, val = bv.validation_indices(rows, .1, 1042)
        self.assertEqual(len(train), 72)
        self.assertEqual(len(val), 8)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(80)))
        self.assertEqual((train, val), bv.validation_indices(rows, .1, 1042))
        self.assertEqual({(rows[i][1], rows[i][2]) for i in val}, {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_single_factor_config_differences(self):
        expected = {"legacy": {}, "pivot": {"pivot_mode": "min_count_gap"},
                    "mapping": {"client_id_mapping": "candidate"}, "guarded": {"selector_mode": "guarded"}}
        for arm, overrides in expected.items():
            config = yaml.safe_load((ROOT / f"conf/research_gai_{arm}.yaml").read_text())
            self.assertEqual(config.get("server_opt", {}), overrides)
            self.assertEqual(set(config) - {"defaults", "server_opt"}, set())

    def test_torch_runtime_if_available(self):
        try:
            import torch
            from torch.utils.data import TensorDataset, Subset
        except ImportError:
            self.skipTest("PyTorch not installed locally; run this check in server spurious environment")
        ns = ast_functions("src/optimizers/matrix_inference.py", ["split_by_class"],
                           {"torch": torch, "TensorDataset": TensorDataset})
        labels = torch.tensor([0] * 100 + [1] * 100)
        errors = torch.tensor([0] * 90 + [1] * 10 + [0] * 50 + [1] * 50)
        splits = ns["split_by_class"](TensorDataset(torch.zeros(200, 1), errors, labels, labels))
        self.assertEqual(int(bv.choose_pivot(splits)), 0)
        self.assertEqual(int(bv.choose_pivot(splits, "min_count_gap")), 1)
        conf = {"research": {"enabled": True, "training_seed": 142}}
        with bv.rng_scope(conf, "fit", 3, 1):
            first = torch.rand(5)
        with bv.rng_scope(conf, "fit", 3, 1):
            self.assertTrue(torch.equal(first, torch.rand(5)))
        # Verify nested subset composition through the real repository wrapper.
        from src.datasets.dataset_utils import SubsetDataset
        class FakeSpawrious:
            pass
        FakeSpawrious.__name__ = "Spawrious"
        ds = FakeSpawrious()
        ds.subset = np.array([[str(ROOT / "datasets" / f"sample{i}.png"), i % 2, (i // 2) % 2]
                              for i in range(8)], dtype=object)
        subset = SubsetDataset(Subset(ds, [1, 3, 5, 7]), [2, 0])
        # metadata is relative to the launch directory, like the real training entry.
        rows = bv.metadata(subset)
        self.assertTrue(rows[0][0].endswith("sample5.png"))
        self.assertTrue(rows[1][0].endswith("sample1.png"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
