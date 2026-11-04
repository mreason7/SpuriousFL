"""Dependency-free protocol tests; actual GPU smoke is a separate required step."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT


class SourceStrategy:
    def configure_fit(self, rnd, parameters, manager):
        count = 24 if rnd == 1 else 9
        return [(c, SimpleNamespace(config={'round': rnd}))
                for c in manager.sample(num_clients=count)]

    def aggregate_fit(self, rnd, results, failures):
        self.current_weights[0] += 1
        self.m_t[0] += 2
        return self.current_weights, {}

    def update_stored_client_info(self, results, server_round):
        self.stored_client_data = {r.metrics['cid']: dict(SC=1, CI=2, AI=3) for _, r in results}

    def log_client_weights(self, results):
        pass


class SourceClient:
    def fit(self, weights, config):
        self.local_updates += 1
        return 'source-fit'


def load_classes():
    tree = ast.parse((BUNDLE / 'src/author_protocol.py').read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = dict(SourceStrategy=SourceStrategy, SourceClient=SourceClient, copy=copy,
        json=json, os=os, Path=Path, seed_process=lambda conf: None,
        FitIns=lambda parameters, config: SimpleNamespace(parameters=parameters, config=config),
        ndarrays_to_parameters=lambda weights: tuple(weights),
        model_utils=SimpleNamespace(get_weights=lambda model: model))
    exec(compile(tree, 'author_protocol.py', 'exec'), namespace)
    return namespace


class ProtocolTests(unittest.TestCase):
    def test_historical_files_match_commit(self):
        historical = subprocess.check_output(['git', 'show',
            '3c92e8a5a80be836041e1f43bd921ed862156e67:src/datasets/data_splits.py'], cwd=ROOT, text=True)
        tree = ast.parse(historical)
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'split_mode_to_matrix']
        scope = {}
        exec(compile(tree, 'historical_split.py', 'exec'), scope)
        expected = scope['split_mode_to_matrix']('spawrious_GAI_2')
        self.assertEqual(self.current_split('spawrious_GAI_2_author_20241114'), expected)

    @staticmethod
    def current_split(name):
        tree = ast.parse((ROOT / 'src/datasets/data_splits.py').read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'split_mode_to_matrix']
        scope = {'List': list}
        exec(compile(tree, 'data_splits.py', 'exec'), scope)
        return scope['split_mode_to_matrix']({'seed': 42, 'dataset_options': {'split_mode': name}})

    def test_actual_historical_split(self):
        matrices = self.current_split('spawrious_GAI_2_author_20241114')
        self.assertEqual(len(matrices), 24)
        self.assertEqual([sum(map(sum, m)) for m in matrices], [70] * 4 + [242] * 20)
        self.assertEqual(sum(sum(map(sum, m)) for m in matrices), 5120)
        old = self.current_split('spawrious_GAI_2')
        self.assertEqual(len(old), 25)
        self.assertEqual(sum(sum(map(sum, m)) for m in old), 4600)

    def test_email_budget_and_metadata_preserves_weights_and_momentum(self):
        cls = load_classes()['AuthorStrategy']
        manager = SimpleNamespace(sample=lambda num_clients, **kwargs:
            [SimpleNamespace(cid=str(i)) for i in range(num_clients)])
        temporary_root = ROOT / 'results/author_gai_20261104_checks'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as tmp, patch.dict(os.environ, AUTHOR_PROTOCOL='email'):
            self.assertTrue(Path(tmp).resolve().is_relative_to(temporary_root.resolve()))
            strategy = cls()
            strategy.conf = dict(dataset_options=dict(num_clients=24), exp_id=tmp)
            strategy.shared_copt_params = {}
            strategy.stored_client_data = {}
            strategy.current_weights, strategy.m_t = [0], [0]
            participation = []
            for rnd in range(1, 202):
                tasks = strategy.configure_fit(rnd, None, manager)
                participation.append(len(tasks))
                before = copy.deepcopy((strategy.current_weights, strategy.m_t))
                results = [(c, SimpleNamespace(metrics={'cid': int(c.cid)})) for c, _ in tasks]
                strategy.aggregate_fit(rnd, results, [])
                if rnd == 2:
                    self.assertEqual((strategy.current_weights, strategy.m_t), before)
                    self.assertTrue(all(t.config['metadata_only'] for _, t in tasks))
                    self.assertEqual(len(strategy.stored_client_data), 24)
            self.assertEqual(participation, [24, 24] + [9] * 199)
            self.assertEqual(strategy.current_weights, [200])
            self.assertEqual(strategy.m_t, [400])
            with self.assertRaisesRegex(RuntimeError, 'Failed/missing'):
                strategy.aggregate_fit(201, results[:-1], [])

    def test_metadata_client_does_not_train(self):
        client = load_classes()['AuthorClient']()
        client.cid, client.train_len, client.local_updates = 7, 242, 0
        client.set_parameters = lambda weights, config: setattr(client, 'model', weights)
        seen = []
        def share(opt, metrics, before_train):
            seen.append((client.model, before_train))
            return dict(metrics, SC=1, CI=2, AI=3)
        client.share_client_params_once = share
        weights, count, metrics = client.fit('pretrained-global-after-round-1', {'metadata_only': True})
        self.assertEqual(weights, 'pretrained-global-after-round-1')
        self.assertEqual(seen, [('pretrained-global-after-round-1', True)])
        self.assertEqual(client.local_updates, 0)
        self.assertEqual(count, 1)
        self.assertEqual(metrics['cid'], 7)
        self.assertEqual(client.fit(None, {}), 'source-fit')
        self.assertEqual(client.local_updates, 1)


if __name__ == '__main__':
    unittest.main()
