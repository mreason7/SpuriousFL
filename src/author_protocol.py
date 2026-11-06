"""Opt-in author-email reconstruction on the current repository branch.

Use the historical GAI matrix and legacy N estimator/selector settings, with
one metadata-only event after update 1. Seeding is recorded separately.
"""
import copy
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from flwr.common import FitIns, ndarrays_to_parameters
from src.flower_client import FlowerClient as SourceClient
from src.flower_strategy import MyStrategy as SourceStrategy
from src.models import model_utils

_seeded = set()


def seed_process(conf):
    """One shared seed per process, without per-round/client derived seeds."""
    if os.environ.get('AUTHOR_SEED_MODE') != 'global':
        return
    seed = int(conf['seed'])
    key = (os.getpid(), seed)
    if key in _seeded:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _seeded.add(key)


class AuthorClient(SourceClient):
    def init_model(self):
        seed_process(self.conf)
        return super().init_model()

    def fit(self, weights, config):
        if not config.get('metadata_only', False):
            return super().fit(weights, config)
        self.set_parameters(weights, config)
        metrics = {'cid': self.cid, 'datasize': self.train_len, 'loss': 0.0}
        metrics = self.share_client_params_once(None, metrics, before_train=True)
        return model_utils.get_weights(self.model), 1, metrics


class AuthorStrategy(SourceStrategy):
    def configure_fit(self, server_round, parameters, client_manager):
        count = (self.conf['dataset_options']['num_clients'] if server_round <= 2
                 else self.conf.get('server_opt', {}).get('num_active_clients', 9))
        clients = client_manager.sample(num_clients=count, min_num_clients=count,
                                       client_info=self.stored_client_data, server_round=server_round)
        self.client_update_requested = [int(c.cid) for c in clients] if server_round == 2 else []
        tasks = []
        for client in clients:
            config = copy.deepcopy(self.shared_copt_params)
            config.update(round=server_round, update_info=server_round == 2,
                          metadata_only=server_round == 2)
            tasks.append((client, FitIns(parameters, config)))
        self.expected_ids = {str(c.cid) for c, _ in tasks}
        if len(tasks) != count:
            raise RuntimeError('Incomplete client selection')
        self._event(server_round, 'configured', client_ids=sorted(self.expected_ids, key=int),
                    metadata_only=server_round == 2)
        return tasks

    def aggregate_fit(self, server_round, results, failures):
        actual = {str(res.metrics['cid']) for _, res in results}
        if failures or actual != self.expected_ids or len(results) != len(self.expected_ids):
            raise RuntimeError('Failed/missing/duplicate clients; stop instead of accepting partial round')
        if server_round == 2:
            self.update_stored_client_info(results, server_round)
            for value in self.stored_client_data.values():
                if not all(k in value for k in ('SC', 'CI', 'AI')):
                    raise RuntimeError('Missing DHT')
            self.log_client_weights(results)
            self.train_metrics_aggregated = {'train_loss': 0.0, 'train_lossStd': 0.0}
            self._event(server_round, 'aggregated', model_update=False)
            # Do not call SourceStrategy/FedOpt: weights and momentum stay unchanged.
            return ndarrays_to_parameters(self.current_weights), self.train_metrics_aggregated
        value = super().aggregate_fit(server_round, results, failures)
        self._event(server_round, 'aggregated', model_update=True)
        return value

    def _event(self, server_round, phase, **values):
        path = Path('checkpoints') / self.conf['exp_id'] / 'schedule.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(dict(round=server_round, phase=phase, **values)) + '\n')
