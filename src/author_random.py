"""Random control: one shared all-client update, then 199 random updates."""
import copy

from flwr.common import FitIns
from src.author_protocol import AuthorStrategy
from src.flower_strategy import MyStrategy


class RandomStrategy(MyStrategy):
    _event = AuthorStrategy._event

    def configure_fit(self, server_round, parameters, client_manager):
        count = (self.conf['dataset_options']['num_clients'] if server_round == 1
                 else self.conf.get('server_opt', {}).get('num_active_clients', 9))
        clients = client_manager.sample(num_clients=count, min_num_clients=count,
                                       client_info=self.stored_client_data, server_round=server_round)
        self.client_update_requested = []
        tasks = []
        for client in clients:
            config = copy.deepcopy(self.shared_copt_params)
            config.update(round=server_round, update_info=False, metadata_only=False)
            tasks.append((client, FitIns(parameters, config)))
        self.expected_ids = {str(c.cid) for c, _ in tasks}
        if len(tasks) != count or len(self.expected_ids) != count:
            raise RuntimeError('Incomplete/duplicate random client selection')
        self._event(server_round, 'configured', client_ids=sorted(self.expected_ids, key=int),
                    metadata_only=False)
        return tasks

    def aggregate_fit(self, server_round, results, failures):
        actual = {str(res.metrics['cid']) for _, res in results}
        if failures or actual != self.expected_ids or len(results) != len(self.expected_ids):
            raise RuntimeError('Failed/missing/duplicate random clients')
        value = super().aggregate_fit(server_round, results, failures)
        self.log_client_info(server_round)
        self._event(server_round, 'aggregated', model_update=True)
        return value
