"""Unified author-email protocol; launched once per dataset/method/seed."""
import functools
import hashlib
import json
import os
from pathlib import Path

import train_author_gai as author
from src.author_random import RandomStrategy
from src.datasets import spawrious
from src.datasets.data_splits import split_mode_to_matrix

entry = author.entry
raw_cmnist = entry.data_preparation.CMNIST
raw_spawrious = spawrious.get_dataset
loaded = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def members(dataset):
    if dataset.__class__.__name__ == 'CMNIST':
        return [[i, int(y), int(g)] for i, (y, g) in enumerate(
            zip(dataset.subset.targets, dataset.subset.attr))]
    return [[Path(p).resolve().relative_to(Path('datasets').resolve()).as_posix(),
             int(y), int(g)] for p, y, g in dataset.subset]


def load_data(*args, **kwargs):
    conf = kwargs['conf']
    seed = int(conf['seed'])
    entry.data_preparation.CMNIST = functools.partial(raw_cmnist, attribute_seed=seed)

    def get_dataset(*a, **kw):
        kw.update(seed=seed, stable_order=True)
        return raw_spawrious(*a, **kw)

    spawrious.get_dataset = get_dataset
    result = author.original_load(*args, **kwargs)
    loaded['test'] = result[2]
    return result


def split_data(dataset, conf):
    clients = author.original_split(dataset, conf)
    expected = split_mode_to_matrix(conf)
    num_clients = conf['dataset_options']['num_clients']
    if len(clients) != num_clients or len(expected) != num_clients:
        raise RuntimeError('Client count differs from the configured split')
    all_members = members(dataset)
    test_members = members(loaded['test'])
    seen, manifest = set(), []
    for cid, client in enumerate(clients):
        rows = [all_members[int(i)] for i in client.indices]
        counts = [[0, 0], [0, 0]]
        for identity, y, g in rows:
            if identity in seen:
                raise RuntimeError('Overlapping client samples')
            seen.add(identity)
            counts[y][g] += 1
        if counts != expected[cid]:
            raise RuntimeError(f'Incorrect client matrix {cid}: {counts}')
        manifest.append(dict(cid=cid, counts=counts, samples=rows))
    if conf['dataset_options']['name'] == 'Spawrious' and seen & {r[0] for r in test_members}:
        raise RuntimeError('Train/test overlap')
    total = sum(len(c) for c in clients)
    expected_total = sum(sum(map(sum, matrix)) for matrix in expected)
    if total != expected_total:
        raise RuntimeError('Wrong training sample budget')
    output = Path('checkpoints') / conf['exp_id']
    value = dict(clients=manifest, test=test_members)
    (output / 'data_manifest.json').write_text(json.dumps(value, indent=2))
    (output / 'data_manifest.sha256').write_text(digest(value) + '\n')
    return clients


class InitialWeights:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sha = hashlib.sha256()
        for array in self.current_weights:
            sha.update(str((array.shape, str(array.dtype))).encode())
            sha.update(array.tobytes())
        output = Path('checkpoints') / self.conf['exp_id']
        (output / 'initial_weights.sha256').write_text(sha.hexdigest() + '\n')

    def aggregate_fit(self, server_round, results, failures):
        # Use the same summation order despite asynchronous worker completion.
        results = sorted(results, key=lambda item: int(item[1].metrics['cid']))
        return super().aggregate_fit(server_round, results, failures)


class UnifiedFed(InitialWeights, author.AuthorStrategy):
    pass


class UnifiedRandom(InitialWeights, RandomStrategy):
    pass


class UnifiedManager(entry.MyManager):
    def _sample_impl(self, *args, **kwargs):
        # Legacy selectors return column positions; numeric order keeps them aligned with IDs.
        self.clients = dict(sorted(self.clients.items(), key=lambda item: int(item[0])))
        return super()._sample_impl(*args, **kwargs)


def train(conf, conf_name):
    author.seed_process(conf)
    # Deterministic kernels do not imply identical random trajectories across methods.
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    method = conf['server_opt']['selection_method']
    entry.MyStrategy = UnifiedRandom if method == 'random' else UnifiedFed
    model = author.original_train(conf, conf_name)
    output = Path('checkpoints') / conf['exp_id']
    if not (output / 'final/torchmodel.pt').is_file():
        raise RuntimeError('Missing final model')
    loader = entry.data_preparation.preprocess_data(loaded['test'], conf, shuffle=False)
    loss, accuracy, groups = author.optim_utils.evaluate(model, loader, conf)
    metrics = dict(test_accuracy=accuracy, **groups, test_loss_legacy_normalization=loss)
    (output / 'final_test_metrics.json').write_text(json.dumps(metrics, indent=2))
    source_files = ['train_email_four.py', 'src/author_protocol.py', 'src/author_random.py',
                    'flower_train.py', 'src/flower_client.py', 'src/flower_strategy.py',
                    'src/flower_manager.py', 'src/datasets/cmnist.py',
                    'src/datasets/spawrious.py', 'src/datasets/data_splits.py',
                    'src/optimizers/matrix_inference.py', 'src/optimizers/weighting_strategy.py']
    provenance = dict(protocol='email-four-v1', seed=conf['seed'],
        first_update='FedAvgM', dht='after update 1; once; FedDiverse only',
        code_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},
        seed_mode='one shared seed once per process; data generation/split use the same value',
        client_order='numeric candidate IDs and aggregation order',
        pretrained_assumption=True, data_order='sorted Spawrious filenames',
        evaluation='final checkpoint; periodic evaluation uses test copy, no checkpoint selection')
    (output / 'unified_protocol.json').write_text(json.dumps(provenance, indent=2))
    entry.utils.save_config(conf, str(output / 'config.yaml'))
    print('FINAL_TEST_METRICS', json.dumps(metrics), flush=True)
    return model


entry.data_preparation.load_data = load_data
entry.data_preparation.split_data = split_data
entry.train = train
entry.MyManager = UnifiedManager

if __name__ == '__main__':
    author.main()
