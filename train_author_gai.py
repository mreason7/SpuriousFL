"""Author-email GAI training entry point for the repository branch."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

import hydra

os.environ.setdefault('AUTHOR_PROTOCOL', 'email')
os.environ.setdefault('AUTHOR_SEED_MODE', 'global')

import flower_train as entry
from src.author_protocol import AuthorClient, AuthorStrategy, seed_process
from src.optimizers import optim_utils
from src.datasets import data_preparation

entry.FlowerClient = AuthorClient
entry.MyStrategy = AuthorStrategy
original_train = entry.train
original_load = entry.data_preparation.load_data
original_split = entry.data_preparation.split_data
original_simulate = entry.fl.simulation.start_simulation
loaded = {}


def simulate(**kwargs):
    # The old entry point normally executes as __main__. Importing it from
    # this wrapper changes cloudpickle's handling of module globals. Capture
    # the actual conf/splits explicitly so Ray workers never import empty globals.
    conf, clients, validation = entry.conf, entry.ds_split, entry.val_ds

    def client_fn(cid):
        client = AuthorClient(int(cid), conf)
        loader = data_preparation.preprocess_data(clients[int(cid)], conf, shuffle=True)
        client.load_data(loader, validation)
        client.init_model()
        return client.to_client()

    kwargs['client_fn'] = client_fn
    return original_simulate(**kwargs)


def load_data(*args, **kwargs):
    result = original_load(*args, **kwargs)
    loaded['test'] = result[2]
    return result


def split_data(dataset, conf):
    clients = original_split(dataset, conf)
    if len(clients) != 24 or sum(map(len, clients)) != 5120:
        raise RuntimeError('Historical 24-client/5120-sample GAI split not recovered')
    expected = [[[15, 5], [5, 45]], [[5, 45], [15, 5]],
                [[15, 45], [5, 5]], [[5, 5], [15, 45]]] + [[[1, 120], [1, 120]]] * 20
    manifest = []
    seen = set()
    for cid, client in enumerate(clients):
        counts = [[0, 0], [0, 0]]
        rows = []
        for index in client.indices:
            path, y, g = dataset.subset[index]
            identity = Path(path).resolve().relative_to(Path('datasets').resolve()).as_posix()
            if identity in seen:
                raise RuntimeError('Overlapping client samples')
            seen.add(identity)
            counts[int(y)][int(g)] += 1
            rows.append([identity, int(y), int(g)])
        if counts != expected[cid]:
            raise RuntimeError(f'Wrong client group matrix: {cid}: {counts}')
        manifest.append(dict(cid=cid, counts=counts, samples=rows))
    test_ids = {Path(p).resolve().relative_to(Path('datasets').resolve()).as_posix()
                for p, _, _ in loaded['test'].subset}
    if test_ids & seen:
        raise RuntimeError('Train/test overlap')
    output = Path('checkpoints') / conf['exp_id']
    data = json.dumps(manifest, sort_keys=True, indent=2).encode()
    (output / 'data_manifest.json').write_bytes(data)
    (output / 'data_manifest.sha256').write_text(hashlib.sha256(data).hexdigest() + '\n')
    return clients


def train(conf, conf_name):
    seed_process(conf)
    model = original_train(conf, conf_name)
    output = Path('checkpoints') / conf['exp_id']
    if not (output / 'final/torchmodel.pt').is_file():
        raise RuntimeError('Missing final checkpoint')
    test_loader = entry.data_preparation.preprocess_data(loaded['test'], conf, shuffle=False)
    loss, accuracy, groups = optim_utils.evaluate(model, test_loader, conf)
    metrics = dict(test_accuracy=accuracy, **groups, test_loss_legacy_normalization=loss)
    (output / 'final_test_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    entry.utils.save_config(conf, str(output / 'config.yaml'))
    provenance = dict(author_reference_commit='3c92e8a5a80be836041e1f43bd921ed862156e67',
                      code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
                      protocol=os.environ['AUTHOR_PROTOCOL'], seed=conf['seed'],
                      seed_mode=os.environ['AUTHOR_SEED_MODE'],
                      pretrained_assumption=True,
                      initial_update='historical FedAvgM; email does not specify FedAvg vs FedAvgM')
    provenance['code_sha256'] = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            'train_author_gai.py', 'flower_train.py', 'conf/spawrious_gai_author.yaml',
            'src/author_protocol.py', 'src/flower_client.py', 'src/flower_strategy.py',
            'src/flower_manager.py', 'src/datasets/data_splits.py',
            'src/datasets/spawrious.py', 'src/optimizers/matrix_inference.py',
            'src/optimizers/weighting_strategy.py', 'src/optimizers/subpopbench.py')}
    (output / 'author_protocol.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('FINAL_TEST_METRICS', json.dumps(metrics), flush=True)
    return model


entry.train = train
entry.data_preparation.load_data = load_data
entry.data_preparation.split_data = split_data
entry.fl.simulation.start_simulation = simulate


def main():
    # entry.main was decorated in an imported module, so its relative 'conf'
    # is treated as a package. Reuse its body with an absolute filesystem path.
    launch = hydra.main(
        config_path=str(Path(__file__).resolve().parent / 'conf'),
        config_name='spawrious_gai_author',
        version_base=None,
    )(entry.main.__wrapped__)
    launch()


if __name__ == '__main__':
    main()
