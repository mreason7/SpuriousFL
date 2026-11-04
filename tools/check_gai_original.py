"""Check original source, resolved configs and GAI matrix without training."""
import ast
from pathlib import Path
import subprocess
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '3c92e8a5a80be836041e1f43bd921ed862156e67'
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    result = subprocess.run(['git', 'diff', '--quiet', COMMIT, '--', 'flower_train.py', 'src'], cwd=ROOT)
    require(result.returncode == 0, 'Training code differs from the author commit')

    from hydra import compose, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
    from src.config_params import Config
    ConfigStore.instance().store(group='job', name='federated_training', node=Config)
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base=None):
        configs = [OmegaConf.to_container(compose(config_name=name), resolve=True, throw_on_missing=True)
                   for name in ('spawrious_gai_original', 'spawrious_gai_original_random')]
    fed, random = configs
    for name in ('seed', 'client_opt', 'dataset_options', 'model_options', 'machine'):
        require(fed[name] == random[name], f'Random control differs in {name}')
    differences = {key for key in fed['server_opt']
                   if fed['server_opt'][key] != random['server_opt'][key]}
    require(differences == {'selection_method', 'client_info'}, 'Unexpected control configuration difference')
    require(random['server_opt']['selection_method'] == 'random' and not random['server_opt']['client_info'],
            'Random must not estimate DHT')
    require(fed['server_opt']['rounds'] == 200 and fed['server_opt']['pretrain_rounds'] == 1 and
            fed['server_opt']['num_active_clients'] == 9, 'Wrong update budget or participation')
    require(fed['client_opt']['batch_size'] == 28 and fed['client_opt']['epochs'] == 1 and
            fed['client_opt']['num_steps'] is None, 'Wrong local training budget')
    require((fed['client_opt']['biased_trainer_steps'], fed['client_opt']['left_right_trainer_steps'],
             fed['client_opt']['generalized_cross_entropy_q']) == (50, 10, 0.3), 'Wrong DHT settings')
    require(fed['dataset_options']['split_mode'] == 'spawrious_GAI_2' and
            fed['dataset_options']['num_clients'] == 24, 'Wrong historical split')

    tree = ast.parse((ROOT / 'src/datasets/data_splits.py').read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                 node.name == 'split_mode_to_matrix']
    scope = {'List': List}
    exec(compile(tree, 'historical_data_splits.py', 'exec'), scope)
    matrices = scope['split_mode_to_matrix']('spawrious_GAI_2')
    require(len(matrices) == 24 and sum(sum(map(sum, m)) for m in matrices) == 5120,
            'Historical GAI matrix must have 24 clients and 5120 samples')
    print('ORIGINAL CONFIG CHECK PASSED')
    print('Training source unchanged from', COMMIT)
    print('GAI: 24 clients, 5120 samples; 200 update rounds; 9 selected after round 1')
    print('RNG and first-round DHT timing follow the original source; no extra seed patch')
    print('pretrained=true is an explicit assumption for the attachment placeholder')
    print('No GPU training was started by this check')


if __name__ == '__main__':
    main()
