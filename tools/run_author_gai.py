"""Run author-email GAI on this branch, stopping on failures without auto-retry."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_source():
    for name in ('train_author_gai.py', 'src/author_protocol.py', 'conf/spawrious_gai_author.yaml'):
        if not (ROOT / name).is_file():
            raise RuntimeError(f'Missing training file: {name}')


def assets(directory):
    for name in ('datasets', 'pretrained_ckpt'):
        link = ROOT / name
        target = directory / name
        if not link.exists() and not link.is_symlink():
            if not target.is_dir():
                raise RuntimeError(f'Missing assets: {target}')
            link.symlink_to(target.resolve(), target_is_directory=True)
    for path in (ROOT / 'datasets/spawrious224/0', ROOT / 'datasets/spawrious224/1',
                 ROOT / 'pretrained_ckpt/pt_mobilenetv2_100-4065ukrp/pretrained_imagenet.pth'):
        if not path.exists():
            raise RuntimeError(f'Missing required asset: {path}')


def preflight():
    print('PREFLIGHT: checking PyTorch, configuration, GPU and training imports...', flush=True)
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.config_store import ConfigStore
    from omegaconf import OmegaConf
    from src.config_params import Config
    ConfigStore.instance().store(group='job', name='federated_training', node=Config)
    with initialize_config_dir(config_dir=str(ROOT / 'conf'), version_base=None):
        conf = OmegaConf.to_container(compose(config_name='spawrious_gai_author'), resolve=True,
                                      throw_on_missing=True)
    if conf['dataset_options']['num_clients'] != 24 or conf['server_opt']['rounds'] != 201:
        raise RuntimeError('Wrong author configuration')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment')
    import train_author_gai  # noqa: F401
    weights = ROOT / 'pretrained_ckpt/pt_mobilenetv2_100-4065ukrp/pretrained_imagenet.pth'
    return dict(python=sys.version, gpu=torch.cuda.get_device_name(0),
                pretrained_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                versions={name: importlib.metadata.version(name)
                          for name in ('torch', 'torchvision', 'flwr', 'ray', 'numpy',
                                       'pandas', 'hydra-core', 'timm')})


def verify_run(directory, rounds, protocol):
    for name in ('config.yaml', 'final/torchmodel.pt', 'client_info.json',
                 'client_weights.csv', 'schedule.jsonl', 'data_manifest.json',
                 'data_manifest.sha256', 'final_test_metrics.json', 'author_protocol.json'):
        if not (directory / name).is_file():
            raise RuntimeError(f'Missing artifact: {directory / name}')
    weights = [[float(x) for x in row.split(',')]
               for row in (directory / 'client_weights.csv').read_text().splitlines()]
    expected = [24, 24] + [9] * (rounds - 2) if protocol == 'email' else [24] + [9] * (rounds - 1)
    if len(weights) != rounds or any(len(row) != 24 or sum(row) != count or
                                   any(x not in (0, 1) for x in row)
                                   for row, count in zip(weights, expected)):
        raise RuntimeError('Invalid client participation schedule')
    events = [json.loads(line) for line in (directory / 'schedule.jsonl').read_text().splitlines()]
    updates = [row for row in events if row['phase'] == 'aggregated']
    expected_updates = rounds - 1 if protocol == 'email' else rounds
    if len(updates) != rounds or sum(row['model_update'] for row in updates) != expected_updates:
        raise RuntimeError('Wrong model-update count')
    if protocol == 'email' and [row['round'] for row in updates if not row['model_update']] != [2]:
        raise RuntimeError('DHT collection did not occur exactly once after update 1')
    metrics = json.loads((directory / 'final_test_metrics.json').read_text())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('plan', 'preflight', 'smoke', 'formal'), default='plan')
    parser.add_argument('--protocol', choices=('email',), default='email')
    parser.add_argument('--seed-mode', choices=('global', 'source'), default='global',
                        help='global: shared seed once per process; source: unchanged historical RNG')
    parser.add_argument('--seeds', help='default: 42 for smoke, 1,2,42 for formal')
    parser.add_argument('--assets-root', type=Path, default=ROOT)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    seeds = [int(s) for s in (args.seeds or ('42' if args.stage == 'smoke' else '1,2,42')).split(',')]
    if len(seeds) != len(set(seeds)):
        parser.error('Repeated seed')
    rounds = 3 if args.stage == 'smoke' else (201 if args.protocol == 'email' else 200)
    commands = [[sys.executable, 'train_author_gai.py', '--config-name=spawrious_gai_author', f'seed={seed}',
                 f'server_opt.rounds={rounds}'] +
                (['client_opt.num_steps=2', 'client_opt.biased_trainer_steps=2',
                  'client_opt.left_right_trainer_steps=2'] if args.stage == 'smoke' else [])
                for seed in seeds]
    verify_source()
    if args.stage == 'preflight':
        print('PREFLIGHT: checking readiness only; no training will start.', flush=True)
    else:
        for command in commands:
            print('COMMAND', ' '.join(command), flush=True)
    if args.stage == 'plan':
        return
    os.chdir(ROOT)
    os.environ.update(AUTHOR_PROTOCOL=args.protocol, AUTHOR_SEED_MODE=args.seed_mode,
                      CUDA_VISIBLE_DEVICES=args.gpu)
    print('PREFLIGHT: checking dataset and pretrained weights...', flush=True)
    assets(args.assets_root.resolve())
    environment = preflight()
    print(json.dumps(environment, indent=2), flush=True)
    if args.stage == 'preflight':
        print('PREFLIGHT PASSED', flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output = ROOT / 'run_logs' / f'{args.stage}-{args.protocol}-{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_text(json.dumps(dict(stage=args.stage, protocol=args.protocol,
        seed_mode=args.seed_mode, seeds=seeds, commands=commands, environment=environment), indent=2))
    code_files = ['train_author_gai.py', 'flower_train.py', 'conf/spawrious_gai_author.yaml',
                  'src/author_protocol.py', 'src/flower_client.py', 'src/flower_strategy.py',
                  'src/flower_manager.py', 'src/datasets/data_splits.py',
                  'src/datasets/spawrious.py', 'src/optimizers/matrix_inference.py',
                  'src/optimizers/weighting_strategy.py', 'src/optimizers/subpopbench.py']
    (output / 'code_sha256.json').write_text(json.dumps({
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code_files}, indent=2))
    summary = []
    for seed, command in zip(seeds, commands):
        previous = set((ROOT / 'checkpoints').glob('*/config.yaml'))
        env = dict(os.environ)
        if args.seed_mode == 'global':
            env['PYTHONHASHSEED'] = str(seed)
        log = output / f'seed-{seed}.log'
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Seed {seed} failed; inspect {log}; no automatic retry')
        new = set((ROOT / 'checkpoints').glob('*/config.yaml')) - previous
        if len(new) != 1:
            raise RuntimeError('Expected one new checkpoint directory')
        directory = new.pop().parent
        metrics = verify_run(directory, rounds, args.protocol)
        summary.append(dict(seed=seed, checkpoint=str(directory), **metrics))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('VERIFIED', seed, directory, metrics, flush=True)
    if args.stage == 'formal':
        values = [row['worst_group'] for row in summary]
        aggregate = dict(n=len(values), mean=statistics.mean(values), std_population=statistics.pstdev(values),
                         checkpoint='final', protocol=args.protocol, seed_mode=args.seed_mode)
        (output / 'aggregate.json').write_text(json.dumps(aggregate, indent=2) + '\n')
        print('AGGREGATE', json.dumps(aggregate), flush=True)
    print('COMPLETED', output, flush=True)


if __name__ == '__main__':
    main()
