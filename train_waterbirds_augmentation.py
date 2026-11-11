"""WaterBirds augmentation with baseline data/initialization guards."""
import hashlib
import json
import os
from pathlib import Path

import torch
import train_email_four as unified

original_split = unified.entry.data_preparation.split_data
original_initial = unified.InitialWeights.__init__
original_train = unified.train


def split_data(dataset, conf):
    # Group counting reads transformed images before model initialization. Preserve
    # the CPU RNG here so enabling augmentation cannot also change the initial head.
    # Actual local training and DHT estimation continue to consume RNG normally.
    with torch.random.fork_rng(devices=[]):
        return original_split(dataset, conf)


def checked_initial(self, *args, **kwargs):
    original_initial(self, *args, **kwargs)
    conf = self.conf
    if (conf['dataset_options']['name'] != 'WaterBirds'
            or conf['dataset_options']['aug_crop'] != 24
            or conf['dataset_options']['aug_horizontal_flip'] is not True):
        raise RuntimeError('This entry is exclusively for the WaterBirds augmentation condition.')
    rows = json.loads(Path(os.environ['WB_AUG_BASELINE']).read_text())
    method = 'random' if conf['server_opt']['selection_method'] == 'random' else 'feddiverse'
    matches = [r for r in rows if r['dataset'] == 'waterbirds_dist'
               and r['method'] == method and r['seed'] == conf['seed']]
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one matched baseline run.')
    output = Path('checkpoints') / conf['exp_id']
    actual = {name: (output / name).read_text().strip() for name in matches[0]['hashes']}
    if actual != matches[0]['hashes']:
        raise RuntimeError('Augmentation data or initial weights differ from the original baseline; stopped before training.')
    (output / 'augmentation_control.json').write_text(json.dumps(dict(
        condition='waterbirds-augmentation', baseline_run=matches[0], matched_hashes=actual,
        aug_crop=24, aug_horizontal_flip=True,
        metadata_rng='CPU torch RNG preserved only during driver data splitting/group counting',
        entry_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))


def train(conf, conf_name):
    model = original_train(conf, conf_name)
    path = Path('checkpoints') / conf['exp_id'] / 'unified_protocol.json'
    value = json.loads(path.read_text())
    value.update(condition='waterbirds-augmentation', augmentation_control='augmentation_control.json',
                 aug_crop=24, aug_horizontal_flip=True)
    path.write_text(json.dumps(value, indent=2))
    return model


unified.entry.data_preparation.split_data = split_data
unified.InitialWeights.__init__ = checked_initial
unified.entry.train = train

if __name__ == '__main__':
    unified.author.main()
