"""CPU checks for the paired-run configuration and artifact verifier."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_email_four import resolved_config
from tools.run_estimated_dht_gsc import REFERENCES, metrics, summarize, verify_config


class EstimatedDHTChecks(unittest.TestCase):
    def test_only_information_source_changes_for_each_seed(self):
        estimated = resolved_config('research_complementarity_gsc_estimated_dht')
        reference = resolved_config('research_complementarity_gsc_dht')
        for seed in (42, 43, 44):
            left, right = copy.deepcopy(estimated), copy.deepcopy(reference)
            for conf in (left, right):
                conf['seed'] = seed
                conf['research'].update(training_seed=seed+100, validation_seed=seed+1000)
            verify_config(left, right, seed)
            self.assertEqual(left['client_opt']['biased_trainer_steps'], 50)
            self.assertEqual(left['client_opt']['left_right_trainer_steps'], 10)
            self.assertEqual(left['client_opt']['generalized_cross_entropy_q'], .3)
            left['client_opt']['learning_rate'] *= 2
            with self.assertRaisesRegex(ValueError, 'Controls differ'):
                verify_config(left, right, seed)

    def test_smoke_allows_only_the_declared_budget_changes(self):
        estimated = resolved_config('research_complementarity_gsc_estimated_dht')
        reference = resolved_config('research_complementarity_gsc_dht')
        estimated['server_opt']['rounds'] = 3
        estimated['client_opt'].update(num_steps=2, biased_trainer_steps=2, left_right_trainer_steps=2)
        verify_config(estimated, reference, 42, smoke=True)
        with self.assertRaisesRegex(ValueError, 'Controls differ'):
            verify_config(estimated, reference, 42)

    def test_artifact_metrics_and_invalid_schedule_or_group_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'final').mkdir()
            (directory/'final/torchmodel.pt').write_bytes(b'fixture')
            for name in ('data_manifest.sha256', 'initial_weights.sha256'):
                (directory/name).write_text('a'*64+'\n')
            (directory/'data_manifest.json').write_text('{}')
            info = {str(i): {f'interaction_matrix_{g}': 50 for g in range(4)} for i in range(24)}
            (directory/'client_info.json').write_text(json.dumps(info))
            scores = dict(worst_group=80, test_accuracy=80, y0g0=80, y0g1=80, y1g0=80, y1g1=80)
            (directory/'final_test_metrics.json').write_text(json.dumps(scores))
            (directory/'final_validation_metrics.json').write_text(json.dumps(dict(worst_group=81)))
            rows = [[1]*24, [1]*24, [1]*9+[0]*15]
            participation = directory/'client_weights.csv'
            participation.write_text('\n'.join(','.join(map(str,row)) for row in rows))
            result = metrics(directory, rounds=3)
            self.assertEqual(result['min_group_share_pct'], 25)
            self.assertEqual(result['joint_error'], 0)
            self.assertAlmostEqual(result['effective_clients'], 9)
            scores['y1g1'] = 70
            (directory/'final_test_metrics.json').write_text(json.dumps(scores))
            with self.assertRaisesRegex(ValueError, 'WGA differs'):
                metrics(directory, rounds=3)
            rows[2][9] = 1
            participation.write_text('\n'.join(','.join(map(str,row)) for row in rows))
            with self.assertRaisesRegex(ValueError, 'schedule mismatch'):
                metrics(directory, rounds=3)

    def test_four_arm_summary_and_changed_pair_identity(self):
        import yaml
        reference_config = resolved_config('research_complementarity_gsc_dht')
        estimated_config = resolved_config('research_complementarity_gsc_estimated_dht')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root/'batch'
            batch.mkdir()
            runs = []
            for seed, references in REFERENCES.items():
                folders = dict(references, EstimatedDHT=f'estimated-{seed}')
                for arm, folder in folders.items():
                    directory = root/'checkpoints'/folder
                    (directory/'final').mkdir(parents=True)
                    (directory/'final/torchmodel.pt').write_bytes(b'fixture')
                    for name in ('data_manifest.sha256','initial_weights.sha256'):
                        (directory/name).write_text('a'*64)
                    (directory/'data_manifest.json').write_text('{}')
                    cfg = copy.deepcopy(estimated_config if arm=='EstimatedDHT' else reference_config)
                    cfg['seed'] = seed
                    cfg['research'].update(training_seed=seed+100,validation_seed=seed+1000)
                    (directory/'config.yaml').write_text(yaml.safe_dump(cfg))
                    info = {str(i): {f'interaction_matrix_{g}':50 for g in range(4)} for i in range(24)}
                    (directory/'client_info.json').write_text(json.dumps(info))
                    scores = dict(worst_group=80,test_accuracy=80,y0g0=80,y0g1=80,y1g0=80,y1g1=80)
                    (directory/'final_test_metrics.json').write_text(json.dumps(scores))
                    (directory/'final_validation_metrics.json').write_text(json.dumps(dict(worst_group=81)))
                    rows = [[1]*24]*2 + [[1]*9+[0]*15]*199
                    (directory/'client_weights.csv').write_text('\n'.join(','.join(map(str,row)) for row in rows))
                    if arm=='EstimatedDHT':
                        runs.append(dict(seed=seed,checkpoint=f'checkpoints/{folder}'))
            (batch/'summary.json').write_text(json.dumps(dict(stage='formal',runs=runs)))
            with redirect_stdout(io.StringIO()):
                summarize(batch,root)
            self.assertEqual(len((batch/'comparison.csv').read_text(encoding='utf-8-sig').splitlines()),13)
            changed = root/'checkpoints'/REFERENCES[42]['DHT']/'initial_weights.sha256'
            changed.write_text('b'*64)
            with self.assertRaisesRegex(ValueError,'not paired'):
                summarize(batch,root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
