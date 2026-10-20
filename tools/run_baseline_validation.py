"""Run isolated ablation processes and verify artifacts, never auto-rerun failures."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARMS = {name: f"research_gai_{name}" for name in ("legacy", "pivot", "mapping", "guarded", "corrected", "random")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["plan", "smoke", "formal"], default="plan")
    parser.add_argument("--arms", default="legacy,pivot,mapping,guarded")
    parser.add_argument("--seeds", default="42", help="data seeds; training=seed+100, validation=seed+1000")
    args = parser.parse_args()
    arms = args.arms.split(",")
    if not arms or len(arms) != len(set(arms)) or any(a not in ARMS for a in arms):
        parser.error("Unknown or repeated arm")
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds):
        parser.error("Repeated seed")
    identifier = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = ROOT / "logs" / "baseline_validation" / f"{args.stage}-{identifier}"
    comparisons = {}
    summaries = []
    for seed in seeds:
        for arm in arms:
            tag = f"{args.stage}-{identifier}-{arm}-{seed}"
            command = [sys.executable, "flower_train.py", f"--config-name={ARMS[arm]}",
                       f"seed={seed}", f"research.training_seed={seed+100}",
                       f"research.validation_seed={seed+1000}", f"research.run_tag={tag}"]
            if args.stage == "smoke":
                command += ["server_opt.rounds=3", "client_opt.num_steps=2",
                            "client_opt.biased_trainer_steps=2", "client_opt.left_right_trainer_steps=2"]
            print("COMMAND", " ".join(command), flush=True)
            if args.stage == "plan":
                continue
            output.mkdir(parents=True, exist_ok=True)
            if not summaries:
                (output / "pip-freeze.txt").write_text(
                    subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
            log = output / f"{arm}-seed{seed}.log"
            env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="0")
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace")
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            if code:
                raise SystemExit(f"STOP: {arm} failed with exit {code}; inspect {log}; no auto-retry")
            matches = []
            for path in (ROOT / "checkpoints").glob("*/config.yaml"):
                cfg = yaml.safe_load(path.read_text())
                if (cfg.get("research") or {}).get("run_tag") == tag:
                    matches.append((path.parent, cfg))
            if len(matches) != 1:
                raise SystemExit("STOP: expected exactly one checkpoint directory for run tag")
            directory, cfg = matches[0]
            rounds = 3 if args.stage == "smoke" else 201
            for name in ("final/torchmodel.pt", "client_info.json", "client_weights.csv", "data_manifest.json",
                         "initial_weights.sha256", "data_manifest.sha256", "final_validation_metrics.json", "final_test_metrics.json"):
                if not (directory / name).is_file():
                    raise SystemExit(f"STOP: missing artifact {directory / name}")
            weights = np.loadtxt(directory / "client_weights.csv", delimiter=",")
            if weights.shape != (rounds, 25) or not np.all((weights == 0) | (weights == 1)):
                raise SystemExit("STOP: invalid participation CSV")
            if not np.all(weights[:2].sum(axis=1) == 25) or not np.all(weights[2:].sum(axis=1) == 9):
                raise SystemExit("STOP: update schedule/participation differs from protocol")
            if "DHT collection round 2 completed without a model update" not in log.read_text(encoding="utf-8"):
                raise SystemExit("STOP: missing metadata-only round evidence")
            identity = ((directory / "data_manifest.sha256").read_text().strip(),
                        (directory / "initial_weights.sha256").read_text().strip())
            if seed in comparisons and comparisons[seed] != identity:
                raise SystemExit("STOP: data membership or initial model differs between arms")
            comparisons[seed] = identity
            actual_modes = {key: cfg["server_opt"][key] for key in ("pivot_mode", "client_id_mapping", "selector_mode")}
            expected_modes = dict(pivot_mode="legacy", client_id_mapping="legacy", selector_mode="legacy")
            if arm in ("pivot", "corrected"):
                expected_modes["pivot_mode"] = "min_count_gap"
            if arm in ("mapping", "corrected"):
                expected_modes["client_id_mapping"] = "candidate"
            if arm in ("guarded", "corrected"):
                expected_modes["selector_mode"] = "guarded"
            if actual_modes != expected_modes:
                raise SystemExit("STOP: ablation flag mismatch")
            validation = json.loads((directory / "final_validation_metrics.json").read_text())
            summary = dict(arm=arm, data_seed=seed, training_seed=seed+100,
                           stage=args.stage, config=str(directory.relative_to(ROOT) / "config.yaml"),
                           checkpoint=str(directory.relative_to(ROOT)), modes=actual_modes,
                           manifest_sha256=identity[0], initial_weights_sha256=identity[1],
                           validation=validation,
                           note="Smoke scores are not formal performance results" if args.stage == "smoke" else "Fixed-final-checkpoint result")
            summaries.append(summary)
            (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
            print("VERIFIED", directory, flush=True)
    if args.stage != "plan":
        print("Completed. Summary:", output / "summary.json")


if __name__ == "__main__":
    main()
