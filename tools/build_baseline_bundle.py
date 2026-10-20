"""Package only the reviewed baseline-validation changes for the new server."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "3674a55"
FILES = [
    "src/baseline_validation.py", "src/config_params.py", "src/flower_client.py",
    "src/flower_manager.py", "src/flower_strategy.py", "src/datasets/spawrious.py",
    "src/datasets/data_preparation.py", "flower_train.py", "conf/server_opt/all_defaults.yaml",
    "BASELINE_VALIDATION.md", "tools/check_baseline_validation.py",
    "tools/run_baseline_validation.py",
] + [f"conf/research_gai_{arm}.yaml" for arm in ("base", "legacy", "pivot", "mapping", "guarded", "corrected", "random")]


def main():
    output = ROOT / "deliverables"
    output.mkdir(exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", BASE], cwd=ROOT, text=True).strip()
    entries = []
    archive = output / "baseline_validation_20261020.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(ROOT / "tools/apply_baseline_bundle.py", "apply_baseline_bundle.py")
        for name in FILES:
            old = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=ROOT, capture_output=True)
            current = (ROOT / name).read_bytes()
            entries.append(dict(path=name,
                                base_sha256=hashlib.sha256(old.stdout).hexdigest() if old.returncode == 0 else None,
                                new_sha256=hashlib.sha256(current).hexdigest()))
            bundle.writestr(f"payload/{name}", current)
        bundle.writestr("bundle_manifest.json", json.dumps(dict(base_commit=commit, files=entries), indent=2))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(digest + "\n", encoding="utf-8")
    print(archive)
    print("SHA256:", digest)
    print("Payload files:", len(entries))


if __name__ == "__main__":
    main()
