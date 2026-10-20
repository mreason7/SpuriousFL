"""Apply a verified local code bundle without touching data or checkpoints.

Run this script from the unpacked bundle, not from inside the code payload.
Checks every file before the first write; keeps backups outside the repository.
Reapplying identical content is safe. Refuses unrelated local modifications.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--check", action="store_true", help="Verify only; make no changes")
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    repository = Path(args.repo).resolve()
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    if head != manifest["base_commit"]:
        raise SystemExit("Repository HEAD differs from the audited baseline; stop and inspect")
    changes = []
    for item in manifest["files"]:
        relative = Path(item["path"])
        target = (repository / relative).resolve()
        source = (bundle / "payload" / relative).resolve()
        if not target.is_relative_to(repository) or not source.is_relative_to(bundle / "payload"):
            raise SystemExit("Invalid bundle path")
        if sha(source) != item["new_sha256"]:
            raise SystemExit(f"Bundle checksum failed: {relative}")
        if target.exists() and sha(target) == item["new_sha256"]:
            continue
        if item["base_sha256"] is None:
            if target.exists():
                raise SystemExit(f"New-file target already contains different content: {relative}")
        elif not target.is_file() or sha(target) != item["base_sha256"]:
            raise SystemExit(f"Existing source has local modifications: {relative}")
        changes.append((source, target, relative))
    print(f"Verified bundle; {len(changes)} files need applying")
    if args.check or not changes:
        return
    backup = repository.parent / "baseline_validation_backups" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for source, target, relative in changes:
        if target.exists():
            copy = backup / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, copy)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print("Applied. Original source backups:", backup)
    print("Data, models and experiment logs were not modified.")


if __name__ == "__main__":
    main()
