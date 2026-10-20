"""Check the transfer bundle against pristine baseline files in a temp folder."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parent = ROOT.parent / ".baseline-validation-checks"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temporary:
        base = Path(temporary).resolve()
        assert base.is_relative_to(parent.resolve())
        unpacked = base / "bundle"
        unpacked.mkdir()
        with zipfile.ZipFile(ROOT / "deliverables/baseline_validation_20261020.zip") as archive:
            for name in archive.namelist():
                assert (unpacked / name).resolve().is_relative_to(unpacked.resolve())
            archive.extractall(unpacked)
        manifest = json.loads((unpacked / "bundle_manifest.json").read_text())
        repository = base / "server-copy"
        repository.mkdir()
        originals = {}
        for item in manifest["files"]:
            if item["base_sha256"] is not None:
                content = subprocess.check_output(["git", "show", f"{manifest['base_commit']}:{item['path']}"], cwd=ROOT)
                destination = repository / item["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                originals[item["path"]] = content

        def apply(check=False):
            arguments = ["apply_baseline_bundle.py", "--repo", str(repository)] + (["--check"] if check else [])
            with patch.object(sys, "argv", arguments), patch("subprocess.check_output", return_value=manifest["base_commit"]), redirect_stdout(io.StringIO()):
                runpy.run_path(str(unpacked / "apply_baseline_bundle.py"), run_name="__main__")

        apply(check=True)
        assert all((repository / name).read_bytes() == data for name, data in originals.items())
        assert not (repository / "src/baseline_validation.py").exists()
        apply()
        for item in manifest["files"]:
            assert hashlib.sha256((repository / item["path"]).read_bytes()).hexdigest() == item["new_sha256"]
        backups = list((base / "baseline_validation_backups").iterdir())
        assert len(backups) == 1
        assert all((backups[0] / name).read_bytes() == data for name, data in originals.items())
        apply()  # idempotent replay
        changed = repository / "src/flower_client.py"
        changed.write_text("user local edit\n")
        try:
            apply()
        except SystemExit:
            pass
        else:
            raise AssertionError("Modified server source was not protected")
        assert changed.read_text() == "user local edit\n"
        print("Bundle checks passed: preflight, 19 file hashes, backups, idempotence, local-edit protection")


if __name__ == "__main__":
    main()
