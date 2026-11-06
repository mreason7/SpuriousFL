"""Rename the completed formal batch and update summary log paths."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rename_batch(root, source_name):
    logs = (root / 'run_logs').resolve()
    source = (logs / source_name).resolve()
    destination = (logs / 'email-four-formal').resolve()
    if (source.parent != logs or destination.parent != logs or
            not source.name.startswith('email-four-formal-')):
        raise RuntimeError('Source and destination must be batch directories directly inside run_logs')
    if destination.exists():
        raise RuntimeError(f'Destination already exists; nothing overwritten: {destination}')
    if not source.is_dir():
        raise RuntimeError(f'Source directory not found: {source}')
    original = (source / 'summary.json').read_bytes()
    rows = json.loads(original)
    filenames = []
    for row in rows:
        name = f"{row['dataset']}_{row['method']}_seed{row['seed']}.log"
        if Path(name).name != name or not (source / name).is_file():
            raise RuntimeError(f'Expected training log not found: {name}')
        filenames.append(name)
        row['log'] = str(destination / name)
    if len(filenames) != len(set(filenames)):
        raise RuntimeError('Duplicate log filenames in summary')
    backup = source / 'summary_before_directory_rename.json'
    prepared = source / 'summary.directory_rename.tmp'
    if backup.exists() or prepared.exists():
        raise RuntimeError('A prior rename attempt left preparation files; inspect before retrying')
    # Validate absolute targets above before moving the directory.
    with backup.open('xb') as stream:
        stream.write(original)
    with prepared.open('x', encoding='utf-8') as stream:
        json.dump(rows, stream, indent=2)
    source.rename(destination)
    (destination / prepared.name).replace(destination / 'summary.json')
    print('RENAMED', source, '->', destination)
    print(f'Preserved {len(rows)} results and log filenames; updated summary log paths.')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-name', default='email-four-formal-20261105-131709-487820')
    args = parser.parse_args()
    rename_batch(ROOT, args.source_name)


if __name__ == '__main__':
    main()
