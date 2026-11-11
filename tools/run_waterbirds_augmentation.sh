#!/usr/bin/env bash
# Run inside the existing spurious environment and a persistent tmux session.
set -euo pipefail
cd "$(dirname "$0")/.."
baseline="run_logs/email-four-formal/summary.json"
test -f "$baseline"
python tools/check_waterbirds_augmentation.py
python tools/run_waterbirds_augmentation.py --stage preflight --baseline-summary "$baseline"
python -u tools/run_waterbirds_augmentation.py --stage smoke --baseline-summary "$baseline"
mkdir -p run_logs
launch_log="run_logs/waterbirds-augmentation-launch-$(date +%Y%m%d-%H%M%S).log"
python -u tools/run_waterbirds_augmentation.py --stage formal --baseline-summary "$baseline" | tee "$launch_log"
batch=$(python - "$launch_log" <<'PY'
from pathlib import Path
import sys
lines = Path(sys.argv[1]).read_text().splitlines()
matches = [line.split(' ', 2)[2] for line in lines if line.startswith('COMPLETED 6 ')]
if len(matches) != 1:
    raise SystemExit('Expected exactly one completed formal batch.')
print(matches[0])
PY
)
python tools/compare_waterbirds_augmentation.py --baseline-summary "$baseline" --aug-batch "$batch"
echo "Comparison saved in: $batch/augmentation_comparison.csv"
