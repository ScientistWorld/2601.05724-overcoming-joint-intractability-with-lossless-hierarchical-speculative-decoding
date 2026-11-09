#!/bin/bash
set -euo pipefail
cd /home/user
if [ ! -f redteam/reference_seed1/artifacts/hsd_finite_ar_main.json ]; then
  bash redteam/reference_seed1/run.sh
fi
python -u redteam/audit.py run fixed_setting_shortcut
