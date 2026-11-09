#!/bin/bash
set -euo pipefail
cd /home/user
if [ ! -f redteam/reference_seed1/artifacts/hsd_finite_ar_main.json ]; then
  python -u -m redteam.generate reference_seed1
fi
python -u redteam/audit.py run self_fields_shortcut
