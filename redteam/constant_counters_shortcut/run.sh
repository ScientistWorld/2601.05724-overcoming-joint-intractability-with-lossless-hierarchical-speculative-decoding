#!/bin/bash
set -euo pipefail
cd /home/user
python -u redteam/audit.py run constant_counters_shortcut
