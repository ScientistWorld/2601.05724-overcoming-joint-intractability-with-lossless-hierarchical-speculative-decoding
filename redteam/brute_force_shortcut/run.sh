#!/bin/bash
set -euo pipefail
cd /home/user
python -u redteam/audit.py run brute_force_shortcut
