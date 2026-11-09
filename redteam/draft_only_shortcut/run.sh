#!/bin/bash
set -euo pipefail
cd /home/user
python -u redteam/audit.py run draft_only_shortcut
