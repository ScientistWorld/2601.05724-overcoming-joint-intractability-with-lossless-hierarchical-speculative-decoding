#!/bin/bash
set -euo pipefail
cd /home/user
python -u redteam/audit.py batch
python -m redteam.exact_check
