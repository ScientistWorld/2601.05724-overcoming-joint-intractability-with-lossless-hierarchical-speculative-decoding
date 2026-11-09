#!/bin/bash
set -euo pipefail
cd /home/user
python -u redteam/audit.py run weaker_fewer_samples
