#!/bin/bash

set -e

cd /home/user

SAMPLES=80000 bash scripts/reproduce.sh
python validate.py --compare
bash scripts/evaluate_train.sh /home/user/checkpoints
bash scripts/evaluate_test.sh /home/user/checkpoints
