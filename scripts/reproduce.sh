#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

SAMPLES="${SAMPLES:-50000}"

for scenario in finite_ar_main finite_ar_gamma3 finite_ar_temp08; do
  SAMPLES="${SAMPLES}" bash scripts/baseline.sh "${scenario}"
  SAMPLES="${SAMPLES}" bash scripts/method.sh "${scenario}"
done

bash scripts/evaluate.sh
