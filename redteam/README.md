This audit changes candidate artifacts and, in one explicitly named shortcut,
the editable finite target definition. It never edits `eval/`, the hidden shell
script, the Python environment, or evaluator processes.

Run `bash redteam/run_all.sh` in a compute job. Each case also has its own
`run.sh`; cases that reuse HSD outputs first generate the seed-one reference if
needed. Candidate generation exits, the visible train script runs, and then
`bash scripts/evaluate_test.sh` runs without arguments as a fresh process.
Results are copied into that case's `scores_test.json` and `scores_train.json`.
The original data module, checkpoints, and scoring files are restored after
each case. Generated checkpoints and full logs are ignored by git; retained
scores, small generation diagnostics, failures, and outcome records are audit
evidence. Delete a case's `artifacts/` directory to regenerate its outputs.

`generate.py` reads only data, baseline, and improve-mode method code plus
previously generated method outputs. It contains no test selector or test salt.
`summarize.py` alone reads the frozen original test scores and metric weights,
solely to compare evaluated attempts. It records all normalization populations.
Future attempt populations are unknown, so there is no population-independent
absolute scalar rank. Pareto dominance and exact ties are independent of those
normalization choices.

The three reference runs use disjoint ranges of per-sequence seeds: 1–80,000,
1,000,001–1,080,000, and 2,000,001–2,080,000. Deterministic probability
computations are memoized for CPU efficiency; the algorithm and RNG stream are
unchanged. The larger honest run uses 640,000 draws starting at seed 3,000,001.
