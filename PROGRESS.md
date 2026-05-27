# Progress

## What Works

- Implemented the paper's HSD verifier in `method/hsd.py`, including capped
  prefix ratios, capped branch divergences, backward verification, and capped
  one-token resampling.
- Implemented tokenwise speculative decoding baseline in `baseline/tokenwise.py`.
- Implemented a finite autoregressive benchmark in `data/finite_ar.py`.
- Implemented artifact-only evaluation in `eval/evaluate.py`.
- Evaluation now derives fixed `gamma`, sequence length, and temperature from
  `data/finite_ar.py` rather than trusting method-authored artifact fields.
- Continuing-agent audit confirmed `eval/` does not import from `method/` and
  evaluates method artifacts rather than rerunning the algorithm internally.
- A subsequent audit restored `core_claim_plus`: the implementation is a real
  scaled HSD verifier, `scripts/evaluate.sh` works standalone, and
  `scripts/baseline.sh` now evaluates when run directly.

## Results

Smoke test with 5,000 samples on `finite_ar_main`:

- Tokenwise block efficiency: 3.1437, total variation: 0.1346
- HSD block efficiency: 3.2911, total variation: 0.1316

The smoke test shows the expected block-efficiency advantage. A larger CPU job
reduced sampling noise and evaluated all configured settings.

CPU reproduction job `291c5539-26f` completed successfully with 80,000 samples
per method/setting:

| Experiment | Tokenwise BE | HSD BE | Tokenwise TV | HSD TV |
| --- | ---: | ---: | ---: | ---: |
| `finite_ar_main` | 3.1425 | 3.2886 | 0.0333 | 0.0325 |
| `finite_ar_gamma3` | 3.1417 | 3.2900 | 0.0304 | 0.0326 |
| `finite_ar_temp08` | 2.9916 | 3.1507 | 0.0312 | 0.0324 |

HSD improves block efficiency in all three finite settings while maintaining
distribution distance comparable to tokenwise speculative decoding.

Fresh CPU rerun job `5f35dfcd-72a` completed successfully and restored the
full 80,000-sample artifacts. Current milestone: `core_claim_plus`.

A follow-up implementation audit found no HSD correctness gaps in the finite
implementation. A lightweight exact enumeration of the finite AR setup matched
the target length-4 distribution to numerical precision (`TV ~= 6.4e-17`),
confirming that the empirical TV values below are sampling noise rather than a
systematic fidelity failure.

`scoring/reference.json` uses finite-benchmark block-efficiency targets in the
same units as `scoring/scores.json`. The paper's original Qwen/GSM8K numbers
remain documented in `briefing/overview.md`.
For the fidelity constraint, `reference.json` uses `total_variation=0.0` as the
exact target-distribution ideal; `scores.json` reports finite-sample empirical
TV and should be interpreted with sampling noise.

Verifier audit on 2026-05-27 found no surrogate implementation or eval/method
separation issue. It did identify that the default `scores.json` omitted the
extra diagnostic metrics declared in `reference.json`; `eval/evaluate.py` now
emits those full-support diagnostics in the default score file as well as the
primary block-efficiency and total-variation metrics. `bash scripts/evaluate.sh`
and `python validate.py --compare` both pass after this fix.

Final sub-agent audits on 2026-05-27 reconfirmed the same conclusion: the code
implements capped-ratio HSD with backward verification and one capped branch
resample, not a generic surrogate. The current claim remains scoped to the
scaled finite autoregressive verifier benchmark. `bash scripts/evaluate.sh` and
`python validate.py --compare` pass after the final documentation cleanup.

## Remaining

- Higher milestones would require running the verifier with real LLM draft and
  target models on GSM8K/HumanEval/CNN-DailyMail.
- Reproducing the EAGLE-3 integration would require implementing or porting a
  multi-draft tree framework and running model-scale benchmarks.

## Issues

- The paper's original 14B-72B experiments are too large for a first verifier
  reproduction. The current benchmark scales resources down to exact finite
  autoregressive distributions while preserving the verifier computation.
- The scheduler's quoted absolute log path was not mounted directly, but the
  same logs were available under `jobs/291c5539-26f/`.
- Job `5f35dfcd-72a` was initially preempted on one test node and then reran
  successfully on another node; final exit code was 0.
- `scoring/reference.json` contains finite scaled targets in the same units as
  this benchmark; the original paper's large-model numbers are documented in
  `briefing/overview.md`.
- Continuing audit found and fixed stale documentation that implied
  `scoring/reference.json` held paper table values; it now consistently
  describes the finite scaled benchmark reference values used by the evaluator.
- A later audit found default scoring did not emit all diagnostic metric keys
  declared in `reference.json`; this is fixed in `eval/evaluate.py`.

## Deviations from Paper

- Paper used Qwen2.5-0.5B as draft and Qwen2.5-14B/32B/72B as target on GSM8K,
  HumanEval, and CNN/DailyMail. This reproduction uses finite autoregressive
  target/draft distributions with exact probability tables because the original
  model scale is expensive for an initial verifier-focused reproduction.
- The algorithm itself is not replaced: the implementation still uses the
  paper's capped-ratio HSD verification and resampling rules.
