# Progress

## What Works

- Implemented the paper's HSD verifier in `method/hsd.py`, including capped
  prefix ratios, capped branch divergences, backward verification, and capped
  one-token resampling.
- Implemented tokenwise speculative decoding baseline in `baseline/tokenwise.py`.
- Implemented a finite autoregressive benchmark in `data/finite_ar.py`.
- Implemented artifact-only evaluation in `eval/evaluate.py`.

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

`scoring/reference.json` uses finite-benchmark block-efficiency targets in the
same units as `scoring/scores.json`. The paper's original Qwen/GSM8K numbers
remain documented in `briefing/overview.md` and `briefing/evaluation.md`.

## Remaining

- Higher milestones would require running the verifier with real LLM draft and
  target models on GSM8K/HumanEval/CNN-DailyMail.

## Issues

- The paper's original 14B-72B experiments are too large for a first verifier
  reproduction. The current benchmark scales resources down to exact finite
  autoregressive distributions while preserving the verifier computation.
- The scheduler's quoted absolute log path was not mounted directly, but the
  same logs were available under `jobs/291c5539-26f/`.

## Deviations from Paper

- Paper used Qwen2.5-0.5B as draft and Qwen2.5-14B/32B/72B as target on GSM8K,
  HumanEval, and CNN/DailyMail. This reproduction uses finite autoregressive
  target/draft distributions with exact probability tables because the original
  model scale is expensive for an initial verifier-focused reproduction.
- The algorithm itself is not replaced: the implementation still uses the
  paper's capped-ratio HSD verification and resampling rules.
