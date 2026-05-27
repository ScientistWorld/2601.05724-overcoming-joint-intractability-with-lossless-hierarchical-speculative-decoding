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
is submitted to reduce sampling noise and evaluate all configured settings.

## Remaining

- Run the full CPU reproduction job and inspect `scoring/scores.json`.
- If sampling noise keeps total variation too high, increase samples or replace
  sampling artifacts with exact finite-enumeration artifacts.
- Higher milestones would require running the verifier with real LLM draft and
  target models on GSM8K/HumanEval/CNN-DailyMail.

## Issues

- The paper's original 14B-72B experiments are too large for a first verifier
  reproduction. The current benchmark scales resources down to exact finite
  autoregressive distributions while preserving the verifier computation.

## Deviations from Paper

- Paper used Qwen2.5-0.5B as draft and Qwen2.5-14B/32B/72B as target on GSM8K,
  HumanEval, and CNN/DailyMail. This reproduction uses finite autoregressive
  target/draft distributions with exact probability tables because the original
  model scale is expensive for an initial verifier-focused reproduction.
- The algorithm itself is not replaced: the implementation still uses the
  paper's capped-ratio HSD verification and resampling rules.
