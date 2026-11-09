# Progress

## Current state (iteration 11, hardened)

- `method/hsd_verifier.py`: the paper's HSD. Algorithm 2 (capped prefix ratios,
  capped branch divergences, Eq. 19 backward scan, Eq. 20 resample or bonus
  token) and Algorithm 3 (HSD + recursive reject sampling with replacement).
  It is a verifier program that reports exact outcome distributions;
  `scripts/method.sh` exports it to `checkpoints/hsd/verifier.py`.
- Baselines: `baseline/tokenwise_verifier.py` (Leviathan et al.; multi-draft
  RRS with replacement), `baseline/blockwise_verifier.py` (Sun et al. 2024,
  first draft only).
- Evaluation: `eval/train` (visible model pairs from `data/synthetic_lm.py`)
  and `eval/test` (self-contained, hidden seeds). Both run each program in a
  subprocess, enumerate every draft tuple, score exact block efficiency and gate
  on exact losslessness.
- Experiments: single_draft_main (0.35), single_draft_temperature (0.10),
  single_draft_length (0.15), multi_draft (0.40).

## Results (held-out slice, exact)

| Experiment | tokenwise | blockwise | HSD | HSD vs tokenwise |
|---|---:|---:|---:|---:|
| single_draft_main | 3.1546 | 3.3469 | 3.3469 | +6.10% |
| single_draft_temperature | 3.5041 | 3.6467 | 3.6467 | +4.07% |
| single_draft_length | 3.4109 | 3.5383 | 3.5383 | +3.74% |
| multi_draft | 3.4233 | 3.1881 | 3.5130 | +2.62% |

All three are lossless (max completion TV 1.1e-15). Single-draft HSD is
algebraically identical to block verification, so the paper's HSD > Blockwise
margins do not reproduce (exact tie). See `HARDENING.md` for the headroom
check, variants and shortcut results.

## Remaining / not reproduced

- Real LLM pairs (Qwen2.5 0.5B with 14B–72B), gamma = 10, 11 drafts, and
  decoding speed: scaled down to synthetic pairs with exact enumeration.
- EAGLE-3 integration.
