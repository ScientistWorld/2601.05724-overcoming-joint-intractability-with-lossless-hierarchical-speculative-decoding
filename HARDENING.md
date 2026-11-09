# Hardening report (iteration 11)

**Better results now score higher, and every shortcut that the red team ran is closed.** The old gym ranked submitted histograms and self-reported step counters on a 625-sequence target that anyone could enumerate. The new gym ranks a *verifier program*. The hidden test runs it on hidden target/draft model pairs, shows it only what a real verifier can see, and computes block efficiency and losslessness **exactly**. A lossy, malformed, nondeterministic or crashing program scores as failed. The paper's direction reproduces in all four experiments. A genuinely better verifier beats the reference in the new multi-draft experiment on 18 of 18 settings. Seed noise is zero (a rerun in a different order moves the score by at most 1.1e-14).

All numbers below come from job `28904502-86a` (exit 0, `validate.py --post-design` passed). Raw files are in `eval/test/hardening/results/`, which is hidden along with the test slice.

## 1. What changed and why

| Problem found by the red team | Fix |
|---|---|
| Evaluator trusted method-written `generated_tokens`, `target_steps` and histograms | Methods submit a program, not numbers. The evaluator streams verification problems to it (`python3 -I`, own subprocess, one request at a time) and receives outcome distributions `[draft, tau, token, prob]`. Each request is one target step by construction, and emitted length is `tau + 1 <= gamma + 1`. All scores are recomputed by the evaluator. |
| Ground truth imported from editable `data/finite_ar.py`; enumerable 625-sequence target | Hidden test instances are generated inside `eval/test/evaluate.py` with a private salt and seed. Nothing there imports `data/`, `method/` or `eval/train/`. The program never receives the model, only the rows along its own drafts, so it cannot enumerate or sample the target. |
| Fidelity was an additive penalty, so a lossy decoder could buy rank with efficiency | Losslessness is a hard gate, checked exactly at every context: emitted prefix plus target continuation must equal the target over gamma + 1 tokens (TV <= 1e-9). Failure, or any protocol violation (non-finite, negative or unnormalised probabilities, out-of-range outcomes, different answers to a repeated problem, crash, timeout), scores 0.0 in every experiment. Every valid decoder scores at least 1.0, so a failed method is strictly last in every metric in any population. |
| "Better" meant more Monte Carlo samples, and seeds produced noise | Exact enumeration of every draft (tuple) at every context, so there is no sampling. A rerun with a different stream order differs by at most 1.1e-14. |
| `main` and `gamma3` were the same experiment; duplicated benefit metrics; reference values missing | New experiments mirror the paper's tables, with one ranked metric each (`block_efficiency`, coefficient 1). Reference values are the exact held-out values, and the paper's numbers are kept under `_paper_reported`. |
| Baseline artifacts could be swapped | The evaluator computes its own token-wise value (`fixed_baseline`) on the same instances. |

The paper's method is unchanged: `method/hsd_verifier.py` implements Algorithm 2 (capped prefix ratios, capped branch divergences, Eq. 19 backward scan, Eq. 20 resample or bonus token) and Algorithm 3 (HSD + recursive reject sampling with replacement) and reports their exact outcome distributions. Its acceptance probabilities match the old sampling implementation `method/hsd.py` to 9.2e-15 (16,436 draft positions checked, lengths 2–4, temperatures 1 and 0.8).

## 2. Reproduction finding: HSD equals block verification

The capped scale in HSD, r(X_{1:t}) / max(1, max_{i<=t} r(X_{1:i})), equals block verification's running term p_t (the paper's own Lemma 11). HSD's Eq. 19 denominator equals D*(p,q) + (1 − p_t), which is block verification's denominator. So Algorithm 2 **is** block verification (Sun et al., 2024). I confirmed this numerically: h_t agrees to 1e-14 on every draft, and expected accepted tokens are identical on every instance. Block verification is proven per-step optimal for a single draft. Consequences:

- The paper's HSD > Blockwise margins (Table 1, Theorem 7) do not reproduce; the two tie exactly. `blockwise` is kept as a baseline to show this.
- The single-draft experiments put the reference **at the bound**: no lossless single-draft verifier can beat it, only tie or lose.
- The paper's claims against token-wise verification do reproduce, on the held-out slice:

| Experiment | Paper analogue | token-wise | HSD (reference) | gain |
|---|---|---:|---:|---:|
| single_draft_main | Table 1 | 3.1546 | 3.3469 | +6.10% |
| single_draft_temperature | Table 3a | 3.5041 | 3.6467 | +4.07% |
| single_draft_length | Table 3b | 3.4109 | 3.5383 | +3.74% |
| multi_draft | Table 2 | 3.4233 | 3.5130 | +2.62% |

HSD beats token-wise multi-draft on 18 of 18 multi-draft settings. Train-slice gains are similar (6.19 / 4.39 / 4.44 / 2.31%), so the two slices measure the same thing.

## 3. Headroom check

- **At the bound:** all three single-draft experiments (proved optimum, attained by the reference).
- **Added:** `multi_draft` (paper Table 2 / Algorithm 3; 2–3 i.i.d. drafts), weight 0.40. The paper itself notes Algorithm 3 wastes drafts whose prefix does not match the accepted prefix. A tree verifier that uses an optimal (max-flow) coupling of the drafts' next tokens at shared nodes, and HSD on single chains (hidden variant `tree_coupling`), scores 3.6980 vs 3.5130 (+5.3%). It is higher on **18 of 18** settings (+0.8% to +9.0%) and ties the reference on the single-draft experiments. The comparison is paired: every method sees the same draft tuples and the noise is about 1e-14, so the gain is far above noise. The optimum for this experiment is unknown, so `tree_coupling` is evidence of headroom, not a ceiling.
- **Brute force:** a program cannot enumerate the target, because it never receives it. The one route is accumulating rows across calls. Problems are streamed one at a time in a random global order over about 480k problems, so the complete tree of any context is known only after all of its drafts have been answered (each leaf row appears in exactly one draft). Answers must also be deterministic and are re-checked.
- **Simple baselines** score clearly below the reference: token-wise everywhere; first-draft-only `blockwise` on multi-draft (3.1881).

## 4. Do better results score higher? (variants)

Rank = Σ_exp weight · Σ_metric |coef| · dir · (x − x_ref) / std_pop, with x_ref the frozen `hsd` values in `scoring/scores_test.json` (`eval/test/hardening/rank.py`).

**Before** (old gym, red team's honest population; the only "better" route was more Monte Carlo samples):

| Variant | Rank before |
|---|---:|
| 8x more HSD samples (not a better method) | +1.001 |
| reference seeds 2/3 | −0.84 / −0.72 (noise range 0.84) |
| fewer samples | −5.230 |
| token-wise | −5.584 |

**After** (held-out block efficiency; honest population = rows 1–6; all = plus the 10 shortcuts):

| Attempt | main | temperature | length | multi_draft | Rank (honest) | Rank (all) |
|---|---:|---:|---:|---:|---:|---:|
| tree_coupling (better) | 3.3469 | 3.6467 | 3.5383 | **3.6980** | **+0.488** | +0.044 |
| hsd (reference) | 3.3469 | 3.6467 | 3.5383 | 3.5130 | 0 | 0 |
| hsd rerun, different order | 3.3469 | 3.6467 | 3.5383 | 3.5130 | 0.000 | 0.000 |
| tree_rrs (weaker multi-draft) | 3.3469 | 3.6467 | 3.5383 | 3.4422 | −0.187 | −0.017 |
| blockwise (first draft only) | 3.3469 | 3.6467 | 3.5383 | 3.1881 | −0.858 | −0.077 |
| tokenwise | 3.1546 | 3.5041 | 3.4109 | 3.4233 | −1.847 | −0.083 |
| each shortcut (§5) | 0 | 0 | 0 | 0 | — | −2.086 |

The ordering is the scientifically correct one. The "all" column is compressed only because the failed shortcuts inflate each metric's spread.

## 5. Shortcuts: fix and score before → after

Every program below was run with `scripts/evaluate_test.sh` (`eval/test/hardening/run_hardening.sh`). "After" = failed (0.0 in every experiment, rank −2.086 in the full population, strictly last in any population).

| Red-team shortcut | Rank before | Ported attack (after) | Result after |
|---|---:|---|---|
| counters_shortcut (impossible step count) | **+8.280** | `mass_inflation` (genuine HSD outcomes × 1.25); `tau_overflow` (claims gamma + 1 accepted) | failed: "probabilities sum to 1.25"; "outcome out of range" |
| answers_shortcut (exact histogram, no decoding) | **+0.264** | impossible: no histogram or count is read; only per-problem outcome distributions from a program run on hidden models | closed by construction |
| brute_force_shortcut (enumerate 625 sequences) | **+0.594** | impossible: the program never receives the model (§3) | closed by construction |
| biased_answers_shortcut (slightly wrong distribution) | **+0.242** | `inflated_hsd` (acceptances × 1.02); `uncapped_hsd` (paper's Table A.3 no-capping ablation) | failed: TV 1.56e-2; TV 2.34e-1 |
| fixed_truth_shortcut (edit `data/` truth) | **+0.264** | make the visible draft equal the target in `data/synthetic_lm.py` | test score change **0.0** in all experiments; only train moves (to 5.0/5.0/4.5/4.33) |
| baseline_move_shortcut | 0.000 (tie) | degenerate program named `tokenwise` | it fails; the evaluator's `fixed_baseline` is identical (3.154573 / 3.504090 / 3.410852 / 3.423254) |
| self_fields_shortcut | −0.00006 | not expressible: score files are written only by the evaluator from outcomes | closed |
| draft_only_shortcut | −1.32 (all-finite); **+0.158** (task-failure population) | `draft_only` (accept the whole draft every time) | failed: TV 0.76 |
| constant / constant_counters | −7.50 | `constant` | failed: TV 1.0 |
| nonfinite_shortcut (NaN accepted) | NaN scored | `nan_probs` | failed: non-finite probability |
| empty_shortcut (evaluator crashed) | exit 1 | `empty_outcomes` | failed cleanly (evaluator exits 0) |
| fixed_setting_shortcut | failed | settings live in the evaluator; `tau_overflow` | failed |
| (new) program crash | — | `crash` | failed: no response |
| (new) call-order-dependent program | — | `call_dependent` | failed: determinism probe |

## 6. Reference score before and after

- **Before** (old metrics, `redteam/reference/scores_test.json`): normalised slice block efficiency 3.2890 (main), 3.2886 (gamma3), 3.1709 (temp08), with empirical TV about 0.032 (sampling noise; seed range 0.84 rank units).
- **After** (exact, held-out): 3.3469 (single_draft_main), 3.6467 (temperature), 3.5383 (length), 3.5130 (multi_draft). Maximum completion TV is 1.1e-15 (lossless). The setups differ, so the numbers are not comparable across the two gyms.

## 7. Weights

Experiment weights are main 0.35, temperature 0.10, length 0.15, multi_draft 0.40 (old: 0.6 / 0.2 / 0.2 over three near-duplicate settings). Multi-draft is one of the paper's three stated contributions (Table 2, Appendix G, the EAGLE-3 tree setting). It is also the only experiment where a better verifier can score higher, because the reference is provably optimal in the others. The single-draft experiments keep 0.60 together: they carry the paper's main claim and penalise methods that regress there.

## 8. Left open, stated plainly

- **Reading the hidden evaluator from disk.** A program runs on the same filesystem as `eval/test/` during the hidden job and could read the private salt and seed. No filesystem sandbox is available. This is left to the shortcut grader, and `scoring/CONSTRAINTS.md` forbids it.
- **Cross-call pooling.** A program that memoises its answers and pools rows across requests could in principle use partial off-path information. Random global ordering, one-at-a-time streaming and the determinism probe make this impractical (the full tree is known only after the last draft of a context), and an answer mixture built from partial information would have to stay exactly lossless. It is forbidden in `CONSTRAINTS.md`, and the grader should check for state carried across requests.
- **Scale.** Vocabularies of 3–6 tokens, draft lengths 2–5 and 2–3 drafts replace Qwen2.5 0.5B/14B–72B with gamma = 10 and 11 drafts, because every draft (tuple) is enumerated for exact scoring. Decoding speed (tokens/s) is not measured, by design (no wall-clock metrics). The EAGLE-3 integration is not reproduced.
- **Single-draft experiments cannot reward improvement** (proved optimum). They only reward matching it.
- **Old audit material.** `redteam/` scripts refer to the retired histogram contract and the deleted `data/finite_ar.py`; they document the pre-hardening gym (commit `cba0698`) and do not run against the new code.
