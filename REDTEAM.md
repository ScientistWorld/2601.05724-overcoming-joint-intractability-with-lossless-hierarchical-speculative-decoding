# Scoring audit

**The basic honest ordering passes, but this gym does not reliably reward a better decoding method.** Eight times more genuine HSD samples score higher, and removing hierarchical verification or reducing samples scores lower, beyond the observed seed spread in the honest comparison. However, the fidelity gains measure Monte Carlo sample budget rather than a better limiting distribution. Several shortcuts beat the reference, and an unverified draft-only decoder can outrank it despite failing losslessness. **Exhaustive enumeration solves this test: its 625 possible sequences are too few to require the paper's method.**

All comparisons below use the original `scoring/scores_test.json`, frozen in [redteam/reference/scores_test.json](redteam/reference/scores_test.json). A score is a normalized rank improvement over that reference, whose improvement is zero. Full metrics, populations, standard deviations, and individual contributions are in [redteam/summary.json](redteam/summary.json).

**Honest quality checks.** These are actual calls to the supplied HSD or tokenwise generator, with unchanged scenarios and correct work counters. Deterministic probability arithmetic is memoized without changing random draws. Weighted BE and TV below average metrics with experiment weights 0.6/0.2/0.2; they are diagnostic summaries, not the scalar rank.

| Variant | Samples per experiment | Weighted test BE ↑ | Weighted test TV ↓ | Rank improvement, honest population |
|---|---:|---:|---:|---:|
| Frozen reference / reference seed 1 | 80,000 | 3.265286 | 0.032489 | 0.000000 |
| Reference seed 2 | 80,000 | 3.253492 | 0.033901 | −0.840857 |
| Reference seed 3 | 80,000 | 3.255319 | 0.034097 | −0.720081 |
| [More genuine HSD samples](redteam/better_more_samples/run.sh) | 640,000 | 3.265698 | 0.011700 | **+1.001078** |
| [HSD idea removed: tokenwise verification](redteam/weaker_tokenwise/run.sh) | 80,000 | 3.123554 | 0.032289 | **−5.583893** |
| [Fewer genuine HSD samples](redteam/weaker_fewer_samples/run.sh) | 5,000 | 3.300025 | 0.134368 | **−5.230166** |

The honest population is exactly these six attempts. Its three-seed rank range is **0.840857**. The larger run's gain exceeds that observed range, and both weaker variants lose by much more. This is a more-data check, not a newly improved verifier: both exact HSD and exact tokenwise sampling already have zero true distribution error. Even the 5,000-sample run raises measured BE through slice-mass fluctuations while substantially worsening fidelity.

**Seed spread and reproduction check.** Reference runs use disjoint per-sequence seed ranges starting at 1, 1,000,001, and 2,000,001, each with 80,000 draws per experiment. Seed 1 reproduces every frozen metric within floating-point rounding. All eighteen metric spreads are retained in the summary; representative ranges are:

| Experiment | Test BE min–max | BE range | Test TV min–max | TV range |
|---|---:|---:|---:|---:|
| main | 3.277062–3.288956 | 0.011894 | 0.032465–0.034935 | 0.002470 |
| gamma3 | 3.286596–3.305476 | 0.018880 | 0.031128–0.033472 | 0.002344 |
| temp08 | 3.130797–3.170942 | 0.040145 | 0.032417–0.033572 | 0.001154 |

The three-seed rank range is population dependent: **0.012287** in the all-finite population and **0.033171** in the task-failure population defined below. Gains below the corresponding spread are noise under this audit's rule. Three seeds estimate an observed range, not a confidence guarantee.

The reproduction beats its tokenwise baseline in the paper's reported efficiency direction. Frozen test BE is 3.288956 vs 3.155959 (main), 3.288622 vs 3.137595 (gamma3), and 3.170942 vs 3.012299 (temp08). [Exact integration of all decoding paths](redteam/reference/exact_verification.json), run in the compute job, gives:

| Experiment | Exact HSD raw tokens/target step | Exact tokenwise raw tokens/target step | HSD gain |
|---|---:|---:|---:|
| main | 3.289378 | 3.143806 | 4.63% |
| gamma3 | 3.289378 | 3.143806 | 4.63% |
| temp08 | 3.148506 | 2.988975 | 5.34% |

Both algorithms' exact TV errors are below 1.3e−16. The implementation contains the actual single-draft HSD procedure: capped prefix ratios, capped branch divergences, Equation 19 acceptance, independent backward acceptance draws, and Equation 20 residual sampling (or a target bonus token). It is recognizable as the paper's Algorithm 2, with numerical epsilon/clamping guards.

The following settings and coverage depart from [the local paper](paper/2601.05724.md):

- Deterministic five-token finite target/draft functions replace GPTQ 8-bit Qwen2.5 models. The paper uses a 0.5B draft and 14B/32B/72B targets, with 72B as default.
- Four-token synthetic sequences replace GSM8K, HumanEval, and CNN/DailyMail prompts and continuations. The gym uses gamma 4/3 and temperatures 1/0.8; it does not reproduce the paper's LLM draft-length experiments.
- CPU Python simulation and 80,000 independent short samples replace the paper's GPU decoding setup (a single H20 96 GB by default). Reported work is a serial-step counter, without measured decoding tokens/second or model execution cost.
- The hidden benefit metrics multiply raw efficiency by empirical support-slice mass; this is a different quantity from the paper's raw block efficiency. Hash-selected support categories are not unseen prompts or targets.
- There is no blockwise baseline, multidraft reproduction, EAGLE-3 integration, or dataset-level LLM speed evaluation.

Moreover, main and gamma3 have exactly the same expected efficiency and distribution because length-four truncation makes their different draft lengths immaterial. They are redundant scientific conditions, despite different hash slices and random-sampling results.

**Bounds, baseline, and ranking.** None of the eighteen observed HSD values in the frozen test scores is already at its optimum. Each of the twelve fidelity/error values has lower bound zero and is positive. The reference's block-by-block procedure has raw efficiency headroom from about 3.15–3.29 to four tokens when every sequence finishes in one verifier step. Four is not a universal bound for methods that amortize a cached target across samples, as enumeration demonstrates below. A constant baseline scores far below the reference; ordinary tokenwise verification is also clearly below it in the honest rank. The problem is therefore not a perfect empirical reference. It is a small, fully enumerable target with forgeable evidence.

There is a reference inconsistency: `scoring/reference.json` puts **total_variation at its optimal bound of 0 in all three experiments** (`finite_ar_main`, `finite_ar_gamma3`, `finite_ar_temp08`). It declares but omits reference result values for `prefix_conditional_l1`, `slice_block_efficiency`, `slice_total_variation`, and `slice_mass_error` in each experiment. This report follows the user's specified complete observed `scores_test.json`, rather than inventing those missing values or mixing observed TV with ideal TV.

Ranks are computed as

`sum_experiments weight * sum_metrics abs(coefficient) * direction * (candidate − frozen_reference) / population_std(metric)`.

Direction is +1 for `higher_is_better` and −1 otherwise. Coefficient signs never reverse a direction. The coefficients are 1, 1, 1, 0.5, 0.6, and 0.4 for BE, slice BE, TV, prefix L1, slice TV, and mass error respectively. Population standard deviations use all attempts in the stated population, with no units-based raw sum. Using sample rather than population standard deviations would uniformly rescale a given population's scores, preserving its ordering.

Two additional populations make the dependence on other attempts explicit:

- **All-finite:** all sixteen attempts with finite successful test scores. NaN and the two rejected artifacts are excluded.
- **Task-failure:** the six honest attempts plus `draft_only_shortcut` and `constant_shortcut`. All eight attempts actually ran through the hidden test; this is a concrete possible ranking population, not a hypothetical score construction.

In the all-finite population, tokenwise's sign reverses to +0.003851, below the corresponding 0.012287 seed range; that reversal is not evidence of a real gain. More seriously, the **draft-only shortcut scores +0.157502 in the task-failure population**, above its 0.033171 seed range, despite weighted TV **0.184774** versus reference **0.032489**, and worse values in all twelve fidelity metrics. Thus additive fidelity penalties do not enforce the task constraint. This shortcut loses (−1.323283) in the all-finite population; its success is explicitly population dependent.

For a fixed target, the two benefit metrics are duplicates: `slice_BE = exact_slice_mass * normalized_BE`. After per-metric variance normalization, they contribute the same standardized signal twice. Biased empirical slice mass can increase both, even beyond four for normalized BE. The counter shortcut improves all six benefit values without changing any fidelity values, and the exact-answer shortcut improves all eighteen metrics; these wins do not depend on the population's finite positive scales.

**Shortcut runs.** Every linked folder has a `run.sh`, generation code through `redteam/generate.py`, its outcome, and its train/test scores when evaluation succeeds. Every hidden invocation was exactly `bash scripts/evaluate_test.sh`, without arguments, after the generator exited. The batch used a clean compute container and fresh candidate/train/test subprocesses, restoring original data, checkpoints, and scores between cases. It did not start a new container for each case. No candidate changed `eval/`, the hidden script, the evaluator's process, or the Python environment. Candidate generation used only future-agent-visible data/baseline/method code and its own generated artifacts; hidden scores and paper text were read only for this audit's analysis.

Scores in the table use the all-finite population unless expressly marked. Reference = 0.

| Shortcut and runnable evidence | What it does | Test score vs reference | Works or fails |
|---|---|---:|---|
| [counters_shortcut](redteam/counters_shortcut/run.sh) | Reuses genuine reference histograms; writes `target_steps=1` for 320,000 emitted tokens in each scenario. | **+8.279779**; weighted BE 320,407.916; fidelity unchanged | **Works.** The evaluator trusts an impossible work count. |
| [answers_shortcut](redteam/answers_shortcut/run.sh) | Enumerates the visible exact target; writes largest-remainder counts for a claimed trillion samples and one target step per sequence. No HSD sampling. | **+0.264403**; BE 4.000000; TV 7.50e−11 | **Works.** All eighteen metrics beat the reference without learning a method or knowing the hidden slice. |
| [brute_force_shortcut](redteam/brute_force_shortcut/run.sh) | Enumerates all 625 sequences, caches all 156 target conditionals including preparation, then really draws 640,000 IID samples from that table. | **+0.594351**; BE 16,401.502; TV 0.011511 | **Works.** About 0.16–0.17 seconds per scenario including enumeration and sampling. The full target is too small; enumeration cost amortizes away. |
| [biased_answers_shortcut](redteam/biased_answers_shortcut/run.sh) | Writes a deterministic histogram of `0.99*p + 0.01*uniform`, claiming a trillion samples. Its limiting distribution is deliberately wrong. | **+0.241539**; BE 4.001441; TV 0.002724 | **Works.** Seventeen metrics improve; true TV remains 0.002601 at temperature 1 and 0.003217 at 0.8. Biased analytic counts beat finite-sample lossless counts. |
| [fixed_truth_shortcut](redteam/fixed_truth_shortcut/run.sh) | Replaces the editable target and draft functions in `data/finite_ar.py` with independent uniform distributions; supplies equal counts. Protected evaluation imports the replacement truth. | **+0.264405**; BE 4.000000; TV 1.36e−16 | **Works.** All eighteen metrics improve against an easier, changed task. |
| [baseline_move_shortcut](redteam/baseline_move_shortcut/run.sh) | Retains the reference HSD candidate and replaces the supposedly fixed tokenwise artifacts with constant outputs and bad work counts. | **0.000000**, all eighteen candidate metrics tied | **Works (shortcut tie).** No better candidate. The displayed baseline moves, but does not improve the candidate's rank against the frozen reference. |
| [self_fields_shortcut](redteam/self_fields_shortcut/run.sh) | Writes invented score/success/timing fields and prewrites `scores_test.json`; histograms are unchanged and work counters imply raw BE of one. | −0.000059 | **Fails.** Hidden evaluation overwrites score files and ignores the extra fields. This tiny loss is below seed spread; six benefit metrics are nevertheless strictly worse. |
| [draft_only_shortcut](redteam/draft_only_shortcut/run.sh) | Samples the provided draft without any acceptance or residual correction; declares every draft block successful. | −1.323283 all-finite; **+0.157502 task-failure**; BE 4.138551; TV 0.184774 | **Works in the demonstrated task-failure population.** Wrong-distribution output can buy rank with benefit metrics. |
| [constant_shortcut](redteam/constant_shortcut/run.sh) | Always returns `AAAA`, with maximal claimed per-sequence efficiency. | −7.501839; BE 0; TV 0.996819 | **Fails.** This blindly chosen sequence has no selected test mass. |
| [constant_counters_shortcut](redteam/constant_counters_shortcut/run.sh) | Same constant output, additionally claiming one total target step. | −7.501839 | **Fails.** Multiplying zero selected mass still gives zero benefit. |
| [nonfinite_shortcut](redteam/nonfinite_shortcut/run.sh) | Reuses reference counts and writes NaN as the target-step count. | Test exits 0; both benefit metrics become NaN | **Invalid score accepted; paying rank not established.** Excluded from finite rank calculations. |
| [empty_shortcut](redteam/empty_shortcut/run.sh) | Writes zero samples, empty counts, and zero emitted tokens. | Test exits 1: `ZeroDivisionError` | **Fails.** Failure trace retained. |
| [fixed_setting_shortcut](redteam/fixed_setting_shortcut/run.sh) | Changes artifact gamma and sequence length to one while leaving the visible scenario definitions unchanged. | Test exits 1: gamma mismatch | **Fails.** Fixed setting validation works for this attempt. |

The fixed-truth shortcut demonstrates replacement of a dependency treated as fixed. This gym has no pretrained fixed-model artifact to retrain; its fixed model is the visible Python probability function. The baseline shortcut separately exercises replacement of a baseline artifact. Neither shortcut edits the protected evaluator.

The strongest next-iteration fixes are to protect the target and scenario definitions as evaluation dependencies; execute and independently account for candidate sampling work on a sufficiently large held-out workload; fix the sampling budget and reject nonfinite/impossible artifact values; and enforce distribution fidelity as a constraint before ranking efficiency. Removing duplicate benefit terms and completing the reference values would also make the rank easier to interpret. Hashing a subset of a publicly enumerable answer table does not provide a held-out scientific task.

**Reproducibility and cleanup.** Job `e9ce9711-ed8` completed with exit 0: nineteen cases, seventeen successful hidden invocations (including the NaN case), two explicit rejections, plus exact integration. Run `bash redteam/run_all.sh` as a CPU job, or any individual case's `run.sh`; fresh clones regenerate artifacts, while interrupted jobs can resume from their own cached outputs. Artifacts and full logs are ignored; score JSONs, generation diagnostics, failure traces, and outcomes are committed. The starting commit and pre-existing dirty files are recorded under `redteam/reference/`. Temporary changes outside the audit are restored to that starting commit; `action.yaml` is the required completion signal.
