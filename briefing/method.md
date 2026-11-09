# Paper's Method

How the paper approaches the problem, i.e. the paper's core contributions.

## Key Contribution

Hierarchical Speculative Decoding (HSD) is a lossless verifier that uses
hierarchical branch resampling instead of independent token-wise checks, so it
can accept longer draft prefixes.

## Approach (single draft, Algorithm 2)

Along the draft the verifier computes joint prefix ratios r(X_{1:i}) =
p(X_{1:i})/q(X_{1:i}). It caps them at the maximum-prefix-ratio point
(Definitions 4–5), computes capped branch divergences over each accessible
next-token branch (Definition 6), and scans backward from the end of the draft,
accepting prefix X_{1:t} with probability h_t (Equation 19). If the whole draft
is accepted, a bonus token is sampled from the target. Otherwise one token is
drawn from the capped branch resampling distribution (Equation 20).

## Multi-draft (Algorithm 3)

Drafts are visited in order. A draft is verified only if it extends the
currently accepted prefix, starting from that node with the previous capped
residual installed as the node's target distribution (recursive rejection
sampling with replacement). The final residual is sampled when the drafts run
out.

## Implementation in this workspace

`method/hsd_verifier.py` implements Algorithms 2 and 3 and reports their exact
outcome distribution (protocol in `briefing/evaluation.md`).
`scripts/method.sh` exports it to `checkpoints/hsd/verifier.py`.

## Main Claims

HSD improves block efficiency and decoding speed over token-wise and blockwise
verification while remaining lossless. It is robust across temperatures and
draft lengths, and it also improves multi-draft decoding and EAGLE-3.

## Notes from this reproduction

* Single-draft HSD (Algorithm 2) is algebraically identical to block
  verification (Sun et al., 2024). The capped scale r(X_{1:t})/max(1, max_{i<=t}
  r(X_{1:i})) equals block verification's running term p_t, and the
  denominators of h_t agree. The two methods therefore have identical expected
  efficiency. The paper's reported HSD-over-Blockwise margins are not reproduced
  (exact tie).
* HSD over token-wise verification reproduces in every setting (+3.7% to +6.1%
  single draft, +2.6% multi-draft on the held-out slice).
