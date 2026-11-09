# Overview

- **Paper ID:** 2601.05724
- **Title:** Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding
- **Domain:** Language model inference / speculative decoding
- **TL;DR:** HSD is a lossless speculative decoding verifier that accepts longer draft prefixes by redistributing branch probability mass hierarchically.

## Short Summary

The paper targets the verification bottleneck in speculative decoding. It
proposes capped branch resampling, which lets the verifier perform a backward
block-level acceptance scan and still recover the target distribution with a
single resampling step. The reported experiments show consistent block
efficiency and decoding-speed gains over tokenwise and blockwise verification,
and gains when combined with multi-draft recursive rejection sampling.

## Key Results (paper)

- Average reported improvement is about 6.2% in block efficiency and 6.7% in
  decoding speed across main Qwen2.5 experiments.
- On GSM8K with Qwen2.5-14B, reported block efficiency improves from 5.99
  tokenwise to 6.30 with HSD.
- Multi-draft (11 drafts): +5.9% block efficiency over tokenwise multi-draft.
- EAGLE-3 integration reports a 12.4% decoding-speed gain.

## This gym

Synthetic target/draft pairs with exact scoring. Methods are verifier
programs run on hidden model pairs (see `briefing/evaluation.md`). Four
experiments: single-draft main, temperature, draft length, and multi-draft.
