# Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2601.05724"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2601.05724-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2601.05724](https://arxiv.org/abs/2601.05724) |
| **Domain** | Language model inference / speculative decoding |
| **TL;DR** | HSD is a lossless speculative decoding verifier that accepts longer draft prefixes by redistributing branch probability mass hierarchically. |

## Summary

The paper targets the verification bottleneck in speculative decoding. It
proposes capped branch resampling, which lets the verifier perform a backward
block-level acceptance scan and still recover the target distribution with a
single resampling step. The reported experiments show consistent block
efficiency and decoding-speed gains over tokenwise and blockwise verification.

## Problem

How can an autoregressive generator reduce the number of serial target-model
decoding steps while still sampling from the same distribution as the target
model? The challenge is that multiple proposed tokens can have dependencies
that are easy to exploit for speed but hard to validate without biasing the
output distribution.

## The paper's method

The paper introduces Hierarchical Speculative Decoding (HSD), a lossless
verification method that increases accepted tokens in speculative decoding by
using hierarchical branch resampling rather than independent tokenwise checks.

## Key results of the paper

- Average reported improvement is about 6.2% in block efficiency and 6.7% in
- On GSM8K with Qwen2.5-14B, reported block efficiency improves from 5.99
- EAGLE-3 integration reports a 12.4% decoding-speed gain.

## Run it

```bash
pip install sciworld
sw research 2601.05724 --mode innovate --agent claude-code
sw evaluate 2601.05724
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

