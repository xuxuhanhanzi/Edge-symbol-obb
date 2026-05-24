# Paper Claim Boundaries

Use this reference when converting results into paper text.

## Evidence and Claims

| Evidence | Safe claim | Unsafe claim |
|---|---|---|
| Only PyTorch metric improves | Improves FP32 training/evaluation metric | Improves deployment stability |
| Deployment drop is smaller | Improves quantization/deployment robustness in this setup | Works on all hardware |
| One class improves | Improves that class or suggests class-specific benefit | Improves all categories |
| e20 quick result | Early trend or ablation signal | Final result |
| e100 plus export plus deployment | Formal project result | Universal state-of-the-art |
| Paper inspiration only | Method is inspired by the paper | Method fully reproduces the paper |

## Claim Formula

Use:

```text
We propose X to address Y. X is inspired by A/B/C.
Under setting Z, X achieves M compared with baseline N.
In deployment evaluation, X shows P, so we describe the benefit as Q.
```

## Required Citation Discipline

- Main source: directly supports the module or method.
- Supplementary source: supports a sub-idea, evaluation, or background mechanism.
- Background source: explains history or foundation.
- Do not cite a background paper as the direct origin of a new module.

## Conservative Language

Use:

- "suggests"
- "indicates"
- "in our deployment setting"
- "under the tested AutoDL/RKNN pipeline"
- "improves quantization stability as measured by ..."

Avoid unless fully proven:

- "solves"
- "guarantees"
- "state-of-the-art"
- "universally robust"
- "hardware-independent"

## Limitations

Always record:

- dataset scope
- hardware or simulator scope
- implementation differences from cited papers
- untested baselines
- unresolved failure modes
