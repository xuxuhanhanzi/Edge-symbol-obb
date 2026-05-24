# Innovation Mapping

Use this reference to turn papers into project modules.

## Mapping Table

```markdown
| Innovation module | Main paper source | Supplementary source | Project problem | Proposed change | Main baseline | Engineering baseline | Expected benefit | Metrics | Risk |
|---|---|---|---|---|---|---|---|---|---|
```

## Required Questions

For every innovation candidate, answer:

1. What concrete project problem does it solve?
2. Which paper is the main source?
3. Which papers are only supplementary or background?
4. What code module changes?
5. What stays fixed?
6. What is the official or accepted baseline?
7. What is the local engineering baseline?
8. What result would prove the hypothesis?
9. What result would falsify or weaken the hypothesis?
10. What deployment risk exists?

## Comparison Rules

- Main comparison should use official or accepted baselines.
- Local modified models are engineering baselines unless they are the research target.
- Do not compare a new method only against another modified local model unless the user explicitly asks.
- Keep ablation variables isolated.

## Benefit Types

Classify expected benefit:

| Type | Evidence needed |
|---|---|
| Accuracy | PyTorch and exported-model metrics |
| Stability | variance, convergence, failure rate, or boundary-case metric |
| Quantization robustness | ONNX-to-INT8 drop, calibration sensitivity, deployment AP |
| Speed | latency on target or simulator with caveat |
| Size | parameters, FLOPs, exported model size |
| Generalization | validation/test split or stress-set performance |

## Feasibility Score

Score each candidate:

| Score | Meaning |
|---|---|
| 5 | Small code change, clear paper source, easy metrics |
| 4 | Moderate code change, likely feasible |
| 3 | Useful but needs interface audit |
| 2 | High engineering risk or unclear data fit |
| 1 | Not suitable now |

Use the score to decide implementation order.
