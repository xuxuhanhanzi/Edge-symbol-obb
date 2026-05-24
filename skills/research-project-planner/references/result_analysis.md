# Result Analysis

Use this reference after experiments finish.

## Main Result Table

```markdown
| Model | Main variable | PyTorch mAP50 | PyTorch mAP50-95 | ONNX mAP50 | ONNX mAP50-95 | Deployment mAP50 | INT8/drop | Latency | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
```

## Ablation Table

```markdown
| Module | Variant | Controlled variable | mAP50 | mAP50-95 | Deployment mAP50 | Conclusion |
|---|---|---|---:|---:|---:|---|
```

## Analysis Checklist

Check:

- Did PyTorch improve?
- Did ONNX match PyTorch?
- Did deployment model preserve ONNX performance?
- Did mAP50 and mAP50-95 move in the same direction?
- Did specific classes improve or regress?
- Did latency or model size change?
- Did a failure mode disappear?
- Is the difference large enough to support a claim, or only a trend?

## Drop Calculations

Use:

```text
onnx_drop = pytorch_metric - onnx_metric
deployment_drop = onnx_metric - deployment_metric
total_drop = pytorch_metric - deployment_metric
```

If a method has slightly lower PyTorch accuracy but smaller deployment drop, describe it as deployment-stability evidence, not general accuracy improvement.

## Decision Categories

| Category | Meaning |
|---|---|
| Select | Use for full or final experiment |
| Keep as ablation | Useful evidence but not final method |
| Retest | Result unclear or environment suspect |
| Reject | Worse result or unacceptable risk |

## Per-Class Analysis

When per-class AP is available, identify:

- classes driving the improvement
- classes driving regression
- whether changes match the method's intended effect
- whether small classes or difficult classes are affected
