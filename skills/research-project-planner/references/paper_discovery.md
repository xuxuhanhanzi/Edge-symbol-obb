# Paper Discovery

Use this reference when preparing innovation points or selecting papers.

## Venue Priority

For computer vision and ML projects, prioritize:

- Vision: CVPR, ICCV, ECCV, WACV, TPAMI, IJCV, TIP
- ML: NeurIPS, ICLR, ICML, AAAI, IJCAI
- Multimedia: ACM MM
- Deployment, systems, and acceleration: MLSys, DAC, DATE, ASPLOS, RTAS
- Quantization and efficient inference: CVPR, ICCV, ECCV, NeurIPS, ICLR, MLSys, DAC, TPAMI, TNNLS

Prefer the last three years. Use older papers only as background, foundations, or when no recent paper covers the topic.

## Search Strategy

Build keyword groups:

```text
task keywords + method keywords + deployment keywords + dataset/domain keywords
```

Examples:

```text
oriented object detection angle representation CVPR 2025
rotated object detection Gaussian bounding box Cholesky
object detection post-training quantization regression branch
edge deployment INT8 object detection NPU
lightweight oriented object detection industrial symbols
```

Search sources:

- official conference proceedings
- arXiv
- OpenReview
- Semantic Scholar or Google Scholar
- Papers with Code
- official GitHub repositories
- project pages linked by papers

## Selection Criteria

For each paper, classify relevance:

| Level | Meaning |
|---|---|
| Strong | Directly supports the proposed module or method |
| Medium | Supports a related design or evaluation idea |
| Background | Explains a foundation but is not the main source |
| Not relevant | Similar words but not useful for this project |

Prefer papers that:

- address a problem the project actually has
- include clear experiments and baselines
- expose implementable modules
- have code or precise implementation details
- fit target constraints such as AutoDL training or edge deployment

Reject papers that:

- are recent but not relevant
- only improve a task unrelated to the project
- require data, operators, or hardware unavailable to the project
- cannot be mapped to a testable project hypothesis

## Paper Audit Table

Use this table in `docs/paper_reference_relevance_audit.md`:

| Innovation | Paper | Year/Venue | Paper URL | Code URL | Relevance | Project module | Expected benefit | Risk |
|---|---|---|---|---|---|---|---|---|

## Required Notes

Every selected paper must record:

- title
- authors if useful
- year and venue
- paper URL
- code URL or "not found"
- core problem
- core innovation
- experiment setting
- project relevance
- whether it is a main source, supplementary source, or background source
