---
name: research-project-planner
description: End-to-end research project planning and experiment management for machine learning and computer vision projects. Use when Codex needs to help with innovation-point preparation, recent top-conference paper discovery, paper-to-method mapping, experiment planning, AutoDL run execution, staged experiment logging, result analysis, deployment evaluation, ablation planning, or paper-claim boundary control.
---

# Research Project Planner

Use this skill to manage a research project as a reproducible evidence chain:

```text
project context -> paper discovery -> paper reading -> innovation mapping
-> experiment plan -> AutoDL execution -> result analysis -> paper claims
```

## Core Rules

- Read existing `docs/`, experiment records, configs, scripts, and user-provided logs before proposing new work.
- Prefer papers from the last three years and prioritize top venues in the target field.
- Record paper URLs, code URLs, method details, experiment settings, and relevance judgments.
- Treat official or widely accepted baselines as the main comparison target; treat local modified models as engineering baselines unless the user states otherwise.
- Plan each stage before running it, and create a stage record for every meaningful experiment.
- Keep one primary variable per experiment whenever possible.
- For deployment-oriented projects, evaluate PyTorch, exported model, target-platform model, INT8 drop, latency, and model size.
- Do not overclaim. Tie every claim to a paper, code change, command, result table, or log.

## Workflow

1. **Project intake**
   - Scan existing project docs and code.
   - Summarize task, dataset, current baseline, current progress, and unresolved risks.
   - For detailed guidance, read `references/experiment_record_template.md`.

2. **Paper discovery**
   - Search recent top-conference or high-quality journal papers.
   - Record title, year, venue, paper link, code link, method summary, and project relevance.
   - Use `references/paper_discovery.md`.

3. **Paper reading**
   - Create one Chinese paper note per selected paper.
   - Explain the problem, innovation, principle, experiment setting, results, limitations, and project adaptation.
   - Use `references/paper_reading_template.md`.

4. **Innovation mapping**
   - Map each candidate innovation to a paper source, project module, expected benefit, baseline, metric, and risk.
   - Use `references/innovation_mapping.md`.

5. **Experiment planning**
   - Define hypothesis, primary variable, fixed settings, baselines, metrics, success gate, failure handling, and artifacts.
   - Use `references/experiment_planning.md`.

6. **AutoDL execution**
   - Default to AutoDL as the remote experiment platform unless the user says otherwise.
   - Record environment, paths, commands, conda envs, logs, weights, exports, and platform-specific issues.
   - Use `references/autodl_runbook.md`.

7. **Result analysis**
   - Compare PyTorch, ONNX, deployment model, per-class AP, INT8 drop, latency, and artifacts.
   - Use `references/result_analysis.md`.

8. **Paper claim control**
   - Convert results into conservative paper-ready claims.
   - Mark unsupported claims as unverified.
   - Use `references/paper_claims.md`.

## Optional Scripts

Use the scripts only when they match the project layout:

- `scripts/init_stage_record.py`: create a stage experiment record from a template.
- `scripts/collect_run_summary.py`: extract high-level metrics from common run/log files.
- `scripts/update_experiment_table.py`: append or update a markdown experiment table.

Run scripts from the project root and inspect generated output before relying on it.

## Output Standards

Prefer concrete artifacts over abstract advice:

- `docs/project_context_summary.md`
- `docs/paper_reference_relevance_audit.md`
- `docs/paper_<short_name>_cn.md`
- `docs/<module>_design.md`
- `docs/experiments/<date>_<stage>_<run>.md`
- consolidated experiment tables

If the evidence is incomplete, state exactly what is missing and what experiment or reading task would resolve it.
