# Project Structure

本仓库分成“核心代码、实验配置、数据入口、脚本入口、实验记录、本地产物”六类。

```text
Edge-symbol-obb/
  ultralytics/                 # Ultralytics 核心代码和本地模型修改
  configs/
    baseline/                  # 官方 OBB 架构参考
    rv1106/                    # 当前 RV1106 轻量化配置
  datasets/                    # 可提交的数据集 yaml 和说明
  scripts/
    data/                      # 数据转换、检查、统计脚本
    train/                     # 训练入口
    eval/                      # PyTorch/ONNX 评估入口
    export/                    # ONNX 导出入口
    deploy_rv1106/             # RKNN/RV1106 部署入口
  experiments/                 # 正式实验记录摘要
  docs/                        # 项目设计、实验协议和报告
  artifacts/
    local/                     # 本地大产物，默认不提交
```

## Root Directory Rule

根目录只保留项目级文件：

- README / license / citation
- packaging files
- experiment plan and upstream notice
- top-level config for docs/build tooling

训练、评估、导出、部署脚本不再放在根目录。

## Script Naming Rule

脚本文件名应能表达用途：

- `train_*`: 训练入口
- `eval_*` or `val_*`: 评估/验证入口
- `export_*`: 模型导出入口
- `convert_*`: 格式转换入口
- `check_*` or `analyze_*`: 数据/结果检查入口

## Artifact Rule

权重、ONNX、RKNN、推理图片、日志和校准列表默认放在 `artifacts/local/`、`runs/` 或外部存储中，不提交到 Git。正式论文表格只引用可追溯路径和摘要指标。
