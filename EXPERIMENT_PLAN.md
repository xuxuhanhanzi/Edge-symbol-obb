# Edge-symbol-obb Execution Plan

完整计划见桌面文件 `C:\Users\27475\Desktop\Edge-symbol-obb_实验计划.md`。本文件只记录当前仓库内的执行入口和阶段 0 交付物。

## Current Priority

当前第一步是阶段 0：

```text
冻结基础版本 -> 建立可复现实验台账 -> 拆清官方 baseline 与当前 RV1106 改动版
```

阶段 0 完成后，才进入正式 baseline 训练和 QG-OBB Head/SOF-FPN 实现。

## Stage 0 Deliverables

- `UPSTREAM.md`: 当前代码来源、版本、已有本地修改
- `LICENSE_NOTICE.md`: 上游许可证和本地发布注意事项
- `datasets/industrial_symbol.yaml`: 工业符号数据集入口
- `configs/baseline/yolov8n_obb_official_arch.yaml`: 官方 YOLOv8n-OBB 架构参考
- `configs/rv1106/yolov8n_obb_rv1106_m2.yaml`: 当前轻量化 RV1106 baseline 配置
- `docs/experiment_protocol.md`: 实验记录规则
- `docs/baseline_report.md`: baseline 结果台账
- `docs/paper_tables.md`: 论文表格占位
- `docs/project_structure.md`: 当前项目结构和脚本归档规则

## Next Gate

进入阶段 1 前必须确认：

- 本地 `ultralytics` 能导入
- baseline 配置能构建模型
- 数据集 yaml 路径已指向真实数据
- 至少完成一次 YOLOv8n-OBB 或 RV1106 baseline smoke train/val/export
