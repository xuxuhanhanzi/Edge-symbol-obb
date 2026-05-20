# License Notice

本项目基于 Ultralytics YOLO 代码进行修改，继承上游 AGPL-3.0 license。仓库根目录的 `LICENSE` 文件为主要许可证文本。

## Upstream

- Upstream project: Ultralytics YOLO
- Upstream website: `https://docs.ultralytics.com`
- Upstream source: `https://github.com/ultralytics/ultralytics`
- Local package name: `ultralytics`
- Local package version recorded in this snapshot: `8.2.82`

## Local Project Scope

本仓库的本地修改主要服务于：

- 工业二维码/条形码 OBB 检测
- 灰度单通道输入
- 轻量化 OBB 模型结构
- RV1106/RKNN 导出与 INT8 部署实验
- 角度损失和量化稳定性实验

## Data and Weights

数据集、训练权重、ONNX、RKNN、日志和可视化结果不自动继承代码许可证。使用、发布或写论文前需要分别确认：

- 数据集来源和授权
- 标注文件授权
- 预训练权重来源
- 生成模型或第三方工具的输出授权
- 工业现场图片是否可公开

## Publication Checklist

论文或开源发布前至少需要完成：

- 保留 AGPL-3.0 license
- 明确说明基于 Ultralytics 修改
- 标注本地修改范围
- 确认数据集和权重可发布
- 将不可公开数据替换为可公开示例或只发布统计结果
