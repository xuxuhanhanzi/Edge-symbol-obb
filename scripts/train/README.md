# Training Scripts

训练入口和 smoke check 放在本目录。

当前可用：

- `smoke_obb.py`: 验证本地 `ultralytics` 导入，以及 baseline/RV1106 配置是否能构建模型。
- `train_rv1106_gray_obb.py`: 历史 RV1106 灰度 OBB 训练入口，包含灰度预处理 monkey patch。
