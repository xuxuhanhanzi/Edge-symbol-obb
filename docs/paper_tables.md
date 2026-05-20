# Paper Tables

## Main Results

| Model | Params | FLOPs | mAP50 | mAP50-95 | Angle Error/Loss | FPS-GPU | FPS-RV1106 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n-OBB official | | | | | | | |
| YOLOv8s-OBB official | | | | | | | |
| RV1106 lightweight baseline | | | | | | | |
| + QG-OBB Head | | | | | | | |
| + SOF-FPN | | | | | | | |
| Full Model | | | | | | | |

## Ablation

| QG-OBB | SOF-FPN | GIS-Aug | mAP50 | mAP50-95 | Angle Error/Loss | INT8 Drop |
|---|---|---|---:|---:|---:|---:|
| No | No | No | | | | |
| Yes | No | No | | | | |
| No | Yes | No | | | | |
| Yes | Yes | No | | | | |
| Yes | Yes | Yes | | | | |

## Deployment

| Model | Format | Precision | Size | mAP50 | mAP Drop | FPS | Latency |
|---|---|---|---:|---:|---:|---:|---:|
| RV1106 lightweight baseline | PyTorch | FP32 | | | | | |
| RV1106 lightweight baseline | RKNN | INT8 | | | | | |
| Full Model | PyTorch | FP32 | | | | | |
| Full Model | RKNN | INT8 | | | | | |
