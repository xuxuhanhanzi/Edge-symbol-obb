# Stage 0 Environment Check

Date: 2026-05-20

## Python Environment

```text
Python executable: E:\anaconda3\python.exe
Python version: 3.12.7
torch: 2.6.0+cu118
torchvision: 0.21.0
timm: 0.9.16
local ultralytics version: 8.2.82
```

## Dependency Fixes Applied

The first smoke check failed because the active Python environment did not provide `torchvision` package metadata. After installing `torchvision==0.21.0 --no-deps`, import advanced to the next missing dependency: `timm`.

The current local code imports `timm.layers.weight_init` in `ultralytics/nn/modules/block.py`, so `timm>=0.9.16` is now recorded in both:

- `requirements.txt`
- `pyproject.toml`

## Smoke Check

Command:

```text
python -B scripts\train\smoke_obb.py
```

Result:

```text
import_ok
ultralytics_version=8.2.82
official_arch_reference: task=obb, model_build_ok
rv1106_m2: task=obb, model_build_ok
```

## Remaining Environment Notes

`pip` reports an invalid distribution warning for `~orch` under `E:\anaconda3\Lib\site-packages`. This did not block the smoke check, but the environment should be cleaned before long training runs if package operations become unstable.
