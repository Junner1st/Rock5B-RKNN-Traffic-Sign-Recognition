# Rock5B RKNN Traffic Sign Recognition

Video recognition experiment for running an RKNN traffic sign detection model on Rock5B / RK3588 with RKNNLite.

## Requirements

System packages:

```bash
sudo apt update
sudo apt install python3-opencv python3-numpy ffmpeg
```

On Ubuntu Rockchip noble, the old Debian bookworm package names `rknpu2-rk3588` and `python3-rknnlite2` may not exist. The Rockchip kernel/PPA can already provide the NPU driver, while the Python runtime can be installed in this submodule uv environment with the `runtime` group:

```bash
source /home/stoner/.local/bin/env
uv sync --group runtime
```

The runtime group installs `rknn-toolkit-lite2`, whose Python import is `rknnlite.api`.

This project expects an RK3588 RKNN model at:

```text
best_rknn_model/best-rk3588.rknn
```

It's provided on Releases.

The class names are loaded from:

```text
best_rknn_model/metadata.yaml
```

## Configuration

Recognition settings live in [src/config.py](src/config.py):

```python
RECOGNITION_VIDEO_PATH = data/videos/sample-day-1.mp4
RECOGNITION_MODEL_PATH = best_rknn_model/best-rk3588.rknn
RECOGNITION_OUTPUT_PATH = data/recognized/sample_recognized_tracked.mp4
RECOGNITION_CONFIDENCE = 0.25
RECOGNITION_IOU = 0.5
RECOGNITION_INPUT_SIZE = 736
```

`RECOGNITION_MAX_FRAMES = None` means the full video is processed. Set it to an integer only when you want a short benchmark or quick smoke test.

## Run Video Recognition

Run with the default paths from `src/config.py`:

```bash
python3 src/recognition.py
```

The output video is written to:

```text
data/recognized/sample_recognized_tracked.mp4
```

Run without saving video, useful for speed testing:

```bash
python3 src/recognition.py --no-save-video
```

Run a short smoke test:

```bash
python3 src/recognition.py --max-frames 20 --no-save-video
```

`--max-frames 20` means "only process the first 20 frames." It is not required for normal recognition.

## Common Options

```bash
python3 src/recognition.py \
  --video data/videos/sample-day-1.mp4 \
  --model best_rknn_model/best-rk3588.rknn \
  --output data/recognized/result.mp4 \
  --conf 0.25 \
  --iou 0.5 \
  --input-size 736
```

Useful flags:

| Flag | Description |
| --- | --- |
| `--no-save-video` | Run recognition and print speed statistics without writing an output video. |
| `--max-frames N` | Process only the first `N` frames. |
| `--log-interval 0` | Disable periodic progress logs. |
| `--conf VALUE` | Detection confidence threshold. |
| `--iou VALUE` | NMS IOU threshold. |

## Output Statistics

At the end, the program prints:

- frames processed
- preprocessing time
- RKNN inference time
- postprocess/NMS time
- average time per frame
- average recognition per second
- total detections counted

Example:

```text
=== Recognition Summary ===
Frames processed: 3601
Preprocess time: 32.574s
RKNN inference time: 233.433s
Postprocess/NMS time: 14.440s
Detection pipeline time: 280.457s
Average per frame: 0.0779s
Average recognition per second: 12.84
Average rolling 1s recognition per second: 11.81
Detections counted: 1180
```

## RKNN Output Test

To verify that the RKNN model can load and inspect its output tensor shapes:

```bash
python3 src/image-recognition-test.py
```

This test expects an image named `test.jpg` in the project root.

## Notes

The RKNN runtime may print a warning like:

```text
Query dynamic range failed ... static shape RKNN model
```

For this static-shape RKNN model, that warning is expected and does not prevent inference.

## yolo-ver-comp Bridge

This repository owns the RKNN-specific pieces used by `yolo-ver-comp`: exporting `.pt` weights to RKNN, running RKNNLite on image datasets, running RKNNLite on videos, and writing RKNN reports.

First-time `.pt` export requires Ultralytics plus RKNN Toolkit 2 in the Python environment that runs this bridge. RKNN runtime testing requires RKNNLite on the target machine.

RKNNLite also needs Rockchip's runtime shared library:

```bash
git clone --depth 1 https://github.com/airockchip/rknn-toolkit2.git /tmp/rknn-toolkit2
sudo cp /tmp/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
sudo ldconfig
```

When exporting on arm64, `rknn-toolkit2` may build `onnxoptimizer` from source. Install build tools first:

```bash
sudo apt update
sudo apt install cmake build-essential
```

The export dependency group also includes `setuptools>=70,<81` for the legacy `pkg_resources` module and `onnxslim` so Ultralytics does not try to auto-install missing export helpers at runtime.

Dataset test from a trained yolo-ver-comp run:

```bash
python3 src/yolo_ver_comp_bridge.py test \
  --weights ../yolo-ver-comp/runs/run_yolo11n_1/train/weights/best.pt \
  --dataset-root ../yolo-ver-comp/data \
  --data-yaml ../yolo-ver-comp/data/data.yaml \
  --split test \
  --target rk3588 \
  --input-size 640 \
  --conf 0.25 \
  --iou 0.45 \
  --output-dir ../yolo-ver-comp/runs/run_yolo11n_1/test_rknn \
  --report-yaml ../yolo-ver-comp/runs/run_yolo11n_1/reports/test_report_rknn.yaml \
  --report-md ../yolo-ver-comp/runs/run_yolo11n_1/reports/test_report_rknn.md
```

Video test:

```bash
python3 src/yolo_ver_comp_bridge.py test \
  --weights ../yolo-ver-comp/runs/run_yolo11n_1/train/weights/best.pt \
  --video ../yolo-ver-comp/data/videos/sample-day-1.mp4 \
  --split test \
  --target rk3588 \
  --input-size 640 \
  --conf 0.25 \
  --iou 0.45 \
  --output-dir ../yolo-ver-comp/runs/run_yolo11n_1/test_rknn \
  --report-yaml ../yolo-ver-comp/runs/run_yolo11n_1/reports/test_report_rknn.yaml \
  --report-md ../yolo-ver-comp/runs/run_yolo11n_1/reports/test_report_rknn.md
```

If `--weights` points to a `.pt` file, the bridge exports and reuses:

```text
<weights-dir>/<weights-stem>_rknn_model/<weights-stem>-rk3588.rknn
```

### Submodule uv Environment

This submodule is its own uv project. Do not add the RKNN dependencies to `yolo-ver-comp`; keep them isolated here.

For export plus runtime in the submodule environment:

```bash
source /home/stoner/.local/bin/env
uv sync --directory external/Rock5B-RKNN-Traffic-Sign-Recognition --group export --group runtime
```

If the `.rknn` model already exists, yolo-ver-comp only includes the `runtime` group by default. It includes `export` automatically only when the expected RKNN export is missing.

For a system Python runtime on Rock5B, such as Debian packages that provide `rknnlite`, call yolo-ver-comp with:

```bash
ROCK5B_RKNN_PYTHON=python3 uv run python src/test.py --run-dir run_yolo11n_1 --adapter rknn
```

To choose which submodule uv groups yolo-ver-comp includes, set a comma-separated list:

```bash
ROCK5B_RKNN_UV_GROUPS=export,runtime uv run python src/test.py --run-dir run_yolo11n_1 --adapter rknn
```

Dependency groups:

| Group | Purpose |
| --- | --- |
| `export` | Desktop/server RKNN export dependencies. |
| `runtime` | Rock5B RKNNLite runtime dependencies through `rknn-toolkit-lite2`. |

Some runtime packages may still be better installed from Rockchip/Debian packages on older Debian images, but Ubuntu Rockchip noble does not appear to publish the old `python3-rknnlite2` package name.
