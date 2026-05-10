# Rock5B RKNN Traffic Sign Recognition

Video recognition experiment for running an RKNN traffic sign detection model on Rock5B / RK3588 with RKNNLite.

## Requirements

System packages:

```bash
sudo apt update
sudo apt install rknpu2-rk3588 python3-rknnlite2 python3-opencv python3-numpy ffmpeg
```

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
