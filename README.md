# Intel YOLO webcam demo

Run real-time YOLO11n object detection from a laptop webcam or a USB camera on a workstation. The preview displays labeled boxes, processed FPS, inference time, the selected device/model, and skipped frames. It runs locally and uses no webcam recordings by default.

## Set up once

### Windows launcher

Install [Python 3.12 for Windows x86-64](https://www.python.org/downloads/windows/), extract the ZIP, then double-click `run_windows.bat`. The first run uses the internet to install dependencies and prepare the models. Later runs reuse them and open the default webcam. Press **1** or **2** in the preview to switch modes.

For a USB camera or a specific inference device, run the launcher from PowerShell in the extracted folder.

```powershell
.\run_windows.bat --camera 1
.\run_windows.bat --camera 0 --device NPU --precision int8
.\run_windows.bat --camera 0 --device CPU --precision int8 --profile
```

### Manual setup for Windows or Linux

Extract the ZIP, open a terminal in `intel-yolo-webcam`, and use 64-bit Python 3.12. On Windows, use PowerShell. On Linux, run within a desktop session for the preview, or add `--headless`.

| Step | Windows PowerShell | Linux |
| --- | --- | --- |
| Create a virtual environment | `py -3.12 -m venv .venv` | `python3.12 -m venv .venv` |
| Activate it | `.\.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` in each command below.

```text
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-export.txt
python export_model.py --int8
python demo.py --list-devices
```

Export downloads the official COCO YOLO11n weights and the COCO128 calibration set. It writes three static models under `models/` with 416×416 input. The FP32 model is the CPU reference. FP16 refers to a weight-compressed IR, and INT8 refers to a calibrated graph that can still contain floating-point operations. The OpenVINO plugin selects kernel precision by operation and hardware. COCO128 is used for quick demo calibration. To calibrate with representative images, pass `--data your-dataset.yaml`. A machine that already has the exported `models/` directory only needs `python -m pip install -r requirements.txt`.

OpenVINO lists accelerators it can see through installed drivers. On a Core Ultra laptop with a missing device entry, check the Intel driver setup and rerun `--list-devices`. [Intel NPU setup](https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-npu.html) · [Intel GPU setup](https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-gpu.html)

## Run the webcam

```text
python demo.py --camera 0
```

Camera `0` is usually the built-in webcam on a laptop or the first USB camera on a workstation. Try `--camera 1` if the wrong camera opens. If a USB camera supports MJPEG at 60 FPS, request that mode with `--mjpg --camera-fps 60`. The program prints the mode reported by the camera backend. If the camera opens but supplies no frames, try `--camera-backend dshow` on Windows or `--camera-backend v4l2` on Linux.

With the preview window focused, use these keys.

| Key | Mode |
| --- | --- |
| **1** | CPU FP32 baseline. Runs the same YOLO11n model with FP32 inference and no GPU or NPU offload. OpenVINO still applies its standard CPU optimizations, which can include vectorized FP32 kernels. |
| **2** | Optimized path. Uses the device and model selected during startup. Pass `--device NPU --precision int8` or `--device CPU --precision int8` to choose what this key compares against the baseline. |
| **Q** or **Esc** | Exit. |

Both paths are warmed up before the preview opens. The demo starts in Optimized mode when available. Switching keeps the same camera feed and model input size and starts a fresh FPS measurement. The preview shows which paths are available. Pressing an unavailable mode key leaves the current mode active. The controls require a preview window.

| Run | Command |
| --- | --- |
| Core Ultra NPU | `python demo.py --camera 0 --device NPU --precision int8` |
| Core Ultra Intel GPU | `python demo.py --camera 0 --device GPU --precision fp16` |
| Xeon INT8 and AMX kernel check | `python demo.py --camera 0 --device CPU --precision int8 --profile` |
| CPU FP32 reference | `python demo.py --camera 0 --device CPU --precision fp32` |
| USB camera with no desktop preview | `python demo.py --camera 1 --headless --frames 120` |
| Local video playback at processing speed | `python demo.py --video clip.mp4` |

For a smaller input, run `python export_model.py --size 320 --int8`, then add `--size 320` to the demo and benchmark commands. Compare at the same input size when measuring device speedups. Smaller inputs can make distant objects harder to detect.

Without a device override, the preview probes the accelerator and reduced-precision CPU candidates, picks the combination with the lowest measured single-frame inference latency for key **2**, and prints the execution device. Startup can take several seconds or longer while the accelerator compiler runs. A failed compile is reported and the remaining combinations are tried. On a Xeon, AMX is a CPU instruction set. OpenVINO selects eligible kernels for the calibrated graph, and `--profile` shows whether executed kernel names include AMX.

While the CPU prepares frames, decodes detections and draws the preview, inference runs on the selected device. Webcam capture runs on a separate thread that keeps the newest frame. If detection falls behind, the preview counts skipped frames and stays current. The frame age shown is measured from when OpenCV returned the frame.

## Measure FPS and verify AMX

```text
python demo.py --camera 0 --benchmark --json webcam-results.json
python demo.py --video clip.mp4 --benchmark --frames 240 --json video-results.json
```

Each run captures or reads up to 16 frames and replays the same frames for each device and model variant. A fixed video clip makes comparisons repeatable across machines or runs.

| Metric | What it measures |
| --- | --- |
| Live FPS | Completed preview frames per second, limited by incoming camera FPS and processing time |
| Infer p50 / p95 | Single-request inference latency on preprocessed frames |
| Detector FPS | Resize and letterbox, inference, suppression of overlapping detections, and drawing, with capture and display excluded |
| Async infer FPS | Saturated inference throughput on preprocessed frames, using asynchronous requests |
| vs CPU FP32 | Detector FPS divided by the OpenVINO CPU FP32 reference from the same run |

To isolate the AMX contribution, compare **the same INT8 model, video, and request count**. On Linux, run the first command with your normal environment, then run the second with oneDNN's ISA ceiling.

```bash
python demo.py --video clip.mp4 --device CPU --precision int8 --benchmark --frames 240 --requests 4 --json int8-default.json
ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI python demo.py --video clip.mp4 --device CPU --precision int8 --benchmark --frames 240 --requests 4 --json int8-vnni.json
```

Run the same pair with `--profile` separately to check the kernel names. The oneDNN ceiling depends on build support and only governs affected dispatch. Attribute the speedup to AMX only after verifying relevant executed kernels change and that other factors stay comparable. The CPU FP32 comparison measures the combined effect of quantization and dispatch. [OpenVINO CPU plugin](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.html) · [oneDNN ISA controls](https://uxlfoundation.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html)
