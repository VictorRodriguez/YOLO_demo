"""Live YOLO11n detection and replay benchmarks with Intel OpenVINO."""
import argparse
import ctypes
from dataclasses import dataclass
import json
from pathlib import Path
import platform
import re
from threading import Condition, Thread
from time import perf_counter

import cv2
import numpy as np
import openvino as ov

from coco import CLASSES

PRECISIONS = ("fp32", "fp16", "int8")


def property_or(obj, name, default):
    try:
        return obj.get_property(name)
    except RuntimeError:
        return default


def amx_status(capabilities):
    if any("AMX" in str(item).upper() for item in capabilities):
        return "reported by OpenVINO"
    if platform.system() == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text()
        except OSError:
            return "unknown"
        flags = re.findall(r"^flags\s*:\s*(.*)$", text, re.MULTILINE)
        if flags:
            return ("AMX-TILE and AMX-INT8 exposed by Linux"
                    if all({"amx_tile", "amx_int8"} <= set(row.split()) for row in flags)
                    else "AMX-INT8 not exposed on all visible CPUs")
    if platform.system() == "Windows" and platform.machine().lower() in {"amd64", "x86_64"}:
        try:
            query = ctypes.WinDLL("kernel32").GetEnabledXStateFeatures
            query.argtypes = []
            query.restype = ctypes.c_uint64
            tile_state = (1 << 17) | (1 << 18)
            if query() & tile_state == tile_state:
                return "AMX tile state exposed by Windows"
        except (AttributeError, OSError):
            pass
    return "unknown; use --profile on CPU INT8 to check kernel names"


def show_devices(core):
    print(f"OpenVINO {ov.__version__} | {platform.system()} {platform.machine()}")
    for device in core.available_devices:
        try:
            name = core.get_property(device, "FULL_DEVICE_NAME")
            capabilities = core.get_property(device, "OPTIMIZATION_CAPABILITIES")
        except RuntimeError as error:
            print(f"  {device}: {str(error).strip()}")
            continue
        print(f"  {device}: {name} [{', '.join(map(str, capabilities))}]")
        if device == "CPU":
            print(f"    AMX: {amx_status(capabilities)}")


@dataclass(frozen=True)
class Target:
    device: str
    precision: str

    def __str__(self):
        return f"{self.device}/{self.precision.upper()}"


def candidates(devices, models, device, precision, benchmark):
    if device == "auto":
        selected = [name for name in devices if name.split(".")[0] in {"CPU", "GPU", "NPU"}]
    else:
        selected = [name for name in devices if name == device or name.startswith(device + ".")]
        if not selected:
            raise RuntimeError(f"{device} is unavailable. Run --list-devices to check drivers.")
    targets = []
    for name in selected:
        family = name.split(".")[0]
        order = ("fp16", "int8", "fp32") if family == "GPU" else ("int8", "fp16", "fp32")
        variants = [p for p in order if p in models and precision in {"auto", p}]
        if family == "CPU" and precision == "auto":
            # CPU live prioritizes the calibrated graph. FP32 is the CPU reference.
            if benchmark:
                variants = [p for p in ("fp32", "int8", "fp16") if p in models]
            else:
                variants = [p for p in ("int8", "fp32", "fp16") if p in models]
        targets.extend(Target(name, p) for p in variants)
    return targets


def prepare(frame, size):
    height, width = frame.shape[:2]
    gain = min(size / width, size / height)
    resized_w, resized_h = round(width * gain), round(height * gain)
    resized_w, resized_h = max(1, resized_w), max(1, resized_h)
    left, top = (size - resized_w) // 2, (size - resized_h) // 2
    right, bottom = size - resized_w - left, size - resized_h - top
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor *= np.float32(1 / 255)
    return tensor, (resized_w / width, resized_h / height, left, top)


def infer(request, data):
    request.infer({0: data}, share_inputs=True, share_outputs=True)
    return request.get_output_tensor(0).data


def draw_detections(frame, output, transform, confidence, iou):
    predictions = output[0].T
    scores_by_class = predictions[:, 4:]
    mask = scores_by_class.max(axis=1) >= confidence
    predictions = predictions[mask]
    if not len(predictions):
        return 0
    classes = predictions[:, 4:].argmax(axis=1)
    scores = predictions[np.arange(len(classes)), classes + 4].tolist()
    cx, cy, widths, heights = predictions[:, :4].T
    boxes = np.column_stack((cx - widths / 2, cy - heights / 2, widths, heights))
    keep = cv2.dnn.NMSBoxesBatched(boxes.tolist(), scores, classes.tolist(),
                                   confidence, iou)
    gain_x, gain_y, left, top = transform
    height, width = frame.shape[:2]
    count = 0
    for index in np.asarray(keep).reshape(-1)[:100]:
        x, y, w, h = boxes[index]
        x1 = int(np.clip((x - left) / gain_x, 0, width - 1))
        x2 = int(np.clip((x + w - left) / gain_x, 0, width - 1))
        y1 = int(np.clip((y - top) / gain_y, 0, height - 1))
        y2 = int(np.clip((y + h - top) / gain_y, 0, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        color = (70, 236, 144)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES[classes[index]]} {scores[index]:.0%}"
        cv2.putText(frame, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2, cv2.LINE_AA)
        count += 1
    return count


class Camera:
    """Keep one fresh frame and retrieve it once, even when inference falls behind."""

    def __init__(self, args):
        backend = {"auto": cv2.CAP_ANY, "msmf": cv2.CAP_MSMF,
                   "dshow": cv2.CAP_DSHOW, "v4l2": cv2.CAP_V4L2}[args.camera_backend]
        self.capture = cv2.VideoCapture(args.camera, backend)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open camera {args.camera}. Check permissions and camera index.")
        if args.mjpg:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.camera_fps:
            self.capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.sequence = 0
        self.delivered = 0
        self.latest = None
        self.done = False
        self.ready = Condition()
        print(f"Camera {args.camera} | {self.capture.getBackendName()} | "
              f"{self.capture.get(cv2.CAP_PROP_FRAME_WIDTH):g}x"
              f"{self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT):g} | "
              f"reported {self.capture.get(cv2.CAP_PROP_FPS):g} FPS", flush=True)
        self.thread = Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        try:
            while not self.done:
                ok, frame = self.capture.read()
                with self.ready:
                    if not ok or self.done:
                        break
                    self.sequence += 1
                    self.latest = (frame, perf_counter(), self.sequence)
                    self.ready.notify_all()
        finally:
            self.capture.release()
            with self.ready:
                self.done = True
                self.ready.notify_all()

    def next(self):
        with self.ready:
            self.ready.wait_for(lambda: self.sequence > self.delivered or self.done, timeout=5)
            if self.sequence <= self.delivered:
                raise RuntimeError("No fresh camera frame. Check the camera or try a different backend.")
            self.delivered = self.sequence
            return self.latest

    def close(self):
        with self.ready:
            self.done = True
            self.ready.notify_all()
        self.thread.join(timeout=1)


def compile_target(core, target, models, size, mode, profile=False):
    config = {"PERFORMANCE_HINT": mode}
    if target.device.split(".")[0] == "CPU" and target.precision == "fp32":
        config["INFERENCE_PRECISION_HINT"] = "f32"
    if profile and target.device == "CPU":
        config[ov.properties.enable_profiling()] = True
    model = core.read_model(models[target.precision])
    if (len(model.inputs) != 1 or model.input(0).partial_shape.is_dynamic
            or list(model.input(0).shape) != [1, 3, size, size]
            or model.input(0).element_type != ov.Type.f32):
        raise ValueError("Expected a static float32 input [1, 3, size, size]. Run export_model.py.")
    if (len(model.outputs) != 1 or model.output(0).partial_shape.is_dynamic
            or list(model.output(0).shape) != [1, 4 + len(CLASSES), 21 * (size // 32) ** 2]):
        raise ValueError("Expected the raw COCO YOLO11n output. Export with NMS disabled.")
    return core.compile_model(model, target.device, config)


def report_profile(request):
    entries = [row for row in request.get_profiling_info() if row.real_time.total_seconds() > 0]
    amx = [row for row in entries if "amx" in row.exec_type.lower()]
    if amx:
        print(f"AMX evidence: {len(amx)} executed nodes with AMX in the OpenVINO kernel name.")
    else:
        print("AMX evidence: the executed-node names did not identify AMX.")
    for row in sorted(amx or entries, key=lambda row: row.real_time, reverse=True)[:5]:
        print(f"    {row.node_name} | {row.exec_type} | {row.real_time.total_seconds() * 1e3:.3f} ms")


def probe_live(core, targets, models, args, sample):
    best = None
    for target in targets:
        try:
            print(f"Probing {target}...", flush=True)
            compiled = compile_target(core, target, models, args.size, "LATENCY", args.profile)
            request = compiled.create_infer_request()
            for _ in range(5):
                infer(request, sample)
            start = perf_counter()
            runs = 0
            while runs < 12 or perf_counter() - start < 0.30:
                infer(request, sample)
                runs += 1
            latency = (perf_counter() - start) / runs
            print(f"  {latency * 1e3:.2f} ms/frame", flush=True)
            if best is None or latency < best[0]:
                best = latency, target, compiled, request
        except (RuntimeError, ValueError) as error:
            print(f"  Skipped {target}: {str(error).strip()}", flush=True)
    if best is None:
        return None
    _, target, compiled, request = best
    print(f"Ready {target} | execution devices: {compiled.get_property('EXECUTION_DEVICES')}")
    if args.profile and target.device == "CPU":
        report_profile(request)
    return target, request


def live_modes(core, targets, models, args, sample):
    if args.headless:
        selected = probe_live(core, targets, models, args, sample)
        return {None: selected} if selected else {}

    modes = {}
    baseline = Target("CPU", "fp32")
    if baseline.device in core.available_devices and baseline.precision in models:
        reference = probe_live(core, [baseline], models, args, sample)
        if reference:
            modes[ord("1")] = reference
    optimized = probe_live(
        core, [target for target in targets if target != baseline], models, args, sample
    )
    if optimized:
        modes[ord("2")] = optimized
    return modes


def run_live(core, targets, models, args, next_frame):
    first = next_frame()
    if first is None:
        raise RuntimeError("No frames in the video.")
    sample, _ = prepare(first[0], args.size)
    modes = live_modes(core, targets, models, args, sample)
    if not modes:
        raise RuntimeError("No requested device/model combination could run.")
    active_key = ord("2") if ord("2") in modes else next(iter(modes))
    mode_names = {ord("1"): "Baseline", ord("2"): "Optimized", None: "Live"}
    target, request = modes[active_key]
    print(f"{mode_names[active_key]} active | {target}", flush=True)
    controls = " | ".join(
        f"{chr(key)} {mode_names[key]}"
        + (f" ({modes[key][0]})" if key in modes else " unavailable")
        for key in (ord("1"), ord("2"))
    ) + " | Q quit"
    item = next_frame() if args.video is None else first
    start = window_start = perf_counter()
    count = window_count = dropped = 0
    last_sequence = item[2] - 1 if item is not None else 0
    fps = None
    while item is not None:
        frame, timestamp, sequence = item
        dropped += sequence - last_sequence - 1
        last_sequence = sequence
        tensor, transform = prepare(frame, args.size)
        before = perf_counter()
        output = infer(request, tensor)
        inference_ms = (perf_counter() - before) * 1e3
        found = draw_detections(frame, output, transform, args.conf, args.iou)
        count += 1
        window_count += 1
        now = perf_counter()
        if now - window_start >= 1:
            fps = window_count / (now - window_start)
            if args.headless:
                print(f"{target} | {fps:.1f} FPS | {inference_ms:.1f} ms inference | "
                      f"{found} detections | {dropped} skipped", flush=True)
            window_start, window_count = now, 0
        if not args.headless:
            rate = f"{fps:.1f} FPS" if fps is not None else "Measuring FPS"
            label = f"{mode_names[active_key]} | {target} | {rate} | {inference_ms:.1f} ms infer"
            detail = (f"Frame age {(now - timestamp) * 1e3:.0f} ms | "
                      f"{dropped} skipped | {found} objects")
            for y, line in ((28, label), (52, detail), (76, controls)):
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (12, 12, 12), 4, cv2.LINE_AA)
                cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (245, 245, 245), 1, cv2.LINE_AA)
            cv2.imshow("Intel YOLO Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty("Intel YOLO Webcam", cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in modes and key != active_key:
                active_key = key
                target, request = modes[active_key]
                window_start, window_count, fps = perf_counter(), 0, None
                print(f"{mode_names[active_key]} active | {target}", flush=True)
        if args.frames and count >= args.frames:
            break
        item = next_frame()
    elapsed = perf_counter() - start
    print(f"Session total | {count} processed in {elapsed:.2f} s | "
          f"{count / max(elapsed, 1e-9):.2f} FPS | {dropped} skipped")


def run_benchmark(core, targets, models, args, next_frame):
    frames = []
    for _ in range(16):
        item = next_frame()
        if item is None:
            break
        frames.append(item[0])
    if not frames:
        raise RuntimeError("No frames to benchmark.")
    prepared = [prepare(frame, args.size)[0] for frame in frames]
    print(f"Replaying {len(frames)} frames on each target. Capture and GUI are outside the timing.")
    rows = []
    for target in targets:
        try:
            print(f"Benchmarking {target}...", flush=True)
            row = benchmark_target(core, target, models, args, frames, prepared)
            rows.append(row)
        except (RuntimeError, ValueError) as error:
            print(f"  Skipped {target}: {str(error).strip()}", flush=True)
    if not rows:
        raise RuntimeError("No requested device/model combination could run.")
    baseline = next((row for row in rows if row["target"] == "CPU/FP32"), None)
    for row in rows:
        row["detector_speedup_vs_cpu_fp32"] = (
            row["detector_fps"] / baseline["detector_fps"] if baseline else None
        )
    print()
    print(f"{'Target':<14} {'Infer p50 ms':>13} {'Infer p95 ms':>13} "
          f"{'Detector FPS':>12} {'Async infer FPS':>15} {'Jobs':>5} {'vs CPU FP32':>11}")
    for row in rows:
        speedup = row["detector_speedup_vs_cpu_fp32"]
        ratio = f"{speedup:.2f}x" if speedup is not None else "-"
        print(f"{row['target']:<14} {row['p50_ms']:13.2f} {row['p95_ms']:13.2f} "
              f"{row['detector_fps']:12.2f} {row['async_fps']:15.2f} {row['jobs']:5d} {ratio:>11}")
    if args.json:
        report = {"openvino": ov.__version__, "size": args.size, "frames_per_target": args.frames,
                  "replay_frames": len(frames), "rows": rows}
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def benchmark_target(core, target, models, args, frames, prepared):
    latency_model = compile_target(core, target, models, args.size, "LATENCY", args.profile)
    request = latency_model.create_infer_request()
    for sample in prepared[:5]:
        infer(request, sample)
    if args.profile and target.device == "CPU":
        report_profile(request)
    times = []
    for index in range(args.frames):
        sample = prepared[index % len(prepared)]
        start = perf_counter()
        infer(request, sample)
        times.append((perf_counter() - start) * 1e3)

    start = perf_counter()
    detection_count = 0
    for index in range(args.frames):
        frame = frames[index % len(frames)].copy()
        sample, transform = prepare(frame, args.size)
        output = infer(request, sample)
        detection_count += draw_detections(frame, output, transform, args.conf, args.iou)
    detector_fps = args.frames / (perf_counter() - start)
    del request, latency_model

    throughput_model = compile_target(core, target, models, args.size, "THROUGHPUT")
    optimal = int(property_or(throughput_model, "OPTIMAL_NUMBER_OF_INFER_REQUESTS", 1))
    jobs = args.requests or max(1, min(8, optimal))
    queue = ov.AsyncInferQueue(throughput_model, jobs)
    for index in range(jobs * 2):
        queue.start_async({0: prepared[index % len(prepared)]}, share_inputs=True)
    queue.wait_all()
    start = perf_counter()
    for index in range(args.frames):
        queue.start_async({0: prepared[index % len(prepared)]}, share_inputs=True)
    queue.wait_all()
    async_fps = args.frames / (perf_counter() - start)
    return {"target": str(target), "execution_devices": list(
                throughput_model.get_property("EXECUTION_DEVICES")),
            "p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "detector_fps": detector_fps, "async_fps": async_fps, "jobs": jobs,
            "detections": detection_count}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0, help="Camera index, usually 0 or 1")
    source.add_argument("--video", type=Path, help="Optional local video instead of a camera")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-devices", action="store_true")
    mode.add_argument("--benchmark", action="store_true", help="Replay 16 captured frames")
    parser.add_argument("--device", default="auto", help="auto, CPU, NPU, GPU, GPU.0, etc.")
    parser.add_argument("--precision", choices=("auto", *PRECISIONS), default="auto")
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--size", type=int, default=416, help="Model input size")
    parser.add_argument("--camera-backend", choices=("auto", "msmf", "dshow", "v4l2"), default="auto")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--camera-fps", type=int, default=30, help="Requested FPS, 0 keeps camera default")
    parser.add_argument("--mjpg", action="store_true", help="Request USB camera MJPEG mode")
    parser.add_argument("--headless", action="store_true", help="Skip the preview window")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--frames", type=int, help="Live frame limit or benchmark iterations (default 120)")
    parser.add_argument("--requests", type=int, help="Async jobs in benchmark (default plugin hint, capped at 8)")
    parser.add_argument("--profile", action="store_true", help="Print executed CPU kernel names and AMX evidence")
    parser.add_argument("--json", type=Path, help="Optional benchmark JSON output")
    args = parser.parse_args()
    args.device = "auto" if args.device.lower() == "auto" else args.device.upper()
    if args.size < 32 or args.size % 32:
        parser.error("--size must be a positive multiple of 32")
    if not 0 < args.conf < 1 or not 0 < args.iou <= 1:
        parser.error("Use 0 < --conf < 1 and 0 < --iou <= 1")
    if args.camera < 0 or min(args.width, args.height) < 1 or args.camera_fps < 0:
        parser.error("Camera index and FPS must be nonnegative, dimensions positive")
    if args.frames is not None and args.frames < 1 or args.requests is not None and args.requests < 1:
        parser.error("--frames and --requests must be positive")
    if not args.benchmark and (args.requests is not None or args.json is not None):
        parser.error("--requests and --json require --benchmark")
    if args.benchmark and args.frames is None:
        args.frames = 120
    return args


def main():
    args = parse_args()
    cv2.setNumThreads(1)
    core = ov.Core()
    show_devices(core)
    if args.list_devices:
        return
    models = {precision: path for precision in PRECISIONS
              if (path := args.models / f"yolo11n-{args.size}-{precision}.xml").is_file()
              and path.with_suffix(".bin").is_file()}
    targets = candidates(core.available_devices, models, args.device, args.precision, args.benchmark)
    if not targets:
        raise RuntimeError(f"No matching models in {args.models}. Run export_model.py --size {args.size}.")
    camera = capture = None
    try:
        if args.video is None:
            camera = Camera(args)
            next_frame = camera.next
        else:
            capture = cv2.VideoCapture(str(args.video))
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video {args.video}.")
            sequence = 0

            def next_frame():
                nonlocal sequence
                ok, frame = capture.read()
                sequence += 1
                return (frame, perf_counter(), sequence) if ok else None

        runner = run_benchmark if args.benchmark else run_live
        runner(core, targets, models, args, next_frame)
    finally:
        if camera is not None:
            camera.close()
        if capture is not None:
            capture.release()
        if not args.headless and not args.benchmark:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        raise SystemExit(f"Error: {error}") from None
