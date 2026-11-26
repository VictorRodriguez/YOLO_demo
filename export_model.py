"""Export the same COCO YOLO11n checkpoint to static OpenVINO models."""
import argparse
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import openvino as ov
from ultralytics import YOLO

from coco import CLASSES

WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"


def publish(model, output, fp16=False):
    staged = output.parent / ("." + output.stem + ".xml")
    ov.save_model(model, staged, compress_to_fp16=fp16)
    staged.with_suffix(".bin").replace(output.with_suffix(".bin"))
    staged.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=416, help="Square input, divisible by 32")
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--weights", type=Path, help="Local official yolo11n.pt checkpoint")
    parser.add_argument("--int8", action="store_true", help="Also export calibrated INT8")
    parser.add_argument("--data", default="coco128.yaml", help="Ultralytics calibration dataset YAML")
    args = parser.parse_args()
    if args.size < 32 or args.size % 32:
        parser.error("--size must be a positive multiple of 32")
    args.output.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="yolo11-export-") as temp:
        checkpoint = Path(temp) / "yolo11n.pt"
        if args.weights:
            shutil.copyfile(args.weights, checkpoint)
        else:
            print("Downloading official YOLO11n weights", flush=True)
            urlretrieve(WEIGHTS_URL, checkpoint)
        settings = dict(format="openvino", imgsz=args.size, batch=1, dynamic=False,
                        nms=False, device="cpu")
        model = YOLO(str(checkpoint), task="detect")
        if tuple(model.names.values()) != CLASSES:
            parser.error("Use the official COCO YOLO11n weights for this demo.")
        core = ov.Core()
        exported = Path(model.export(**settings)) / "yolo11n.xml"
        fp32 = core.read_model(exported)
        for precision in ("fp32", "fp16"):
            output = args.output / f"yolo11n-{args.size}-{precision}.xml"
            publish(fp32, output, fp16=precision == "fp16")
            print(f"Ready: {output}", flush=True)
        if args.int8:
            exported = Path(model.export(**settings, quantize=8, data=args.data)) / "yolo11n.xml"
            output = args.output / f"yolo11n-{args.size}-int8.xml"
            publish(core.read_model(exported), output)
            print(f"Ready: {output}", flush=True)


if __name__ == "__main__":
    main()
