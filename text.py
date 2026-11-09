"""Evaluate the baseline model and share evaluation code with the ablation entries."""

from copy import deepcopy
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "lv_yolo11.yaml"
PROJECT_PATH = ROOT / "runs" / "detect"
VAL_CONFIG = {
    "split": "test",
    "imgsz": 576,
    "iou": 0.445,
    "augment": True,
    "batch": 16,
    "quantize": 16,
    "rect": True,
    "agnostic_nms": False,
    "max_det": 300,
}


def validate(model_path, result_name, plots=True):
    """Run validation with the trained model and return Ultralytics metrics."""
    for path in (model_path, DATA_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    model = YOLO(model_path)
    _, _, _, gflops = model.info(imgsz=VAL_CONFIG["imgsz"])
    if not gflops:
        import torch
        import thop

        network = deepcopy(model.model).eval()
        parameter = next(network.parameters())
        image = torch.zeros(
            (1, parameter.shape[1], VAL_CONFIG["imgsz"], VAL_CONFIG["imgsz"]),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        with torch.no_grad():
            gflops = thop.profile(network, inputs=[image], verbose=False)[0] / 1e9 * 2
        del network, image
    metrics = model.val(
        data=DATA_PATH,
        **VAL_CONFIG,
        device=0,
        workers=0,
        project=PROJECT_PATH,
        name=result_name,
        exist_ok=True,
        plots=plots,
        verbose=False,
    )
    return metrics, float(gflops)


def print_metrics(metrics, gflops):
    """Print overall and per-class detection metrics."""
    print(f"GFLOPs (single forward @ {VAL_CONFIG['imgsz']}x{VAL_CONFIG['imgsz']}, excluding TTA): {gflops:.6f}")
    print(f"\nmAP50: {metrics.box.map50:.6f}")
    print(f"mAP75: {metrics.box.map75:.6f}")
    print(f"mAP50-95: {metrics.box.map:.6f}")
    print("\nPer-class metrics:")
    for index, class_id in enumerate(metrics.box.ap_class_index):
        name = metrics.names[int(class_id)]
        precision, recall, map50, map50_95 = metrics.box.class_result(index)
        map75 = metrics.box.all_ap[index, 5]
        print(
            f"{name}: P={precision:.6f}, R={recall:.6f}, "
            f"mAP50={map50:.6f}, mAP75={map75:.6f}, mAP50-95={map50_95:.6f}"
        )
    print(f"\nResults: {metrics.save_dir}")


def main(model_name="best.pt", result_name="baseline_result", plots=True):
    """Run validation and print its metrics."""
    model_path = ROOT / model_name
    print(f"Model: {model_path}")
    print(f"Configuration: {VAL_CONFIG}")
    metrics, gflops = validate(model_path, result_name, plots)
    print_metrics(metrics, gflops)


if __name__ == "__main__":
    main()
