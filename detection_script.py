import argparse
from pathlib import Path
import json
import sys
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def run_inference(weights, source, conf, out_dir, logs_file, imgsz=640):
    model = YOLO(weights)
    p = Path(source)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_file = Path(logs_file)
    logs_file.parent.mkdir(parents=True, exist_ok=True)

    if p.is_dir():
        imgs = sorted([*p.glob('*.jpg'), *p.glob('*.jpeg'), *p.glob('*.png')])
    elif p.is_file():
        imgs = [p]
    else:
        raise SystemExit(f"Source not found: {source}")

    results_log = []
    for img_path in imgs:
        print(f"Processing {img_path} ...")
        results = model(str(img_path), conf=conf, imgsz=imgsz)
        r = results[0]
        img = r.orig_img.copy()
        annotator = Annotator(img)

        detections = []
        for box in r.boxes:
            xyxy = [float(x) for x in box.xyxy[0].tolist()]
            conf_score = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id] if model.names else str(cls_id)
            annotator.box_label(xyxy, f"{label} {conf_score:.2f}")
            detections.append({
                "label": label,
                "confidence": conf_score,
                "bbox": xyxy
            })

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), annotator.result())

        results_log.append({
            "image": img_path.name,
            "num_detections": len(detections),
            "detections": detections
        })

        print(f"Saved annotated image to: {out_path}")

    with open(logs_file, "w", encoding="utf8") as f:
        json.dump({"results": results_log}, f, indent=2)
    print(f"Wrote logs to: {logs_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--source", required=True, help="Folder of images or a single image file")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--out", default="submission/Part_1_Glove_Detection/output", help="Output folder for annotated images")
    parser.add_argument("--logs", default="submission/Part_1_Glove_Detection/logs/detections.json", help="JSON log file path")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    args = parser.parse_args()

    run_inference(args.weights, args.source, args.conf, args.out, args.logs, imgsz=args.imgsz)
