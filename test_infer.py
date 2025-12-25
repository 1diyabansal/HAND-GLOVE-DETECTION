import os, json, cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

weights = r"E:\hand detection\submission\Part_1_Glove_Detection\models\best.pt"
image_path = r"E:\hand detection\images\train\1596291465549_jpg.rf.cba9d895c53bca6182e11b9531f1dd2a.jpg"

out_img = r"E:\hand detection\submission\Part_1_Glove_Detection\output\one_test_result.jpg"
out_json = r"E:\hand detection\submission\Part_1_Glove_Detection\logs\one_test_result.json"

os.makedirs(os.path.dirname(out_img), exist_ok=True)
os.makedirs(os.path.dirname(out_json), exist_ok=True)

print("Loading model:", weights)
model = YOLO(weights)
print("Model names:", model.names)

print("Reading image:", image_path)
img = cv2.imread(image_path)
results = model(img, conf=0.45)

annotator = Annotator(img.copy())
detections = []

for r in results:
    for box in r.boxes:
        xyxy = box.xyxy[0].tolist()
        score = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        annotator.box_label(xyxy, f"{label} {score:.2f}")
        detections.append({
            "label": label,
            "confidence": score,
            "bbox": xyxy
        })

annot_img = annotator.result()
cv2.imwrite(out_img, annot_img)

with open(out_json, "w") as f:
    json.dump({"image": image_path, "detections": detections}, f, indent=4)

print("Saved image to:", out_img)
print("Saved JSON to:", out_json)
print("num_detections =", len(detections))
