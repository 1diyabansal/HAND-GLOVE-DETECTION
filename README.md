1. What this project does

This project detects whether a hand is gloved or bare in images.

The model outputs:

gloved_hand

bare_hand

It also generates:

Annotated images (bounding boxes + labels)

A JSON log containing detection results for every image

2. Dataset Used

A hand-detection dataset stored locally.

Training images:
E:\hand detection\images\train

Validation images:
E:\hand detection\images\val

Total images processed: 713

The dataset contains both gloved and bare hands.

3. Model Used

YOLOv8 (Ultralytics)

Base training model: yolov8n.pt

Final trained model: models/best.pt

Classes

gloved_hand

bare_hand

4. Training (Simple Explanation)

Model was trained for 50 epochs using:

yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640


Image size: 640

After training, best.pt was used for inference.

5. What Each Script Does
detection_script.py

Runs detection on a folder of images.
Outputs:

Annotated images → /output

Detection logs → /logs/detections.json

write_detections.py

Processes the entire dataset and creates the final complete JSON log used in submission.

test_infer.py

Tests the model on a single image.

 append_one.py

Used only once to add a missing JSON entry.

6. How to Run the Project
Step 1 — Activate virtual environment
& "E:\hand detection\venv\Scripts\Activate.ps1"

Step 2 — Install required packages
pip install ultralytics opencv-python

Step 3 — Run full detection
python write_detections.py

Step 4 — Run single image test
python test_infer.py

7. What Worked Well

Model correctly detected gloved vs bare hands

JSON logs and annotated outputs were generated successfully

Good accuracy on most images

8. What Can Be Improved

Add more training images (especially different lighting + glove types)

Train longer on GPU for higher accuracy

Implement multiprocessing for faster batch processing

“I also implemented a webcam real-time inference script, which is not required for the submission but demonstrates additional capability