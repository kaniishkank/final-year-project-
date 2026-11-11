import os
import shutil
from ultralytics import YOLO

# 1. Path to checkpoint
checkpoint_path = "eviguard_checkpoints/yolov8_perfect/weights/last.pt"
if not os.path.exists(checkpoint_path):
    checkpoint_path = "runs/detect/eviguard_checkpoints/yolov8_perfect/weights/last.pt"

print(f"[EviGuard Engine] Resuming training from checkpoint: {checkpoint_path}")

# 2. Load model and resume
model = YOLO(checkpoint_path)
results = model.train(resume=True)

# 3. Copy final best weights to root and backend/weights
candidates = [
    "eviguard_checkpoints/yolov8_perfect/weights/best.pt",
    "runs/detect/eviguard_checkpoints/yolov8_perfect/weights/best.pt"
]

copied = False
for cp in candidates:
    if os.path.exists(cp):
        shutil.copy(cp, "best.pt")
        os.makedirs("backend/weights", exist_ok=True)
        shutil.copy(cp, "backend/weights/best.pt")
        print(f"\n>>> Deep-trained weights '{cp}' successfully deployed to root and backend/weights/! <<<")
        copied = True
        break

if not copied:
    print("\nWarning: Could not locate resulting best.pt file.")
