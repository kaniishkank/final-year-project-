import os
import glob
import shutil
import torch
from ultralytics import YOLO

# Detect compute device
device = 0 if torch.cuda.is_available() else "cpu"
print(f"[EviGuard Engine] Training initialized on device: {device}")

yaml_files = glob.glob("dataset/**/data.yaml", recursive=True)
data_yaml = yaml_files[0] if yaml_files else "dataset/data.yaml"

# Load YOLOv8 Small for higher capacity on micro-chits
# (Falls back to yolov8n.pt if system is pure CPU to ensure completion)
base_weights = "yolov8s.pt" if torch.cuda.is_available() else "yolov8n.pt"
print(f"[EviGuard Engine] Base weights loaded: {base_weights}")
model = YOLO(base_weights)

# Deep training loop optimized for small-object recall & anti-false-positives
results = model.train(
    data=data_yaml,
    epochs=120,          # Deep convergence
    imgsz=640,           # Native resolution for micro-notes & slips
    batch=8 if device == "cpu" else 16,
    device=device,
    optimizer="AdamW",
    lr0=0.003,
    lrf=0.01,
    patience=25,         # Early stopping if loss plateaus
    augment=True,        # Flips, perspective shifts, scaling
    mosaic=1.0,          # Crucial for detecting small chits
    mixup=0.15,
    copy_paste=0.1,      # Superimposes chits/phones on background scenes
    close_mosaic=15,     # Disables mosaic near the end for sharp box boundaries
    project="eviguard_checkpoints",
    name="yolov8_perfect",
    save=True,
    verbose=True,
    exist_ok=True,
    workers=2 if device == "cpu" else 8
)

# Copy final best weights to project root and backend/weights
best_path = "eviguard_checkpoints/yolov8_perfect/weights/best.pt"
if os.path.exists(best_path):
    shutil.copy(best_path, "best.pt")
    os.makedirs("backend/weights", exist_ok=True)
    shutil.copy(best_path, "backend/weights/best.pt")
    print("\n>>> Deep-trained weights 'best.pt' successfully placed in root and backend/weights/! <<<")
else:
    print(f"\nWarning: Expected checkpoint not found at {best_path}")
