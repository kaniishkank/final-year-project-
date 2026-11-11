import os
import shutil
from ultralytics import YOLO

def main():
    print("=== Starting Local YOLOv8 Malpractice Model Training ===")
    model_path = "yolov8n.pt"
    if not os.path.exists(model_path):
        print(f"Base model {model_path} not found locally. Ultralytics will auto-fetch.")

    print(f"Loading base YOLOv8 weights from {model_path}...")
    model = YOLO(model_path)

    # Ingest / verify dataset
    data_yaml = os.path.abspath("dataset/data.yaml")
    if not os.path.exists(data_yaml):
        from scripts.build_comprehensive_dataset import build_dataset
        data_yaml = build_dataset()

    print(f"Training on dataset: {data_yaml}")

    # Train for 15 epochs, imgsz=416, batch=8 on CPU
    results = model.train(
        data=data_yaml,
        epochs=15,
        imgsz=416,
        batch=8,
        device="cpu",
        workers=0, # Windows multiprocessing safe
        project="runs",
        name="yolov8_malpractice",
        exist_ok=True,
        verbose=True
    )

    # Copy best weights to root and backend/weights
    target_root = os.path.abspath("best.pt")
    target_backend_dir = os.path.abspath("backend/weights")
    os.makedirs(target_backend_dir, exist_ok=True)
    target_backend = os.path.join(target_backend_dir, "best.pt")

    search_dirs = [
        os.path.join(str(results.save_dir), "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "yolov8_malpractice", "weights", "best.pt"),
        os.path.join("runs", "yolov8_malpractice", "weights", "best.pt")
    ]

    copied = False
    for p in search_dirs:
        if os.path.exists(p):
            shutil.copy2(p, target_root)
            shutil.copy2(p, target_backend)
            print(f"Successfully copied trained weights from {p} to:")
            print(f"  -> {target_root}")
            print(f"  -> {target_backend}")
            copied = True
            break

    if not copied:
        # Fallback to last.pt
        for p in [os.path.join(str(results.save_dir), "weights", "last.pt"), os.path.join("runs", "yolov8_malpractice", "weights", "last.pt")]:
            if os.path.exists(p):
                shutil.copy2(p, target_root)
                shutil.copy2(p, target_backend)
                print(f"Copied last weights to {target_root} and {target_backend}")
                copied = True
                break

    print("=== YOLOv8 Malpractice Training Complete ===")

if __name__ == "__main__":
    main()
