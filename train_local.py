import os
import shutil
from ultralytics import YOLO

def main():
    print("=== Starting Local YOLO Training ===")
    model_path = "yolo26n.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print(f"Loading base weights from {model_path}...")
    model = YOLO(model_path)

    data_yaml = os.path.abspath("exam_dataset/data.yaml")
    print(f"Training on dataset: {data_yaml}")

    # Train for 30 epochs, imgsz=416, batch=8, CPU
    results = model.train(
        data=data_yaml,
        epochs=30,
        imgsz=416,
        batch=8,
        device="cpu",
        workers=0, # Windows multiprocessing safe
        project="runs",
        name="train_chits",
        exist_ok=True,
        verbose=True
    )

    # Locate best.pt
    target_best = os.path.abspath("best.pt")
    possible_bests = [
        os.path.join(str(results.save_dir), "weights", "best.pt"),
        os.path.join("runs", "detect", "runs", "train_chits", "weights", "best.pt"),
        os.path.join("runs", "train_chits", "weights", "best.pt")
    ]

    copied = False
    for p in possible_bests:
        if os.path.exists(p):
            shutil.copy2(p, target_best)
            print(f"Successfully copied trained weights from {p} to {target_best}")
            copied = True
            break

    if not copied:
        possible_lasts = [
            os.path.join(str(results.save_dir), "weights", "last.pt"),
            os.path.join("runs", "detect", "runs", "train_chits", "weights", "last.pt"),
            os.path.join("runs", "train_chits", "weights", "last.pt")
        ]
        for p in possible_lasts:
            if os.path.exists(p):
                shutil.copy2(p, target_best)
                print(f"Copied last weights from {p} to {target_best}")
                copied = True
                break

    if not copied:
        print("Warning: Neither best.pt nor last.pt found.")

    print("=== Training Complete ===")

if __name__ == "__main__":
    main()
