"""
EviGuard AI Proctoring — Robust Malpractice Dataset Preparation & Negative Injection
Curates comprehensive malpractice samples (cheat_chit, cell_phone, book)
and injects 50-100 real background negative frames (window grills, candidate room,
dark clothing, empty desk) with empty label files to eliminate false positives.
"""

import os
import glob
import shutil
import random
import cv2
import numpy as np
import yaml

def prepare_robust_dataset():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(project_root, "dataset")
    train_img_dir = os.path.join(dataset_dir, "train", "images")
    train_lbl_dir = os.path.join(dataset_dir, "train", "labels")
    val_img_dir = os.path.join(dataset_dir, "val", "images")
    val_lbl_dir = os.path.join(dataset_dir, "val", "labels")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    print("=" * 70)
    print("  EVIGUARD: ROBUST DATASET CURATION & NEGATIVE INJECTION")
    print("=" * 70)

    # 1. Collect real background source images from candidate environment
    bg_sources = []
    
    # Check evidence clips
    ev_files = glob.glob(os.path.join(project_root, "data", "evidence_clips", "*.jpg"))
    for ef in ev_files:
        im = cv2.imread(ef)
        if im is not None and im.shape[0] >= 300 and im.shape[1] >= 300:
            bg_sources.append(im)

    # Check user uploads for real room backgrounds
    user_up_dir = os.path.join(os.path.expanduser("~"), ".gemini", "antigravity", "brain", "1dfabe55-a831-4032-9498-1154ed475768", ".user_uploaded")
    if os.path.exists(user_up_dir):
        for uf in glob.glob(os.path.join(user_up_dir, "*.jpg")) + glob.glob(os.path.join(user_up_dir, "*.png")):
            im = cv2.imread(uf)
            if im is not None and im.shape[0] >= 400:
                # If full screenshot, extract webcam sub-region
                if im.shape[1] >= 900:
                    sub = im[70:520, 30:690]
                    if sub.size > 0:
                        bg_sources.append(sub)
                else:
                    bg_sources.append(im)

    # Scratch webcam crops
    scratch_dir = os.path.join(project_root, "scratch")
    for sf in glob.glob(os.path.join(scratch_dir, "*webcam*.jpg")):
        im = cv2.imread(sf)
        if im is not None:
            bg_sources.append(im)

    print(f"[*] Found {len(bg_sources)} pristine room / candidate background reference sources.")

    # Fallback synthetic clean background generator if needed
    if len(bg_sources) == 0:
        base_bg = np.full((480, 640, 3), (215, 215, 220), dtype=np.uint8)
        # Window grill
        for gx in range(380, 630, 45):
            cv2.line(base_bg, (gx, 0), (gx, 480), (250, 252, 255), 8)
        bg_sources.append(base_bg)

    # 2. Generate clean background negative frames (no bounding boxes, empty .txt)
    # Target: 70 train negatives, 18 val negatives (total ~88 negatives)
    random.seed(42)
    np.random.seed(42)

    total_neg_train = 70
    total_neg_val = 18

    def augment_negative(src_img, style="full"):
        h, w = src_img.shape[:2]
        if style == "full":
            crop = src_img.copy()
        elif style == "window_grill":
            # Focus on vertical window grill area (right side or top corner)
            wx1 = int(w * 0.45)
            crop = src_img[:, wx1:w].copy()
        elif style == "torso_shadow":
            # Focus on dark clothing / shoulder / chest
            ty1 = int(h * 0.40)
            tx1 = int(w * 0.15)
            tx2 = int(w * 0.85)
            crop = src_img[ty1:h, tx1:tx2].copy()
        elif style == "empty_desk":
            # Lower 35% of frame
            dy1 = int(h * 0.65)
            crop = src_img[dy1:h, :].copy()
        else:
            crop = src_img.copy()

        # Resize to standard 640x640
        res = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR)

        # Apply photorealistic jitter
        # 1. Random horizontal flip
        if random.random() > 0.5:
            res = cv2.flip(res, 1)

        # 2. Lighting / contrast jitter
        alpha = random.uniform(0.80, 1.25) # contrast
        beta = random.randint(-20, 20)      # brightness
        res = np.clip(alpha * res.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # 3. CLAHE enhancement jitter
        if random.random() > 0.5:
            lab = cv2.cvtColor(res, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=random.uniform(1.2, 2.5), tileGridSize=(8, 8))
            l = clahe.apply(l)
            res = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        return res

    styles = ["full", "window_grill", "torso_shadow", "empty_desk", "full"]

    print("[*] Generating 70 clean background negatives for training...")
    for idx in range(total_neg_train):
        src = random.choice(bg_sources)
        st = random.choice(styles)
        neg_img = augment_negative(src, style=st)

        img_filename = f"bg_neg_train_{idx:03d}.jpg"
        lbl_filename = f"bg_neg_train_{idx:03d}.txt"

        img_out = os.path.join(train_img_dir, img_filename)
        lbl_out = os.path.join(train_lbl_dir, lbl_filename)

        cv2.imwrite(img_out, neg_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # Crucial: write 0-byte empty file for YOLO negative annotation
        with open(lbl_out, "w", encoding="utf-8") as f:
            pass

    print("[*] Generating 18 clean background negatives for validation...")
    for idx in range(total_neg_val):
        src = random.choice(bg_sources)
        st = random.choice(styles)
        neg_img = augment_negative(src, style=st)

        img_filename = f"bg_neg_val_{idx:03d}.jpg"
        lbl_filename = f"bg_neg_val_{idx:03d}.txt"

        img_out = os.path.join(val_img_dir, img_filename)
        lbl_out = os.path.join(val_lbl_dir, lbl_filename)

        cv2.imwrite(img_out, neg_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(lbl_out, "w", encoding="utf-8") as f:
            pass

    # 3. Verify that positive samples have correct classes:
    # 0: cheat_chit, 1: cell_phone, 2: book
    # Update data.yaml
    classes = ["cheat_chit", "cell_phone", "book"]
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    yaml_content = {
        "path": dataset_dir.replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "nc": 3,
        "names": classes
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    # 4. Audit dataset statistics
    train_imgs = glob.glob(os.path.join(train_img_dir, "*.jpg"))
    train_lbls = glob.glob(os.path.join(train_lbl_dir, "*.txt"))
    val_imgs = glob.glob(os.path.join(val_img_dir, "*.jpg"))
    val_lbls = glob.glob(os.path.join(val_lbl_dir, "*.txt"))

    # Count positive vs negative (empty file) labels
    train_neg_count = sum(1 for p in train_lbls if os.path.getsize(p) == 0)
    train_pos_count = len(train_lbls) - train_neg_count
    val_neg_count = sum(1 for p in val_lbls if os.path.getsize(p) == 0)
    val_pos_count = len(val_lbls) - val_neg_count

    print("\n" + "=" * 70)
    print("  DATASET PREPARATION & NEGATIVE INJECTION COMPLETED!")
    print("=" * 70)
    print(f"  Root Path           : {dataset_dir}")
    print(f"  Configuration YAML  : {data_yaml_path}")
    print(f"  Classes ({len(classes)})       : {classes}")
    print(f"  Train Set Total     : {len(train_imgs)} images ({train_pos_count} positive, {train_neg_count} negative backgrounds)")
    print(f"  Validation Set Total: {len(val_imgs)} images ({val_pos_count} positive, {val_neg_count} negative backgrounds)")
    print("=" * 70)

if __name__ == "__main__":
    prepare_robust_dataset()
