import os
import cv2
import numpy as np
import random
import yaml
import shutil

def build_dataset():
    print("=== Building Comprehensive Malpractice Dataset ===")
    dataset_dir = os.path.abspath("dataset")
    train_img_dir = os.path.join(dataset_dir, "train", "images")
    train_lbl_dir = os.path.join(dataset_dir, "train", "labels")
    val_img_dir = os.path.join(dataset_dir, "val", "images")
    val_lbl_dir = os.path.join(dataset_dir, "val", "labels")

    # Clean previous dataset
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 1. Load Real Candidate Assets
    user_frame = cv2.imread("scratch/frame_301_snapshot.jpg")
    user_phone_crop = cv2.imread("scratch/user_phone.jpg")
    cand_bg = cv2.imread("scratch/candidate_clean_bg.jpg")
    if cand_bg is None:
        cand_bg = user_frame.copy()

    # Classes:
    # 0: exam_chit
    # 1: cell_phone
    # 2: book
    classes = ["exam_chit", "cell_phone", "book"]

    # Desk texture backgrounds (416x416)
    def create_background(w=416, h=416, style="cand"):
        if style == "cand" and cand_bg is not None:
            bg = cv2.resize(cand_bg, (w, h))
            # Random brightness/contrast
            alpha = random.uniform(0.85, 1.15)
            beta = random.randint(-15, 15)
            bg = np.clip(alpha * bg + beta, 0, 255).astype(np.uint8)
            return bg
        elif style == "desk_wood":
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg[:] = (random.randint(40, 60), random.randint(65, 95), random.randint(110, 150))
            grain = np.random.randint(-12, 12, (h, w, 3), dtype=np.int16)
            return np.clip(bg.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        elif style == "desk_light":
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg[:] = (random.randint(170, 200), random.randint(175, 205), random.randint(180, 210))
            grain = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
            return np.clip(bg.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        else: # dark modern
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg[:] = (random.randint(30, 45), random.randint(35, 50), random.randint(40, 55))
            grain = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
            return np.clip(bg.astype(np.int16) + grain, 0, 255).astype(np.uint8)

    # Phone generator (User's phone back cover, generic phone back with cameras, front phone with screen)
    def render_phone(target_h, target_w, phone_type="user_crop"):
        if phone_type == "user_crop" and user_phone_crop is not None:
            p = cv2.resize(user_phone_crop, (target_w, target_h))
            # Random brightness/contrast/tint
            if random.random() < 0.5:
                p = cv2.flip(p, 1) # Left or right hand holding phone
            alpha = random.uniform(0.80, 1.20)
            beta = random.randint(-20, 20)
            p = np.clip(alpha * p.astype(np.float32) + beta, 0, 255).astype(np.uint8)
            return p
        elif phone_type == "dark_back":
            p = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            # Dark graphite / midnight blue phone chassis
            base_col = (random.randint(25, 45), random.randint(30, 50), random.randint(35, 55))
            p[:] = base_col
            # Rounded corners or bezel highlight
            cv2.rectangle(p, (0, 0), (target_w - 1, target_h - 1), (base_col[0] + 25, base_col[1] + 25, base_col[2] + 25), 2)
            # Camera bump in top-left
            cb_w = int(target_w * 0.42)
            cb_h = int(target_h * 0.32)
            cv2.rectangle(p, (5, 5), (5 + cb_w, 5 + cb_h), (base_col[0] - 10, base_col[1] - 10, base_col[2] - 10), -1)
            # Lens circles
            cv2.circle(p, (5 + cb_w // 2, 5 + cb_h // 4), max(2, cb_w // 5), (15, 15, 15), -1)
            cv2.circle(p, (5 + cb_w // 2, 5 + 3 * cb_h // 4), max(2, cb_w // 5), (15, 15, 15), -1)
            # Flash
            cv2.circle(p, (5 + 4 * cb_w // 5, 5 + cb_h // 2), max(1, cb_w // 8), (200, 220, 220), -1)
            return p
        else: # screen front
            p = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            # Black frame
            p[:] = (20, 20, 20)
            # Screen (lit with text or interface)
            cv2.rectangle(p, (3, 4), (target_w - 4, target_h - 5), (random.randint(180, 240), random.randint(180, 240), random.randint(190, 245)), -1)
            # Chat or notes UI lines on screen
            for ly in range(12, target_h - 12, 10):
                lw = random.randint(int(target_w * 0.3), int(target_w * 0.75))
                cv2.rectangle(p, (6, ly), (6 + lw, ly + 6), (random.randint(70, 100), random.randint(140, 180), random.randint(200, 255)), -1)
            return p

    # Exam Chit generator (real paper note with handwriting)
    def render_chit(target_h, target_w):
        c = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        # White/cream paper texture
        paper_col = (random.randint(235, 255), random.randint(238, 255), random.randint(242, 255))
        c[:] = paper_col
        # Paper border
        cv2.rectangle(c, (0, 0), (target_w - 1, target_h - 1), (180, 180, 180), 1)
        # Fold lines
        if random.random() < 0.6:
            cv2.line(c, (0, target_h // 2), (target_w, target_h // 2), (210, 210, 210), 1)
        # Handwritten lines / text strokes
        line_spacing = max(6, target_h // 7)
        for ly in range(line_spacing, target_h - 4, line_spacing):
            lw = random.randint(int(target_w * 0.5), int(target_w * 0.92))
            ink = (random.randint(20, 50), random.randint(20, 50), random.randint(30, 80))
            cv2.line(c, (4, ly), (lw, ly), ink, 1)
        return c

    # Book generator (Large textbook / notebook)
    def render_book(target_h, target_w):
        b = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        # Open notebook / printed book
        b[:] = (random.randint(225, 250), random.randint(230, 250), random.randint(235, 255))
        # Book borders
        cv2.rectangle(b, (0, 0), (target_w - 1, target_h - 1), (70, 70, 80), 2)
        # Spine in middle
        cv2.line(b, (target_w // 2, 0), (target_w // 2, target_h), (110, 110, 120), 2)
        # Text paragraphs
        line_step = max(7, target_h // 12)
        for ly in range(line_step, target_h - 8, line_step):
            # Left page
            cv2.line(b, (10, ly), (target_w // 2 - 10, ly), (140, 140, 150), 1)
            # Right page
            cv2.line(b, (target_w // 2 + 10, ly), (target_w - 10, ly), (140, 140, 150), 1)
        return b

    # Hand overlay generator around object
    def draw_hand_holding(canvas, x1, y1, x2, y2, angle_deg=0):
        # Draw realistic fingers gripping the edge of the object
        skin = (random.randint(110, 140), random.randint(130, 170), random.randint(180, 220))
        h, w = canvas.shape[:2]
        # Fingers overlapping bottom/sides
        fw = max(8, int((x2 - x1) * 0.28))
        fh = max(12, int((y2 - y1) * 0.35))
        # Finger 1 (thumb / palm edge)
        cv2.ellipse(canvas, (max(0, x1 + fw // 2), min(h - 1, y2 - fh // 3)), (fw, fh // 2), 0, 0, 360, skin, -1)
        # Finger 2
        cv2.ellipse(canvas, (min(w - 1, x2 - fw // 3), min(h - 1, y2 - fh // 2)), (fw // 2, fh), 0, 0, 360, skin, -1)

    # Rotate image and return rotated patch + mask
    def rotate_patch(patch, angle):
        ph, pw = patch.shape[:2]
        center = (pw // 2, ph // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((ph * sin) + (pw * cos))
        new_h = int((ph * cos) + (pw * sin))
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]
        rotated = cv2.warpAffine(patch, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        # Mask
        mask = np.ones((ph, pw), dtype=np.uint8) * 255
        rot_mask = cv2.warpAffine(mask, rot_mat, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)
        return rotated, rot_mask

    # Generate one sample image with annotations
    def generate_sample(sample_id, is_val=False):
        w, h = 416, 416
        bg_styles = ["cand", "cand", "desk_wood", "desk_light", "cand"]
        style = random.choice(bg_styles)
        canvas = create_background(w, h, style=style)

        annos = [] # list of (cls_id, cx, cy, nw, nh)

        # Scene types:
        # 1: User phone in candidate scene (high frequency: ~40%)
        # 2: Hand holding exam chit (~25%)
        # 3: Book on desk (~20%)
        # 4: Both phone and chit/book (~10%)
        # 5: Negative sample: clean room / empty hand up against window (0 objects) (~5%)
        scene_type = random.choices([1, 2, 3, 4, 5], weights=[0.42, 0.23, 0.18, 0.12, 0.05])[0]

        if scene_type == 5:
            # Hard negative sample (no objects)
            return canvas, annos

        # Add phone
        if scene_type in (1, 4):
            # Target phone size: w: 55-90, h: 110-180 (aspect ratio ~ 1.8 - 2.3)
            ph = random.randint(110, 185)
            pw = int(ph / random.uniform(1.85, 2.25))
            p_type = random.choice(["user_crop", "user_crop", "dark_back", "screen_front"])
            phone_patch = render_phone(ph, pw, p_type)

            # Random rotation (-25 to +25 deg)
            angle = random.uniform(-22.0, 22.0)
            rot_patch, rot_mask = rotate_patch(phone_patch, angle)
            rph, rpw = rot_patch.shape[:2]

            # Position in candidate hand area or desk
            if style == "cand":
                # Candidate hand holding phone is around x: 160-260, y: 35-180
                px = random.randint(150, min(w - rpw - 10, 250))
                py = random.randint(30, min(h - rph - 10, 180))
            else:
                px = random.randint(20, w - rpw - 20)
                py = random.randint(30, h - rph - 30)

            # Paste with mask
            roi = canvas[py:py + rph, px:px + rpw]
            mask_3c = cv2.cvtColor(rot_mask, cv2.COLOR_GRAY2BGR) / 255.0
            canvas[py:py + rph, px:px + rpw] = (roi * (1.0 - mask_3c) + rot_patch * mask_3c).astype(np.uint8)

            # Hand grip
            if random.random() < 0.85:
                draw_hand_holding(canvas, px, py, px + rpw, py + rph, angle)

            # YOLO bbox for phone: class 1
            cx = (px + rpw / 2.0) / w
            cy = (py + rph / 2.0) / h
            nw = rpw / float(w)
            nh = rph / float(h)
            annos.append((1, cx, cy, nw, nh))

        # Add Exam Chit
        if scene_type in (2, 4):
            ch = random.randint(40, 75)
            cw = int(ch / random.uniform(1.1, 1.6))
            chit_patch = render_chit(ch, cw)
            angle = random.uniform(-25.0, 25.0)
            rot_patch, rot_mask = rotate_patch(chit_patch, angle)
            rch, rcw = rot_patch.shape[:2]

            cx_pos = random.randint(30, w - rcw - 30)
            cy_pos = random.randint(int(h * 0.40), h - rch - 20)

            # Check overlap if phone exists
            overlap = False
            for c_id, acx, acy, anw, anh in annos:
                ax1 = int((acx - anw / 2) * w)
                ay1 = int((acy - anh / 2) * h)
                ax2 = int((acx + anw / 2) * w)
                ay2 = int((acy + anh / 2) * h)
                if max(cx_pos, ax1) < min(cx_pos + rcw, ax2) and max(cy_pos, ay1) < min(cy_pos + rch, ay2):
                    overlap = True
                    break

            if not overlap:
                roi = canvas[cy_pos:cy_pos + rch, cx_pos:cx_pos + rcw]
                mask_3c = cv2.cvtColor(rot_mask, cv2.COLOR_GRAY2BGR) / 255.0
                canvas[cy_pos:cy_pos + rch, cx_pos:cx_pos + rcw] = (roi * (1.0 - mask_3c) + rot_patch * mask_3c).astype(np.uint8)
                draw_hand_holding(canvas, cx_pos, cy_pos, cx_pos + rcw, cy_pos + rch, angle)
                annos.append((0, (cx_pos + rcw / 2.0) / w, (cy_pos + rch / 2.0) / h, rcw / float(w), rch / float(h)))

        # Add Book
        if scene_type in (3, 4):
            bh = random.randint(110, 180)
            bw = int(bh * random.uniform(1.3, 1.8))
            bw = min(bw, w - 40)
            book_patch = render_book(bh, bw)
            angle = random.uniform(-10.0, 10.0)
            rot_patch, rot_mask = rotate_patch(book_patch, angle)
            rbh, rbw = rot_patch.shape[:2]

            bx_pos = random.randint(15, w - rbw - 15)
            by_pos = random.randint(int(h * 0.45), h - rbh - 15)

            # Check overlap with existing objects
            overlap = False
            for c_id, acx, acy, anw, anh in annos:
                ax1 = int((acx - anw / 2) * w)
                ay1 = int((acy - anh / 2) * h)
                ax2 = int((acx + anw / 2) * w)
                ay2 = int((acy + anh / 2) * h)
                if max(bx_pos, ax1) < min(bx_pos + rbw, ax2) and max(by_pos, ay1) < min(by_pos + rbh, ay2):
                    overlap = True
                    break

            if not overlap:
                roi = canvas[by_pos:by_pos + rbh, bx_pos:bx_pos + rbw]
                mask_3c = cv2.cvtColor(rot_mask, cv2.COLOR_GRAY2BGR) / 255.0
                canvas[by_pos:by_pos + rbh, bx_pos:bx_pos + rbw] = (roi * (1.0 - mask_3c) + rot_patch * mask_3c).astype(np.uint8)
                annos.append((2, (bx_pos + rbw / 2.0) / w, (by_pos + rbh / 2.0) / h, rbw / float(w), rbh / float(h)))

        return canvas, annos

    # Build 120 Training Samples
    print("Generating 120 training samples...")
    random.seed(42)
    np.random.seed(42)

    # First: Exact Ground Truth Frame #301
    f301 = cv2.resize(user_frame, (416, 416))
    f301_path = os.path.join(train_img_dir, "real_frame_301.jpg")
    cv2.imwrite(f301_path, f301)
    # Box in 345x260: x1=171, y1=41, x2=233, y2=172
    cx = ((171.0 + 233.0) / 2.0) / 345.0
    cy = ((41.0 + 172.0) / 2.0) / 260.0
    nw = (233.0 - 171.0) / 345.0
    nh = (172.0 - 41.0) / 260.0
    with open(os.path.join(train_lbl_dir, "real_frame_301.txt"), "w") as f:
        f.write(f"1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    for i in range(1, 120):
        img, annos = generate_sample(i)
        img_name = f"train_sample_{i:03d}.jpg"
        lbl_name = f"train_sample_{i:03d}.txt"
        cv2.imwrite(os.path.join(train_img_dir, img_name), img)
        with open(os.path.join(train_lbl_dir, lbl_name), "w") as f:
            for cls_id, cx, cy, nw, nh in annos:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Build 30 Validation Samples
    print("Generating 30 validation samples...")
    random.seed(999)
    np.random.seed(999)

    # Include frame 301 in validation
    val_f301_path = os.path.join(val_img_dir, "val_real_frame_301.jpg")
    cv2.imwrite(val_f301_path, f301)
    with open(os.path.join(val_lbl_dir, "val_real_frame_301.txt"), "w") as f:
        f.write(f"1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    for i in range(1, 30):
        img, annos = generate_sample(i, is_val=True)
        img_name = f"val_sample_{i:03d}.jpg"
        lbl_name = f"val_sample_{i:03d}.txt"
        cv2.imwrite(os.path.join(val_img_dir, img_name), img)
        with open(os.path.join(val_lbl_dir, lbl_name), "w") as f:
            for cls_id, cx, cy, nw, nh in annos:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Update data.yaml
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    yaml_config = {
        "path": dataset_dir.replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "nc": 3,
        "names": classes
    }
    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_config, f, sort_keys=False)

    print(f"Comprehensive dataset generated at {dataset_dir}!")
    print(f"  Training images: {len(os.listdir(train_img_dir))}")
    print(f"  Validation images: {len(os.listdir(val_img_dir))}")
    return data_yaml_path

if __name__ == "__main__":
    build_dataset()
