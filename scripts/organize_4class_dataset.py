"""
Organize AI-Generated Images into 4-Class Folder Structure
============================================================

Reorganizes images from:
  ai_generated/class1_raw/
  ai_generated/class2_compressed/
  ai_generated/class3_degraded/
  ai_generated/class4_edited_real/

Into:
  dataset_4class/train/[real|perfect_ai|compressed_ai|edited_ai]/
  dataset_4class/val/[real|perfect_ai|compressed_ai|edited_ai]/
  dataset_4class/test/[real|perfect_ai|compressed_ai|edited_ai]/

Split: 70% train, 15% val, 15% test
"""

import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
PROJECT_ROOT = Path("e:/BML/Semester-VI/Prj-3")
OUTPUT_DIR = PROJECT_ROOT / "dataset_4class"

REAL_SOURCES = [
    PROJECT_ROOT / "food_101" / "food-101" / "food-101" / "images",
    PROJECT_ROOT / "food_image_dataset" / "data" / "UECFOOD256 2",
    PROJECT_ROOT / "food_image_dataset" / "data" / "aicrowd_food_recognition",
    PROJECT_ROOT / "indian_food_data" / "image_for _cuisines" / "data",
]

AI_SOURCES = {
    "perfect_ai": [
        PROJECT_ROOT / "ai_generated" / "indian",          # Old 68 SD3 images
        PROJECT_ROOT / "ai_generated" / "class1_raw",      # SDXL RealVisXL raw
        PROJECT_ROOT / "ai_generated" / "flux_schnell",    # Flux.1 Schnell (Flow-Matching)
        PROJECT_ROOT / "ai_generated" / "stable_cascade",  # Stable Cascade (3-stage)
        PROJECT_ROOT / "ai_generated" / "kandinsky3",      # Kandinsky 2.2 (multilingual)
        PROJECT_ROOT / "ai_generated" / "sdxl_turbo",      # SDXL Turbo + Lightning (distilled)
    ],
    "compressed_ai": [
        PROJECT_ROOT / "ai_generated" / "class2_compressed",
    ],
    "edited_ai": [
        PROJECT_ROOT / "ai_generated" / "class3_degraded",   # SDXL degraded/blurred
        PROJECT_ROOT / "ai_generated" / "class4_edited_real", # Inpainting + overlay fraud
    ],
}

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


def collect_images(source_dirs):
    """Collect all image files from given directories."""
    images = []
    for src_dir in source_dirs:
        if not src_dir.exists():
            print(f"  [!] Skipping missing: {src_dir}")
            continue
        
        for f in src_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(f)
    
    return images


def split_images(images, ratios):
    """Split images into train/val/test."""
    random.shuffle(images)
    total = len(images)
    
    train_end = int(total * ratios["train"])
    val_end = train_end + int(total * ratios["val"])
    
    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }


def main():
    print("=" * 60)
    print("ORGANIZING IMAGES INTO 4-CLASS FOLDER STRUCTURE")
    print("=" * 60)
    
    random.seed(SEED)
    
    # Clear and create output directory
    if OUTPUT_DIR.exists():
        print(f"\nRemoving existing: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ["train", "val", "test"]:
        for class_name in ["real", "perfect_ai", "compressed_ai", "edited_ai"]:
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Collect and split REAL images
    print("\n[1/4] Processing REAL images...")
    real_images = collect_images(REAL_SOURCES)
    print(f"  Total real: {len(real_images)}")
    
    # Sample down to ~12000
    if len(real_images) > 12000:
        real_images = random.sample(real_images, 12000)
        print(f"  Sampled: 12000")
    
    real_splits = split_images(real_images, SPLIT_RATIOS)
    for split, imgs in real_splits.items():
        for idx, img in enumerate(imgs):
            dest = OUTPUT_DIR / split / "real" / f"real_{idx:05d}{img.suffix}"
            shutil.copy2(img, dest)
        print(f"    {split}: {len(imgs)} images")
    
    # Process AI classes
    for ai_class, source_dirs in AI_SOURCES.items():
        print(f"\n[{list(AI_SOURCES.keys()).index(ai_class) + 2}/4] Processing {ai_class.upper()}...")
        ai_images = collect_images(source_dirs)
        print(f"  Total {ai_class}: {len(ai_images)}")
        
        if len(ai_images) == 0:
            print(f"  [!] No images found, skipping")
            continue
        
        ai_splits = split_images(ai_images, SPLIT_RATIOS)
        for split, imgs in ai_splits.items():
            for idx, img in enumerate(imgs):
                dest = OUTPUT_DIR / split / ai_class / f"{ai_class}_{idx:05d}{img.suffix}"
                shutil.copy2(img, dest)
            print(f"    {split}: {len(imgs)} images")
    
    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}:")
        for class_name in ["real", "perfect_ai", "compressed_ai", "edited_ai"]:
            count = len(list((OUTPUT_DIR / split / class_name).iterdir()))
            print(f"  {class_name:15s}: {count:5d} images")
    
    # ── Build dataset_index.csv ──────────────────────────────
    csv_path = PROJECT_ROOT / "dataset_index.csv"
    print(f"\n{'=' * 60}")
    print("BUILDING dataset_index.csv")
    print("=" * 60)

    rows = []
    for split in ["train", "val", "test"]:
        for class_name in ["real", "perfect_ai", "compressed_ai", "edited_ai"]:
            class_dir = OUTPUT_DIR / split / class_name
            if not class_dir.exists():
                continue
            for img_file in sorted(class_dir.iterdir()):
                if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    rel_path = img_file.relative_to(PROJECT_ROOT)
                    rows.append((
                        str(rel_path),   # image_path
                        split,           # split  (train / val / test)
                        class_name,      # label  (real / perfect_ai / compressed_ai / edited_ai)
                    ))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "split", "label"])
        writer.writerows(rows)

    print(f"  Wrote {len(rows)} entries to {csv_path}")

    # Per-split breakdown
    for split in ["train", "val", "test"]:
        split_rows = [r for r in rows if r[1] == split]
        print(f"  {split:6s}: {len(split_rows)} rows")

    print(f"\n{'=' * 60}")
    print(f"Dataset ready: {OUTPUT_DIR}")
    print(f"CSV ready:     {csv_path}")
    print("You can now run: python train_4class_detector.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
