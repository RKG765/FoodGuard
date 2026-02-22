"""
Build Detector Dataset CSV
===========================

Creates a balanced dataset_index.csv for the 3-class detector:

  Class 0: pristine_real   (~5000 images from food_101 + food_image_dataset + indian_food_data)
  Class 1: ai_generated    (68 old SD3 images + future class1-3 pipeline outputs)
  Class 2: manipulated     (future class4_edited_real pipeline outputs)

CSV Columns:
  file_path  : relative path from project root
  label      : 0, 1, or 2
  source     : e.g. Food-101, UECFOOD256, Aicrowd, Indian-Food, SD3-AI, SDXL-AI
  subtype    : e.g. pristine, perfect, blurred, compressed, edited

Usage:
    python scripts/build_detector_csv.py
    python scripts/build_detector_csv.py --real-count 5000
"""

import csv
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT

# ============================================================
# Real image sources  ->  (directory, source_name)
# ============================================================

REAL_IMAGE_SOURCES = {
    "Food-101":  DATA_ROOT / "food_101" / "food-101" / "food-101" / "images",
    "UECFOOD256": DATA_ROOT / "food_image_dataset" / "data" / "UECFOOD256 2",
    "Aicrowd":   DATA_ROOT / "food_image_dataset" / "data" / "aicrowd_food_recognition",
    "Indian-Food": DATA_ROOT / "indian_food_data" / "image_for _cuisines" / "data",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# AI image sources
# ============================================================

OLD_AI_DIR = DATA_ROOT / "ai_generated" / "indian"

# New 4-class dirs (created by generate_ai_images.py after user runs it)
NEW_AI_CLASS_MAP = {
    "class1_raw":        ("SDXL-AI", "perfect"),
    "class2_compressed": ("SDXL-AI", "compressed"),
    "class3_degraded":   ("SDXL-AI", "blurred"),
    "class4_edited_real": ("SDXL-AI", "edited"),  # -> label 2 (manipulated)
}


def collect_real_images() -> list:
    """Collect all real image paths with source names."""
    all_paths = []
    for source_name, base_dir in REAL_IMAGE_SOURCES.items():
        if not base_dir.exists():
            print(f"  [!] Skipping {source_name}: {base_dir} not found")
            continue
        count = 0
        for f in base_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                rel = str(f.relative_to(DATA_ROOT)).replace("\\", "/")
                all_paths.append((rel, source_name))
                count += 1
        print(f"  [OK] {source_name}: {count} images")
    return all_paths


def collect_old_ai_images() -> list:
    """Collect the 68 old SD3-generated images."""
    rows = []
    if not OLD_AI_DIR.exists():
        print(f"  [!] Old AI dir not found: {OLD_AI_DIR}")
        return rows

    for f in sorted(OLD_AI_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            rel = str(f.relative_to(DATA_ROOT)).replace("\\", "/")
            rows.append(rel)

    print(f"  [OK] Old AI (SD3): {len(rows)} images")
    return rows


def collect_new_ai_images() -> tuple:
    """Collect new 4-class pipeline images (if already generated).

    Returns:
        (ai_rows, manipulated_rows) where each row is (file_path, source, subtype)
    """
    ai_rows = []        # label 1: class1-3
    manip_rows = []     # label 2: class4

    for class_name, (source, subtype) in NEW_AI_CLASS_MAP.items():
        d = DATA_ROOT / "ai_generated" / class_name
        if not d.exists():
            continue
        count = 0
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                rel = str(f.relative_to(DATA_ROOT)).replace("\\", "/")
                row = (rel, source, subtype)
                if class_name == "class4_edited_real":
                    manip_rows.append(row)
                else:
                    ai_rows.append(row)
                count += 1
        if count > 0:
            print(f"  [OK] {class_name}: {count} images")

    return ai_rows, manip_rows


def build_csv(real_count: int = 5000, seed: int = 42):
    """Build the balanced dataset_index.csv."""
    print("=" * 60)
    print("BUILDING DETECTOR DATASET CSV")
    print("=" * 60)

    random.seed(seed)

    # --- Collect real images ---
    print("\n[1/3] Collecting real images...")
    all_real = collect_real_images()
    print(f"  Total real available: {len(all_real)}")

    if len(all_real) > real_count:
        sampled_real = random.sample(all_real, real_count)
        print(f"  Sampled: {real_count} images")
    else:
        sampled_real = all_real
        print(f"  Using all: {len(all_real)} images")

    # --- Collect AI images ---
    print("\n[2/3] Collecting AI-generated images...")
    old_ai = collect_old_ai_images()
    new_ai, new_manip = collect_new_ai_images()

    # --- Build CSV rows ---
    print("\n[3/3] Writing dataset_index.csv...")

    csv_path = DATA_ROOT / "dataset_index.csv"
    header = ["file_path", "label", "source", "subtype"]
    rows = []

    # Class 0: Pristine Real
    for rel_path, source in sampled_real:
        rows.append([rel_path, 0, source, "pristine"])

    # Class 1: AI-Generated -- old 68 perfect images
    for rel_path in old_ai:
        rows.append([rel_path, 1, "SD3-AI", "perfect"])

    # Class 1: AI-Generated -- new pipeline classes 1-3
    for rel_path, source, subtype in new_ai:
        rows.append([rel_path, 1, source, subtype])

    # Class 2: Manipulated -- new pipeline class 4
    for rel_path, source, subtype in new_manip:
        rows.append([rel_path, 2, source, subtype])

    # Shuffle
    random.shuffle(rows)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Summary
    label_counts = {0: 0, 1: 0, 2: 0}
    for row in rows:
        label_counts[int(row[1])] += 1

    subtype_counts = {}
    for row in rows:
        st = row[3]
        subtype_counts[st] = subtype_counts.get(st, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"DATASET CSV BUILT: {csv_path}")
    print(f"  Total: {len(rows)} rows")
    print(f"  Class 0 (Pristine Real):  {label_counts[0]}")
    print(f"  Class 1 (AI-Generated):   {label_counts[1]}")
    print(f"  Class 2 (Manipulated):    {label_counts[2]}")
    print(f"\n  Subtypes: {subtype_counts}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build detector dataset CSV")
    parser.add_argument("--real-count", type=int, default=5000,
                        help="Number of real images to sample (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    build_csv(real_count=args.real_count, seed=args.seed)
