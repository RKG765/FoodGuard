"""
Plan B: Direct Object Overlay (Simple, Guaranteed to Work)
===========================================================

Method: Load actual fraud object images, paste onto food photos.
Fast, simple, 100% visible.
"""

import random
import json
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

# Config
PROJECT_ROOT = Path("e:/BML/Semester-VI/Prj-3")
OUTPUT_DIR = PROJECT_ROOT / "ai_generated" / "class4_edited_real"
METADATA_FILE = OUTPUT_DIR / "fraud_metadata.json"

# Will use simple colored shapes as fraud objects
# (You can replace with actual images later)
FRAUD_OBJECTS = {
    "cockroach": 500,
    "housefly": 400,
    "small worm": 400,
    "human hair strand": 500,
    "mold patch": 400,
    "plastic fragment": 400,
    "metal shard": 200,
    "piece of paper": 200,
}

SEED = 42

def load_clean_pool():
    """Load clean 186K pool."""
    import csv
    pristine_5k = set()
    with open(PROJECT_ROOT / "dataset_index.csv", 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if int(row['label']) == 0:
                pristine_5k.add(str((PROJECT_ROOT / row['file_path']).resolve()))
    
    all_real = []
    for src_dir in [
        PROJECT_ROOT / "food_101" / "food-101" / "food-101" / "images",
        PROJECT_ROOT / "food_image_dataset" / "data" / "UECFOOD256 2",
        PROJECT_ROOT / "food_image_dataset" / "data" / "aicrowd_food_recognition",
        PROJECT_ROOT / "indian_food_data" / "image_for _cuisines" / "data",
    ]:
        if src_dir.exists():
            for f in src_dir.rglob("*"):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    all_real.append(str(f.resolve()))
    
    clean_pool = [p for p in all_real if p not in pristine_5k]
    return clean_pool


def create_fraud_object_image(obj_type, size=80):
    """Create simple fraud object (colored shape)."""
    # Simple colored blobs as placeholders
    obj_img = Image.new('RGBA', (size, size), (0,0,0,0))
    
    if "cockroach" in obj_type:
        # Brown ellipse
        from PIL import ImageDraw
        draw = ImageDraw.Draw(obj_img)
        draw.ellipse([10, 20, 70, 60], fill=(80, 50, 30, 255))
    elif "hair" in obj_type:
        # Dark wavy line
        from PIL import ImageDraw
        draw = ImageDraw.Draw(obj_img)
        draw.line([(10,10), (30,25), (50,20), (70,30)], fill=(40,30,25,255), width=2)
    elif "mold" in obj_type:
        # Green splatter
        from PIL import ImageDraw
        draw = ImageDraw.Draw(obj_img)
        draw.ellipse([15, 15, 65, 65], fill=(100, 140, 60, 200))
    else:
        # Default gray blob
        from PIL import ImageDraw
        draw = ImageDraw.Draw(obj_img)
        draw.ellipse([20, 20, 60, 60], fill=(120, 120, 120, 220))
    
    return obj_img


def add_fraud_to_image(source_path, fraud_object):
    """Add fraud object to food image via direct overlay."""
    try:
        source = Image.open(source_path).convert("RGB")
    except:
        return None, None
    
    source = source.resize((512, 512), Image.LANCZOS)
    
    # Create fraud object
    obj_size = random.randint(60, 100)
    fraud_img = create_fraud_object_image(fraud_object, obj_size)
    
    # Random position (center-biased)
    x = random.randint(int(512 * 0.3), int(512 * 0.7) - obj_size)
    y = random.randint(int(512 * 0.3), int(512 * 0.7) - obj_size)
    
    # Paste with alpha blending
    source.paste(fraud_img, (x, y), fraud_img)
    
    # Slight blur for blending
    result = source.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    mask_area = (obj_size * obj_size) / (512 * 512) * 100
    
    metadata = {
        "source_real": str(source_path),
        "object": fraud_object,
        "mask_area_percent": float(mask_area),
        "position": [x, y],
        "size": obj_size,
        "timestamp": datetime.now().isoformat(),
    }
    
    return result, metadata


def main():
    print("=" * 70)
    print("PLAN B: Direct Object Overlay (SIMPLE, WORKS)")
    print("=" * 70)
    
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    clean_pool = load_clean_pool()
    print(f"Clean pool: {len(clean_pool):,} images")
    
    total = sum(FRAUD_OBJECTS.values())
    print(f"Generating {total} fraud images...")
    
    # Load existing metadata for resume support
    overlay_meta_file = OUTPUT_DIR / "overlay_metadata.json"
    all_metadata = []
    if overlay_meta_file.exists():
        with open(overlay_meta_file, 'r') as f:
            all_metadata = json.load(f)
    
    # Count existing overlay images
    existing_count = sum(1 for f in OUTPUT_DIR.glob("overlay_*.png"))
    
    # Build flat list of all objects
    all_objects = []
    for obj_type, count in FRAUD_OBJECTS.items():
        all_objects.extend([obj_type] * count)
    random.shuffle(all_objects)
    total = len(all_objects)
    
    if existing_count >= total:
        print(f"Already have {existing_count} overlay images, target is {total}. Done.")
        return
    
    print(f"Resuming from {existing_count}/{total} overlay images...")
    
    import time
    batch_start = time.time()
    
    for img_idx in range(existing_count, total):
        fraud_object = all_objects[img_idx]
        source = random.choice(clean_pool)
        result, meta = add_fraud_to_image(source, fraud_object)
        
        if result is None:
            continue
        
        # Use overlay_ prefix to distinguish from inpainting
        filename = f"overlay_{img_idx:05d}.png"
        result.save(OUTPUT_DIR / filename)
        
        meta["image_path"] = f"ai_generated/class4_edited_real/{filename}"
        meta["label"] = 2
        meta["method"] = "simple_overlay"
        all_metadata.append(meta)
        
        if (img_idx + 1) % 100 == 0:
            done = img_idx - existing_count + 1
            elapsed = time.time() - batch_start
            rate = elapsed / done if done > 0 else 0.1
            eta_min = rate * (total - img_idx - 1) / 60
            print(f"  [{done}/{total - existing_count}] {rate:.2f}s/img | ETA: {eta_min:.1f}min")
            # Save checkpoint
            with open(overlay_meta_file, 'w') as f:
                json.dump(all_metadata, f, indent=2)
    
    # Final save
    with open(overlay_meta_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"[OK] COMPLETE: {len(all_metadata)} overlay fraud images generated")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Metadata: {overlay_meta_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
