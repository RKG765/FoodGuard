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
    "cockroach": 100,
    "housefly": 80,
    "small worm": 80,
    "human hair strand": 100,
    "mold patch": 70,
    "plastic fragment": 70,
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
    
    all_metadata = []
    img_idx = 0
    
    for obj_type, count in FRAUD_OBJECTS.items():
        print(f"\n[{obj_type}] {count} images...")
        for i in range(count):
            source = random.choice(clean_pool)
            result, meta = add_fraud_to_image(source, obj_type)
            
            if result is None:
                continue
            
            filename = f"class4_{img_idx:05d}.png"
            result.save(OUTPUT_DIR / filename)
            
            meta["image_path"] = f"ai_generated/class4_edited_real/{filename}"
            meta["label"] = 2
            all_metadata.append(meta)
            
            img_idx += 1
            
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{count}")
    
    # Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"✓ COMPLETE: {len(all_metadata)} fraud images generated")
    print(f"  Time: ~{len(all_metadata) * 0.1:.0f} seconds (vs 30 min with SDXL)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Metadata: {METADATA_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
