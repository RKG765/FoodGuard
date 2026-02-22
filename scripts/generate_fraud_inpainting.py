"""
Semantic Fraud Image Generation - Production Pipeline
======================================================

RealVisXL V4 Inpainting for Food Fraud Detection

Strategy:
- Load real image from clean pool (186K - excluded 5K pristine)
- Create small irregular mask (5-12% area)
- Inpaint fraud object (cockroach, hair, mold, etc.)
- Save with full metadata tracking

Target: 500 high-quality fraud images
"""

import json
import random
import csv
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import StableDiffusionXLInpaintPipeline

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT
CSV_PATH = PROJECT_ROOT / "dataset_index.csv"
OUTPUT_DIR = DATA_ROOT / "ai_generated" / "class4_edited_real"
METADATA_FILE = OUTPUT_DIR / "fraud_metadata.json"

# RealVisXL V4.0 (as user specified)
MODEL_ID = "SG161222/RealVisXL_V4.0"
IMAGE_SIZE = 512
INFERENCE_STEPS = 26
GUIDANCE_SCALE = 4.5  # Very low to prevent object dominance, allow natural blending

# Fraud objects with natural distribution (expanded per user request)
FRAUD_OBJECTS = {
    "cockroach": 60,
    "housefly": 50,
    "mosquito": 50,
    "bee": 40,
    "ant": 40,
    "small worm": 40,
    "human hair strand": 60,
    "mold patch": 50,
    "plastic fragment": 60,
    "piece of paper": 50,
    "metal shard": 50,
    "dead insect": 50,
}

NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, drawing, unrealistic, blurry, "
    "distorted, extra objects, duplicate, watermark, text, "
    "clean surface, spotless food, no contamination, pristine dish, "  # Against clean food
    "glossy, shiny, macro photography, studio lighting, 3d render, "  # Against glossy objects
    "perfect focus, sharp detail, centered composition, professional product shot"  # Against dominance
)

SEED = 42

# =============================================================================
# CLEAN POOL EXTRACTION
# =============================================================================

def load_clean_pool():
    """Load clean pool: all real images EXCEPT the 5K pristine used in CSV."""
    print("Loading clean pool (186K real images excluding 5K pristine)...")
    
    #  Extract pristine 5K from CSV
    pristine_5k = set()
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['label']) == 0:  # Real class
                full_path = PROJECT_ROOT / row['file_path']
                pristine_5k.add(str(full_path.resolve()))
    
    print(f"  Pristine real in CSV: {len(pristine_5k)}")
    
    # Collect all real from master pool
    all_real_dirs = [
        DATA_ROOT / "food_101" / "food-101" / "food-101" / "images",
        DATA_ROOT / "food_image_dataset" / "data" / "UECFOOD256 2",
        DATA_ROOT / "food_image_dataset" / "data" / "aicrowd_food_recognition",
        DATA_ROOT / "indian_food_data" / "image_for _cuisines" / "data",
    ]
    
    all_real = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for source_dir in all_real_dirs:
        if source_dir.exists():
            for f in source_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in extensions:
                    all_real.append(str(f.resolve()))
    
    print(f"  Total real pool: {len(all_real):,}")
    
    # Clean pool = all - pristine 5K
    clean_pool = [p for p in all_real if p not in pristine_5k]
    
    print(f"  Clean pool (for editing): {len(clean_pool):,}")
    print(f"  ✓ Zero overlap with pristine guaranteed")
    
    return clean_pool

# =============================================================================
# MASK GENERATION
# =============================================================================

def create_irregular_mask(width=512, height=512):
    """
    Create irregular blob mask covering PRECISE 8-12% of image.
    
    FIX 2: Center-biased placement (30-70% of width/height)
    FIX 3: Minimum 8% (not 5%) for better SDXL response
    """
    from PIL import ImageFilter
    
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # PRECISE area calculation: 2-4% (very small, Gemini-like subtle insertion)
    target_percent = random.uniform(0.02, 0.04)
    target_area = target_percent * (width * height)
    radius = int(np.sqrt(target_area / np.pi))
    
    # FIX 2: Center-biased position (30-70% of image, avoid corners/edges)
    center_min = int(width * 0.30)
    center_max = int(width * 0.70)
    x = random.randint(max(radius, center_min), min(width - radius, center_max))
    y = random.randint(max(radius, center_min), min(height - radius, center_max))
    
    # Draw 2-3 overlapping ellipses for irregular shape
    num_blobs = random.randint(2, 3)
    for _ in range(num_blobs):
        blob_r = int(radius * random.uniform(0.7, 1.0))
        offset_x = random.randint(-radius//2, radius//2)
        offset_y = random.randint(-radius//2, radius//2)
        blob_x = max(blob_r, min(width - blob_r, x + offset_x))
        blob_y = max(blob_r, min(height - blob_r, y + offset_y))
        draw.ellipse(
            [blob_x - blob_r, blob_y - blob_r, blob_x + blob_r, blob_y + blob_r],
            fill=255
        )
    
    # Blur edges for natural blending
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    return mask

# =============================================================================
# INPAINTING PIPELINE
# =============================================================================

def setup_inpainting_pipeline():
    """Load RealVisXL V4 inpainting pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"\nLoading RealVisXL V4 Inpainting Pipeline...")
    print(f"  Device: {device}")
    
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None,
    )
    pipe = pipe.to(device)
    
    # Memory optimizations
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("  ✓ xformers enabled")
    except:
        pass
    
    pipe.enable_attention_slicing()
    # Fixed deprecated method
    pipe.vae.enable_slicing()
    
    return pipe, device


def generate_fraud_image(pipe, device, source_path, fraud_object):
    """
    Generate one fraud image via inpainting.
    
    Returns:
        (output_image, metadata_dict)
    """
    # Load source
    try:
        source_img = Image.open(source_path).convert("RGB")
    except:
        return None, None
    
    source_img = source_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    
    # Create mask
    mask = create_irregular_mask(IMAGE_SIZE, IMAGE_SIZE)
    mask_percent = (np.array(mask) > 0).sum() / (IMAGE_SIZE * IMAGE_SIZE) * 100
    
    # GEMINI-STYLE EDITING: Scene preservation, scale matching, detail reduction
    # Food is PRIMARY subject, object is SECONDARY at correct scale with reduced detail
    templates = [
        f"Food photo as main subject, with a small real-world-sized {fraud_object} matching the scene blur level, same detail resolution as the food, weak contact shadow, casual lighting",
        f"Photograph of a meal where the food is the focus, containing a tiny naturally-sized {fraud_object} with detail level matching surrounding scene, slightly soft focus like the background, ambient light",
        f"Natural food image with correct focus on the dish, small {fraud_object} at realistic proportions, matching scene sharpness and blur, reduced detail like background elements, subtle shadow",
        f"Casual photo of food plate as primary subject, tiny {fraud_object} matching real-world scale relative to dish, same blur and detail level as adjacent food, weak lighting integration",
        f"Amateur food shot focusing on the meal, small real-sized {fraud_object} with detail resolution matching the scene depth, natural blur consistency, subtle presence",
    ]
    prompt = random.choice(templates)
    
    # Generate
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=source_img,
            mask_image=mask,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            strength=0.99,  # CRITICAL: controls how much to modify masked area
            generator=generator,
        )
    
    result_img = output.images[0]
    
    # Metadata
    metadata = {
        "source_real": str(source_path),
        "object": fraud_object,
        "mask_area_percent": float(mask_percent),
        "prompt": prompt,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    
    return result_img, metadata

# =============================================================================
# MAIN GENERATION
# =============================================================================

def main():
    print("=" * 70)
    print("SEMANTIC FRAUD IMAGE GENERATION")
    print("RealVisXL V4 Inpainting Pipeline")
    print("=" * 70)
    
    # Setup
    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load clean pool
    clean_pool = load_clean_pool()
    
    # Setup pipeline
    pipe, device = setup_inpainting_pipeline()
    
    # Generate distribution
    total_images = sum(FRAUD_OBJECTS.values())
    print(f"\n{'=' * 70}")
    print(f"TARGET: {total_images} fraud images")
    print(f"{'=' * 70}")
    for obj, count in FRAUD_OBJECTS.items():
        print(f"  {obj:20s}: {count:3d} images")
    
    # Generate
    print(f"\n{'=' * 70}")
    print("GENERATING...")
    print(f"{'=' * 70}")
    
    all_metadata = []
    img_idx = 0
    
    # RANDOMIZE object selection instead of sequential generation
    # Create weighted list of all objects
    all_objects = []
    for obj_type, count in FRAUD_OBJECTS.items():
        all_objects.extend([obj_type] * count)
    
    random.shuffle(all_objects)  # Randomize order
    total = len(all_objects)
    
    print(f"Generating {total} images with randomized fraud objects...")
    
    for img_idx, fraud_object in enumerate(all_objects):
        # Random source from clean pool
        source_path = random.choice(clean_pool)
        
        # Generate
        result_img, metadata = generate_fraud_image(
            pipe, device, source_path, fraud_object
        )
        
        if result_img is None:
            continue
        
        # Save
        filename = f"class4_{img_idx:05d}.png"
        save_path = OUTPUT_DIR / filename
        result_img.save(save_path)
        
        # Update metadata
        metadata["image_path"] = f"ai_generated/class4_edited_real/{filename}"
        metadata["label"] = 2
        all_metadata.append(metadata)
        
        if (img_idx + 1) % 20 == 0:
            print(f"    Progress: {img_idx + 1} / {total}")
    
    # Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("✓ GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total generated: {len(all_metadata)} images")
    print(f"Metadata saved: {METADATA_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nNext: Update dataset_index.csv and rebuild dataset")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
