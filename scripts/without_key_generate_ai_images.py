"""
AI Food Image Generator for Fraud Detection (CORRECTED SAFE VERSION)
-------------------------------------------------------------------

• Stable Diffusion 3 (gated, approved)
• EXACT 70% phone-like realism
• EXACT 30% over-perfect
• Resolution: 512x512
• Resume-safe
• Crash-safe
• NO abstract / tiled / icon images
"""

import os
import csv
import json
import random
from pathlib import Path
from datetime import datetime

# =========================
# 🔐 HUGGING FACE TOKEN
# =========================
# PASTE YOUR READ TOKEN HERE
os.environ["HF_TOKEN"] = "enter your secret key here"

# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT
OUTPUT_DIR = DATA_ROOT / "ai_generated"
CHECKPOINT_FILE = OUTPUT_DIR / "generation_checkpoint.json"

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

IMAGES_PER_CATEGORY = {
    "indian": 333,
    "fast_food": 333,
    "street_food": 333,
    "desserts": 333,
    "beverages": 333,
    "continental": 333,
}

PHONE_RATIO = 0.7  # EXACT

# =========================
# PROMPTS
# =========================

PHONE_LIKE_QUALITY = [
    "smartphone photo, handheld",
    "slight motion blur, casual phone camera",
    "natural noise, imperfect focus",
    "uneven framing, real phone capture",
    "compressed image, social media upload",
]

OVER_PERFECT_QUALITY = [
    "professional food photography",
    "studio lighting, sharp focus",
    "advertisement style food photo",
]

LIGHTING_CONDITIONS = [
    "natural indoor lighting",
    "dim restaurant lighting",
    "overhead kitchen lighting",
    "phone camera flash",
]

CATEGORY_PROMPTS = {
    "indian": ["butter chicken with naan", "biryani rice", "samosa with chutney"],
    "fast_food": ["cheeseburger", "pepperoni pizza", "french fries"],
    "street_food": ["pani puri", "shawarma wrap", "tacos"],
    "desserts": ["chocolate lava cake", "cheesecake slice"],
    "beverages": ["latte coffee", "milkshake"],
    "continental": ["grilled salmon", "pasta carbonara"],
}

NEGATIVE_PROMPT = (
    "illustration, cartoon, painting, drawing, vector, icon, logo, "
    "pattern, abstract, collage, grid, tiled, repeated, "
    "3d render, cgi, unreal, anime, text, watermark, menu, ui"
)

# =========================
# DIFFUSERS
# =========================

import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusionXLPipeline


def setup_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        print("Loading Stable Diffusion 3 Medium...")
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=dtype,
            use_auth_token=True,
        )
        model_name = "sd3"
    except Exception as e:
        print("SD3 unavailable → Falling back to SDXL")
        print("Reason:", e)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if device == "cuda" else None,
            use_auth_token=True,
        )
        model_name = "sdxl"

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return pipe, device, model_name


# =========================
# CHECKPOINT
# =========================

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return []


def save_checkpoint(metadata):
    CHECKPOINT_FILE.write_text(json.dumps(metadata, indent=2))


def count_existing(category):
    d = OUTPUT_DIR / category
    return len(list(d.glob("*.png"))) if d.exists() else 0


# =========================
# GENERATION
# =========================

def generate_category(pipe, device, model, category, target, metadata):
    out_dir = OUTPUT_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = count_existing(category)
    if existing >= target:
        return

    phone_count = int(target * PHONE_RATIO)
    perfect_count = target - phone_count

    quality_plan = (["phone"] * phone_count + ["perfect"] * perfect_count)[existing:]
    random.shuffle(quality_plan)

    for i, qtype in enumerate(quality_plan, start=existing):
        base = random.choice(CATEGORY_PROMPTS[category])
        lighting = random.choice(LIGHTING_CONDITIONS)
        quality = random.choice(PHONE_LIKE_QUALITY if qtype == "phone" else OVER_PERFECT_QUALITY)

        prompt = (
            f"single realistic food photograph of {base}, "
            f"taken with a smartphone camera, "
            f"{lighting}, {quality}, "
            f"real food, real plate, natural imperfections"
        )

        seed = random.randint(0, 2**32 - 1)
        gen = torch.Generator(device=device).manual_seed(seed)

        image = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=28,
            guidance_scale=7.0,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            generator=gen,
        ).images[0]

        path = out_dir / f"ai_{category}_{i:05d}.png"
        image.save(path)

        metadata.append({
            "image_path": str(path.relative_to(DATA_ROOT)),
            "source_dataset": "stable_diffusion",
            "category": category,
            "label": "ai_generated",
            "quality_type": "phone_like" if qtype == "phone" else "over_perfect",
            "model": model,
            "seed": seed,
            "resolution": f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
        })

        if (i + 1) % 25 == 0:
            save_checkpoint(metadata)
            print(f"[{category}] {i+1}/{target}")


# =========================
# CSV
# =========================

def update_dataset_csv(metadata):
    csv_path = DATA_ROOT / "dataset_index.csv"

    rows = []
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [r for r in reader if r[3] != "ai_generated"]
    else:
        header = ["image_path", "source_dataset", "category", "label", "quality_type"]

    for m in metadata:
        rows.append([
            m["image_path"],
            m["source_dataset"],
            m["category"],
            m["label"],
            m["quality_type"],
        ])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# =========================
# MAIN
# =========================

def main():
    print("=== AI FOOD FRAUD DATASET GENERATION ===")
    OUTPUT_DIR.mkdir(exist_ok=True)

    metadata = load_checkpoint()
    pipe, device, model = setup_pipeline()

    for cat, n in IMAGES_PER_CATEGORY.items():
        generate_category(pipe, device, model, cat, n, metadata)

    save_checkpoint(metadata)
    (OUTPUT_DIR / "generation_metadata.json").write_text(json.dumps(metadata, indent=2))
    update_dataset_csv(metadata)

    print("DONE")
    print("Total images:", len(metadata))
    print("Model used:", model)


if __name__ == "__main__":
    main()
