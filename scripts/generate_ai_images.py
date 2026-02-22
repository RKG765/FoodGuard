"""
AI Food Image Generator -- 4-Class Forensic Strategy
=====================================================

Model: RealVisXL V4
Steps: 30
Guidance: 5.0
Resolution: 512x512
Mixed precision + xformers
"""

import os
import io
import csv
import json
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance

# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT
OUTPUT_DIR = DATA_ROOT / "ai_generated"
CHECKPOINT_FILE = OUTPUT_DIR / "generation_checkpoint.json"

IMAGE_SIZE = 512
INFERENCE_STEPS = 25
GUIDANCE_SCALE = 5.0

CLASS_TARGETS = {
    "class1_raw": 600,
    "class2_compressed": 600,
    "class3_degraded": 400,
    "class4_edited_real": 400,
}

REAL_IMAGE_DIRS = [
    DATA_ROOT / "food_101" / "food-101" / "images",
    DATA_ROOT / "food_image_dataset",
    DATA_ROOT / "indian_food_data",
]

# =============================================================================
# PROMPTS
# =============================================================================

FOOD_PROMPTS = [
    "butter chicken with naan bread",
    "biryani rice with raita",
    "samosa with green chutney",
    "paneer tikka masala",
    "cheeseburger with fries",
    "pepperoni pizza",
    "pani puri",
    "chocolate lava cake",
    "cheesecake slice",
    "grilled salmon with vegetables",
    "pasta carbonara",
    "ramen bowl with egg",
]

QUALITY_MODIFIERS = [
    "smartphone photo, natural lighting",
    "casual phone camera shot",
    "restaurant table photo",
    "indoor lighting",
    "overhead food photo",
    "slight motion blur",
]

NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, drawing, sketch, "
    "abstract, geometric pattern, kaleidoscope, tiled, collage, "
    "cgi, 3d render, surreal, fantasy, watermark, text, logo"
)

# =============================================================================
# PIPELINE SETUP
# =============================================================================

def setup_pipeline():
    import torch
    from diffusers import StableDiffusionXLPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading RealVisXL_V4.0...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0",
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe = pipe.to(device)

    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    return pipe, device

def generate_prompt():
    base = random.choice(FOOD_PROMPTS)
    quality = random.choice(QUALITY_MODIFIERS)
    return f"{base}, realistic food photography, {quality}, natural detail"

def generate_single_image(pipe, device):
    import torch

    prompt = generate_prompt()
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        generator=generator,
    ).images[0]

    return image, prompt, seed

# =============================================================================
# POST PROCESSING
# =============================================================================

def apply_jpeg_compression(image):
    resize_dim = random.choice([384, 448, 640])
    image = image.resize((resize_dim, resize_dim), Image.LANCZOS)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    quality = random.randint(40, 85)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    return Image.open(buffer).convert("RGB")

def apply_blur_noise(image):
    if random.random() < 0.7:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(1.5, 5.0), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def collect_real_images():
    real_paths = []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for base_dir in REAL_IMAGE_DIRS:
        if base_dir.exists():
            for f in base_dir.rglob("*"):
                if f.suffix.lower() in extensions:
                    real_paths.append(f)
    return real_paths

def apply_real_image_edit(image, all_real_paths):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    edits = random.sample(["crop_paste", "color_shift", "rotate"], k=2)

    for edit in edits:
        if edit == "crop_paste" and all_real_paths:
            donor_path = random.choice(all_real_paths)
            try:
                donor = Image.open(donor_path).convert("RGB")
                donor = donor.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

                pw = random.randint(60, 140)
                ph = random.randint(60, 140)
                sx = random.randint(0, IMAGE_SIZE - pw)
                sy = random.randint(0, IMAGE_SIZE - ph)
                patch = donor.crop((sx, sy, sx + pw, sy + ph))

                dx = random.randint(0, IMAGE_SIZE - pw)
                dy = random.randint(0, IMAGE_SIZE - ph)

                if random.random() < 0.5:
                    image.paste(patch, (dx, dy))
                else:
                    alpha = random.uniform(0.4, 0.7)
                    base = image.crop((dx, dy, dx + pw, dy + ph))
                    blended = Image.blend(base, patch, alpha)
                    image.paste(blended, (dx, dy))
            except:
                pass

        elif edit == "color_shift":
            arr = np.array(image, dtype=np.float32)
            channel = random.randint(0, 2)
            arr[:, :, channel] += random.uniform(-20, 20)
            arr = np.clip(arr, 0, 255)
            image = Image.fromarray(arr.astype(np.uint8))

        elif edit == "rotate":
            image = image.rotate(random.uniform(-6, 6), resample=Image.BICUBIC)

    return image

# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return json.load(open(CHECKPOINT_FILE, "r"))
    return {}

def save_checkpoint(checkpoint):
    json.dump(checkpoint, open(CHECKPOINT_FILE, "w"), indent=2)

def count_existing(class_name):
    d = OUTPUT_DIR / class_name
    if not d.exists():
        return 0
    return len(list(d.glob("*.*")))

# =============================================================================
# CLASS GENERATION
# =============================================================================

def generate_class(pipe, device, class_name, target, checkpoint):
    out_dir = OUTPUT_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = count_existing(class_name)
    if existing >= target:
        return

    meta = checkpoint.get(class_name, [])
    remaining = target - existing

    print(f"{class_name}: generating {remaining}")

    real_images = collect_real_images()

    for i in range(existing, target):
        if class_name == "class4_edited_real":
            src_path = random.choice(real_images)
            image = Image.open(src_path).convert("RGB")
            image = apply_real_image_edit(image, real_images)
        else:
            image, prompt, seed = generate_single_image(pipe, device)

            if class_name == "class2_compressed":
                image = apply_jpeg_compression(image)
            elif class_name == "class3_degraded":
                image = apply_blur_noise(image)

        filename = f"{class_name}_{i:05d}.png"
        image.save(out_dir / filename)

        meta.append({
            "image_path": f"ai_generated/{class_name}/{filename}",
            "timestamp": datetime.now().isoformat(),
        })

        if i % 25 == 0:
            checkpoint[class_name] = meta
            save_checkpoint(checkpoint)

    checkpoint[class_name] = meta
    save_checkpoint(checkpoint)

# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    checkpoint = load_checkpoint()

    pipe, device = setup_pipeline()

    for cls, target in CLASS_TARGETS.items():
        generate_class(pipe, device, cls, target, checkpoint)

    print("Generation complete.")

if __name__ == "__main__":
    main()
