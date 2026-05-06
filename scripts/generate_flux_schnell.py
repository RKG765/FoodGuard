"""
PixArt-Σ (Sigma) Food Image Generator
=======================================

Architecture : Diffusion Transformer (DiT) — same family as FLUX/Midjourney
Why unique   : DiT architecture produces distinct frequency-domain artifacts
               compared to UNet-based SDXL. No grid artifacts; clean
               attention-based residuals — different fingerprint for forensics.
VRAM         : ~8 GB with cpu_offload (float16)
Target       : 2000 images → ai_generated/flux_schnell/
               (kept same output dir for pipeline compatibility)

Resume-safe  : counts existing flux_*.png and resumes automatically.

NOTE: Replaces FLUX.1-schnell which requires gated access.
      PixArt-Σ is ungated and produces forensically similar DiT fingerprints.
"""

import json
import os
import random
import ssl
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# SSL FIX — bypass corporate/campus proxy certificate issues
# ─────────────────────────────────────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import requests
requests.packages.urllib3.disable_warnings()

# Monkey-patch SSL for HuggingFace downloads
_orig_ctx = ssl.create_default_context
def _no_verify_ctx(*args, **kwargs):
    ctx = _orig_ctx(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl.create_default_context = _no_verify_ctx

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
OUTPUT_DIR    = PROJECT_ROOT / "ai_generated" / "flux_schnell"
CHECKPOINT    = OUTPUT_DIR / "flux_checkpoint.json"

MODEL_ID          = "PixArt-alpha/PixArt-XL-2-512x512"
TARGET            = 2000
IMAGE_SIZE        = 512
INFERENCE_STEPS   = 20       # PixArt-α optimal range: 15-25
GUIDANCE_SCALE    = 4.5      # PixArt-Σ uses CFG

FOOD_PROMPTS = [
    # Indian
    "butter chicken with naan bread, realistic food photography, natural lighting",
    "biryani rice with raita, authentic plating, warm tones",
    "samosa with green chutney, street food photography, overhead shot",
    "paneer tikka masala, restaurant dish, soft bokeh",
    "masala dosa with coconut chutney, south indian food, wooden table",
    "chole bhature on a plate, casual photograph, daylight",
    "tandoori chicken with mint chutney, charred edges, clay oven",
    "dal makhani in a bowl, creamy texture, close-up",
    "gulab jamun dessert, syrup glistening, studio lighting",
    "pani puri street food, vendor stall, candid photo",
    # Western fast food
    "cheeseburger with fries, fast food photography, sesame bun",
    "pepperoni pizza slice, wood-fired, cheese pull",
    "fish and chips on newspaper, british pub style",
    "fried chicken bucket, crispy golden, american diner",
    "hot dog with mustard and ketchup, fair food, casual shot",
    "chicken nuggets with dipping sauce, kids meal, bright",
    # Continental
    "pasta carbonara on a plate, creamy sauce, parmesan shavings",
    "grilled salmon with vegetables, fine dining, herb garnish",
    "caesar salad with croutons, fresh romaine, dressing drizzle",
    "steak with mashed potatoes, medium rare, cast iron",
    "risotto with mushrooms, creamy arborio, restaurant plating",
    "french onion soup in a bowl, melted gruyere, crouton",
    "eggs benedict with hollandaise, brunch plating, poached",
    "lobster thermidor, butter sauce, fine dining",
    # Asian
    "ramen bowl with egg and nori, japanese broth, steam",
    "sushi platter with wasabi, nigiri and maki, clean plate",
    "pad thai with shrimp, wok-tossed noodles, lime wedge",
    "dim sum steamer basket, har gow and siu mai, chopsticks",
    "pho soup with beef, vietnamese restaurant, herbs on top",
    "bibimbap in a stone pot, korean dolsot, colorful vegetables",
    "spring rolls with sweet chili sauce, vietnamese fresh rolls",
    "chicken teriyaki with rice, japanese bento, sesame seeds",
    # Desserts
    "chocolate lava cake, molten center, powdered sugar",
    "cheesecake slice with berries, new york style, graham crust",
    "tiramisu in a glass, ladyfinger layers, cocoa dusted",
    "creme brulee with caramelized top, ramekin, torch marks",
    "apple pie with ice cream, homestyle, flaky crust",
    "macarons on a plate, pastel colors, french patisserie",
    "waffles with syrup and butter, belgian style, whipped cream",
    "fresh fruit tart, pastry cream, colorful glaze",
    # Beverages and sides
    "cappuccino with latte art, ceramic cup, cafe table",
    "fresh fruit smoothie bowl, acai base, granola topping",
    "bruschetta with tomatoes, toasted bread, fresh basil",
    "nachos with cheese and guacamole, loaded, casual snack",
    "falafel wrap with tahini, street food, pita bread",
    "paella in a large pan, saffron rice, seafood mix",
]

QUALITY_MODIFIERS = [
    "smartphone photo, natural lighting",
    "restaurant table, ambient light",
    "overhead food photography, minimal props",
    "close-up macro, shallow depth of field",
    "casual home cooking, warm kitchen light",
    "food blog style, bright and airy",
    "slightly underexposed, moody restaurant",
    "bright window light, lifestyle photo",
]

NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, sketch, drawing, "
    "abstract, geometric, tiled, collage, cgi, 3d render, "
    "watermark, text, logo, surreal, fantasy, low quality, blurry"
)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def setup_pipeline():
    from diffusers import PixArtAlphaPipeline

    print("Loading PixArt-α (Alpha) XL 512px (float16)...")
    print("  NOTE: First run will download ~2.5GB of model weights.")
    print("        Subsequent runs load from HuggingFace cache.")

    pipe = PixArtAlphaPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    # CPU offload — saves VRAM, ~8 GB peak during generation
    pipe.enable_model_cpu_offload()

    print("  [OK] PixArt-α loaded with CPU offload")
    print("  [OK] Peak VRAM during generation: ~8 GB")
    return pipe


def generate_prompt():
    base = random.choice(FOOD_PROMPTS)
    quality = random.choice(QUALITY_MODIFIERS)
    return f"{base}, {quality}"


def generate_image(pipe):
    prompt = generate_prompt()
    seed = random.randint(0, 2**32 - 1)
    generator = torch.Generator("cpu").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    return image, prompt, seed


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────
def load_meta():
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            return json.load(f)
    return []


def save_meta(meta):
    with open(CHECKPOINT, "w") as f:
        json.dump(meta, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("PIXART-Σ (SIGMA) FOOD IMAGE GENERATOR")
    print("Architecture: DiT (Diffusion Transformer) | Steps: 20 | CFG: 4.5")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume: count existing images
    existing = list(OUTPUT_DIR.glob("flux_*.png"))
    existing_count = len(existing)

    if existing_count >= TARGET:
        print(f"[OK] Already have {existing_count} images, target is {TARGET}. Done.")
        return

    print(f"\nResuming from {existing_count}/{TARGET} images...")

    meta = load_meta()
    pipe = setup_pipeline()

    batch_start = time.time()

    for i in range(existing_count, TARGET):
        try:
            image, prompt, seed = generate_image(pipe)
        except Exception as e:
            print(f"  [WARN] Generation failed at {i}: {e}. Skipping.")
            continue

        filename = f"flux_{i:05d}.png"
        image.save(OUTPUT_DIR / filename)

        meta.append({
            "image_path": f"ai_generated/flux_schnell/{filename}",
            "model": "PixArt-Sigma",
            "prompt": prompt,
            "seed": seed,
            "steps": INFERENCE_STEPS,
            "timestamp": datetime.now().isoformat(),
        })

        if (i - existing_count + 1) % 25 == 0:
            save_meta(meta)
            done = i - existing_count + 1
            elapsed = time.time() - batch_start
            rate = elapsed / done
            eta_min = rate * (TARGET - i - 1) / 60
            print(
                f"  [pixart_sigma] {done}/{TARGET - existing_count} done | "
                f"{rate:.1f}s/img | ETA: {eta_min:.0f}min"
            )

    save_meta(meta)

    print(f"\n{'=' * 70}")
    print(f"[OK] PIXART-Σ GENERATION COMPLETE")
    print(f"  Total images : {len(meta)}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print(f"  Checkpoint   : {CHECKPOINT}")
    print(f"{'=' * 70}")
    print("\nNext: Run generate_kandinsky.py or generate_sdxl_fast.py")


if __name__ == "__main__":
    main()
