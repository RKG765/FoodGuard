"""
Stable Cascade Food Image Generator
3-Stage Würstchen architecture — distinct FFT block artifacts vs SDXL.
Target: 1500 images → ai_generated/stable_cascade/
Resume-safe: counts existing cascade_*.png
"""
import json, os, random, ssl, time
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

_orig_ctx = ssl.create_default_context
def _no_verify_ctx(*args, **kwargs):
    ctx = _orig_ctx(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl.create_default_context = _no_verify_ctx

_orig_session_init = requests.Session.__init__
def _patched_session_init(self, *args, **kwargs):
    _orig_session_init(self, *args, **kwargs)
    self.verify = False
requests.Session.__init__ = _patched_session_init

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "ai_generated" / "stable_cascade"
CHECKPOINT   = OUTPUT_DIR / "cascade_checkpoint.json"

PRIOR_MODEL_ID   = "stabilityai/stable-cascade-prior"
DECODER_MODEL_ID = "stabilityai/stable-cascade"
TARGET = 1500; IMAGE_SIZE = 512
PRIOR_STEPS = 20; DECODER_STEPS = 10
GUIDANCE_PRIOR = 4.0; GUIDANCE_DEC = 0.0

FOOD_PROMPTS = [
    "butter chicken with naan bread, realistic food photography, natural lighting",
    "biryani rice with raita, authentic plating, warm tones",
    "samosa with green chutney, street food photography, overhead shot",
    "paneer tikka masala, restaurant dish, soft bokeh",
    "masala dosa with coconut chutney, south indian food",
    "cheeseburger with fries, fast food photography, sesame bun",
    "pepperoni pizza slice, wood-fired, cheese pull",
    "fish and chips on newspaper, british pub style",
    "pasta carbonara, creamy sauce, parmesan shavings",
    "grilled salmon with vegetables, fine dining, herb garnish",
    "caesar salad with croutons, fresh romaine",
    "steak with mashed potatoes, medium rare, cast iron",
    "ramen bowl with egg and nori, japanese broth, steam",
    "sushi platter with wasabi, nigiri and maki",
    "pad thai with shrimp, wok-tossed noodles, lime wedge",
    "bibimbap in a stone pot, korean dolsot, vegetables",
    "pho soup with beef, vietnamese restaurant, herbs",
    "chocolate lava cake, molten center, powdered sugar",
    "cheesecake slice with berries, new york style",
    "tiramisu in a glass, ladyfinger layers, cocoa dusted",
    "creme brulee with caramelized top, ramekin",
    "apple pie with ice cream, homestyle, flaky crust",
    "macarons on a plate, pastel colors, french patisserie",
    "waffles with syrup and butter, belgian style",
    "cappuccino with latte art, ceramic cup, cafe",
    "bruschetta with tomatoes, toasted bread, fresh basil",
    "nachos with cheese and guacamole, loaded snack",
    "falafel wrap with tahini, street food, pita bread",
    "paella in a large pan, saffron rice, seafood",
    "dim sum steamer basket, har gow, chopsticks",
]
QUALITY_MODS = [
    "smartphone photo, natural lighting", "restaurant table, ambient light",
    "overhead shot, minimal props", "close-up macro, shallow depth",
    "casual home cooking, warm light", "food blog style, bright and airy",
]
NEG = "cartoon, illustration, painting, sketch, abstract, cgi, watermark, text"


def setup():
    from diffusers import StableCascadePriorPipeline, StableCascadeDecoderPipeline
    print("Loading Stable Cascade Prior (Stage C)...")
    prior = StableCascadePriorPipeline.from_pretrained(PRIOR_MODEL_ID, torch_dtype=torch.bfloat16)
    prior.enable_model_cpu_offload()
    print("Loading Stable Cascade Decoder (Stage B+A)...")
    decoder = StableCascadeDecoderPipeline.from_pretrained(DECODER_MODEL_ID, torch_dtype=torch.float16)
    decoder.enable_model_cpu_offload()
    print("[OK] Both pipelines loaded | Peak VRAM: ~10 GB")
    return prior, decoder


def gen(prior, decoder):
    prompt = f"{random.choice(FOOD_PROMPTS)}, {random.choice(QUALITY_MODS)}"
    seed = random.randint(0, 2**32 - 1)
    g = torch.Generator("cpu").manual_seed(seed)
    out = prior(prompt=prompt, negative_prompt=NEG, height=IMAGE_SIZE, width=IMAGE_SIZE,
                num_inference_steps=PRIOR_STEPS, guidance_scale=GUIDANCE_PRIOR, generator=g)
    img = decoder(image_embeddings=out.image_embeddings.to(torch.float16),
                  prompt=prompt, negative_prompt=NEG, num_inference_steps=DECODER_STEPS,
                  guidance_scale=GUIDANCE_DEC, generator=g).images[0]
    return img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS), prompt, seed


def main():
    print("=" * 70)
    print("STABLE CASCADE FOOD IMAGE GENERATOR")
    print("3-Stage Würstchen | Prior:20 steps | Decoder:10 steps")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_count = len(list(OUTPUT_DIR.glob("cascade_*.png")))
    if existing_count >= TARGET:
        print(f"[OK] Already have {existing_count} images. Done."); return
    print(f"Resuming from {existing_count}/{TARGET}...")
    meta = json.load(open(CHECKPOINT)) if CHECKPOINT.exists() else []
    prior, decoder = setup()
    t0 = time.time()
    for i in range(existing_count, TARGET):
        try:
            img, prompt, seed = gen(prior, decoder)
        except Exception as e:
            print(f"  [WARN] {i}: {e}"); continue
        fname = f"cascade_{i:05d}.png"
        img.save(OUTPUT_DIR / fname)
        meta.append({"image_path": f"ai_generated/stable_cascade/{fname}", "model": "stable-cascade",
                      "prompt": prompt, "seed": seed, "timestamp": datetime.now().isoformat()})
        done = i - existing_count + 1
        if done % 20 == 0:
            json.dump(meta, open(CHECKPOINT, "w"), indent=2)
            rate = (time.time() - t0) / done
            print(f"  [cascade] {done}/{TARGET-existing_count} | {rate:.1f}s/img | ETA: {rate*(TARGET-i-1)/60:.0f}min")
    json.dump(meta, open(CHECKPOINT, "w"), indent=2)
    print(f"[OK] DONE: {len(meta)} images in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
