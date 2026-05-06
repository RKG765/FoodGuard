"""
Kandinsky 2.2 Food Image Generator
=====================================
Architecture : CLIP Image Prior + UNet Decoder (multilingual dataset)
Why unique   : Trained on Russian/multilingual data — texture & noise patterns
               distinctly different from Western SDXL models.
               (Kandinsky 3.0 = 12B params = needs 24GB; using 2.2 instead)
VRAM         : ~8 GB (both pipelines on GPU, fp16)
Target       : 1500 images → ai_generated/kandinsky3/
Resume-safe  : counts existing kand_*.png
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

# Monkey-patch SSL for HuggingFace downloads
_orig_ctx = ssl.create_default_context
def _no_verify_ctx(*args, **kwargs):
    ctx = _orig_ctx(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
ssl.create_default_context = _no_verify_ctx

# Also patch requests Session to disable verify
_orig_session_init = requests.Session.__init__
def _patched_session_init(self, *args, **kwargs):
    _orig_session_init(self, *args, **kwargs)
    self.verify = False
requests.Session.__init__ = _patched_session_init

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "ai_generated" / "kandinsky3"
CHECKPOINT   = OUTPUT_DIR / "kand_checkpoint.json"

PRIOR_MODEL_ID   = "kandinsky-community/kandinsky-2-2-prior"
DECODER_MODEL_ID = "kandinsky-community/kandinsky-2-2-decoder"
TARGET = 1500; IMAGE_SIZE = 512
PRIOR_STEPS = 25; DECODER_STEPS = 50
GUIDANCE = 4.0

FOOD_PROMPTS = [
    "butter chicken with naan bread, realistic food photography",
    "biryani rice with raita, authentic plating",
    "samosa with green chutney, street food, overhead shot",
    "paneer tikka masala, restaurant dish, bokeh",
    "masala dosa with coconut chutney, south indian food",
    "chole bhature, casual photograph, daylight",
    "tandoori chicken, charred edges, mint chutney",
    "dal makhani, creamy texture, close-up",
    "gulab jamun dessert, syrup glistening",
    "cheeseburger with fries, food photography, sesame bun",
    "pepperoni pizza, wood-fired, cheese pull",
    "pasta carbonara, creamy sauce, parmesan",
    "grilled salmon with vegetables, fine dining",
    "caesar salad with croutons, fresh romaine",
    "steak with mashed potatoes, medium rare, cast iron",
    "risotto with mushrooms, creamy arborio",
    "ramen bowl with egg and nori, japanese broth",
    "sushi platter with wasabi, nigiri and maki",
    "pad thai with shrimp, wok-tossed noodles",
    "bibimbap in a stone pot, korean dolsot",
    "pho soup with beef, vietnamese restaurant",
    "chocolate lava cake, molten center",
    "cheesecake slice with berries, new york style",
    "tiramisu in a glass, cocoa dusted",
    "creme brulee, caramelized top, ramekin",
    "apple pie with ice cream, homestyle",
    "macarons, pastel colors, french patisserie",
    "waffles with syrup and butter, belgian",
    "cappuccino with latte art, ceramic cup",
    "nachos with cheese and guacamole",
    "falafel wrap with tahini, pita bread",
    "paella in a large pan, saffron rice, seafood",
]
QUALITY_MODS = [
    "natural lighting, realistic", "ambient restaurant light",
    "overhead food photography", "close-up, shallow depth of field",
    "warm kitchen light", "food blog style",
]
NEG = "cartoon, illustration, painting, sketch, abstract, cgi, watermark, text, logo"


def setup():
    from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
    print("Loading Kandinsky 2.2 Prior...")
    prior = KandinskyV22PriorPipeline.from_pretrained(
        PRIOR_MODEL_ID, torch_dtype=torch.float16).to("cuda")
    print("Loading Kandinsky 2.2 Decoder...")
    decoder = KandinskyV22Pipeline.from_pretrained(
        DECODER_MODEL_ID, torch_dtype=torch.float16).to("cuda")
    print("[OK] Kandinsky 2.2 loaded | Peak VRAM: ~8 GB")
    return prior, decoder


def gen(prior, decoder):
    prompt = f"{random.choice(FOOD_PROMPTS)}, {random.choice(QUALITY_MODS)}"
    seed = random.randint(0, 2**32 - 1)
    g = torch.Generator("cuda").manual_seed(seed)
    # Prior: text → image embeddings
    img_emb = prior(prompt=prompt, negative_prompt=NEG,
                    num_inference_steps=PRIOR_STEPS,
                    guidance_scale=GUIDANCE, generator=g)
    # Decoder: image embeddings → pixels
    img = decoder(image_embeds=img_emb.image_embeds,
                  negative_image_embeds=img_emb.negative_image_embeds,
                  height=IMAGE_SIZE, width=IMAGE_SIZE,
                  num_inference_steps=DECODER_STEPS,
                  guidance_scale=GUIDANCE, generator=g).images[0]
    return img, prompt, seed


def main():
    print("=" * 70)
    print("KANDINSKY 2.2 FOOD IMAGE GENERATOR")
    print("CLIP Prior + UNet Decoder | Multilingual training dataset")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_count = len(list(OUTPUT_DIR.glob("kand_*.png")))
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
        fname = f"kand_{i:05d}.png"
        img.save(OUTPUT_DIR / fname)
        meta.append({"image_path": f"ai_generated/kandinsky3/{fname}",
                      "model": "kandinsky-2-2", "prompt": prompt,
                      "seed": seed, "timestamp": datetime.now().isoformat()})
        done = i - existing_count + 1
        if done % 25 == 0:
            json.dump(meta, open(CHECKPOINT, "w"), indent=2)
            rate = (time.time() - t0) / done
            print(f"  [kandinsky] {done}/{TARGET-existing_count} | {rate:.1f}s/img | ETA: {rate*(TARGET-i-1)/60:.0f}min")
    json.dump(meta, open(CHECKPOINT, "w"), indent=2)
    print(f"[OK] DONE: {len(meta)} images in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
