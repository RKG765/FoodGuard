"""
SDXL Turbo + SDXL Lightning Food Image Generator
==================================================
Both are distilled SDXL variants — generates in 1-4 steps.

Why forensically unique:
  Distillation compression → over-sharpened high-freq band in FFT.
  "Bright halo" at 100-300px frequency range — unique to distilled models.
  Your detector learning this = can catch cheap AI fakes instantly.

Turbo  : guidance=0.0, 4 steps, 512px — heavy distillation artifacts
Lightning: guidance=0.0, 4 steps, 1024px → resize to 512
Combined : 750 Turbo + 750 Lightning = 1500 images

VRAM   : ~7-8 GB (fp16)
Output : ai_generated/sdxl_turbo/
Resume : counts turbo_*.png + light_*.png separately
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
OUTPUT_DIR   = PROJECT_ROOT / "ai_generated" / "sdxl_turbo"
CHECKPOINT   = OUTPUT_DIR / "sdxl_fast_checkpoint.json"

TURBO_MODEL_ID    = "stabilityai/sdxl-turbo"
LIGHTNING_REPO    = "ByteDance/SDXL-Lightning"
LIGHTNING_CKPT    = "sdxl_lightning_4step_unet.safetensors"
SDXL_BASE_ID      = "stabilityai/stable-diffusion-xl-base-1.0"

TARGET_EACH = 750   # 750 Turbo + 750 Lightning = 1500 total
IMAGE_SIZE  = 512

FOOD_PROMPTS = [
    "butter chicken with naan bread, realistic food photography",
    "biryani rice with raita, authentic plating",
    "samosa with green chutney, street food photo",
    "paneer tikka masala, restaurant dish",
    "masala dosa with coconut chutney",
    "cheeseburger with fries, fast food photography",
    "pepperoni pizza slice, wood-fired, cheese pull",
    "pasta carbonara, creamy sauce, parmesan",
    "grilled salmon with vegetables, fine dining",
    "ramen bowl with egg and nori, japanese broth",
    "sushi platter with wasabi, nigiri and maki",
    "pad thai with shrimp, wok-tossed noodles",
    "bibimbap in a stone pot, colorful vegetables",
    "chocolate lava cake, molten center",
    "cheesecake slice with berries, new york style",
    "creme brulee, caramelized top",
    "macarons, pastel colors, french patisserie",
    "waffles with syrup and butter, belgian",
    "cappuccino with latte art, ceramic cup",
    "paella in large pan, saffron rice, seafood",
    "steak with mashed potatoes, medium rare",
    "fish and chips, british pub style",
    "dim sum basket, har gow, chopsticks",
    "apple pie with ice cream, homestyle",
    "nachos with cheese and guacamole",
    "pho soup with beef, vietnamese herbs",
    "spring rolls with sweet chili sauce",
    "tiramisu in glass, cocoa dusted",
    "bruschetta with tomatoes, toasted bread",
    "chicken teriyaki with rice, sesame seeds",
]
QUALITY_MODS = [
    "natural lighting, realistic", "ambient restaurant light",
    "overhead food photo", "close-up, shallow depth",
    "warm kitchen light", "food blog style",
]
NEG = "cartoon, illustration, painting, sketch, abstract, cgi, watermark, text"


def setup_turbo():
    from diffusers import AutoPipelineForText2Image
    print("Loading SDXL-Turbo (fp16)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        TURBO_MODEL_ID, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    print("[OK] SDXL-Turbo loaded | VRAM: ~7 GB")
    return pipe


def setup_lightning():
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print("Loading SDXL-Lightning (4-step UNet)...")
    base = StableDiffusionXLPipeline.from_pretrained(
        SDXL_BASE_ID, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    # Lightning REQUIRES trailing timesteps — without this quality degrades badly
    base.scheduler = EulerDiscreteScheduler.from_config(
        base.scheduler.config, timestep_spacing="trailing"
    )
    print("  Downloading Lightning 4-step UNet checkpoint...")
    unet_path = hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT)
    base.unet.load_state_dict(load_file(unet_path, device="cuda"))
    print("[OK] SDXL-Lightning loaded | VRAM: ~8 GB")
    return base


def gen_turbo(pipe):
    prompt = f"{random.choice(FOOD_PROMPTS)}, {random.choice(QUALITY_MODS)}"
    seed = random.randint(0, 2**32 - 1)
    g = torch.Generator("cuda").manual_seed(seed)
    img = pipe(prompt=prompt, num_inference_steps=4,
               guidance_scale=0.0,           # CFG-free
               width=IMAGE_SIZE, height=IMAGE_SIZE, generator=g).images[0]
    return img, prompt, seed


def gen_lightning(pipe):
    prompt = f"{random.choice(FOOD_PROMPTS)}, {random.choice(QUALITY_MODS)}"
    seed = random.randint(0, 2**32 - 1)
    g = torch.Generator("cuda").manual_seed(seed)
    img = pipe(prompt=prompt, num_inference_steps=4,
               guidance_scale=0.0,           # CFG-free distilled
               width=1024, height=1024, generator=g).images[0]
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    return img, prompt, seed


def run_batch(gen_fn, model_name, prefix, target, meta, t0):
    existing = len(list(OUTPUT_DIR.glob(f"{prefix}_*.png")))
    if existing >= target:
        print(f"  [OK] {model_name}: already has {existing}/{target}. Skipping.")
        return meta
    print(f"\n  [{model_name}] Resuming from {existing}/{target}...")
    pipe_fn, pipeline = gen_fn
    for i in range(existing, target):
        try:
            img, prompt, seed = pipeline(pipe_fn)
        except Exception as e:
            print(f"    [WARN] {i}: {e}"); continue
        fname = f"{prefix}_{i:05d}.png"
        img.save(OUTPUT_DIR / fname)
        meta.append({"image_path": f"ai_generated/sdxl_turbo/{fname}",
                      "model": model_name, "prompt": prompt,
                      "seed": seed, "timestamp": datetime.now().isoformat()})
        done = i - existing + 1
        if done % 50 == 0:
            json.dump(meta, open(CHECKPOINT, "w"), indent=2)
            rate = (time.time() - t0) / max(done, 1)
            print(f"    [{prefix}] {done}/{target-existing} | {rate:.1f}s/img | ETA: {rate*(target-i-1)/60:.0f}min")
    return meta


def main():
    print("=" * 70)
    print("SDXL TURBO + LIGHTNING FOOD IMAGE GENERATOR")
    print("Distilled SDXL | 4 steps | CFG-free | ~1-2s per image")
    print("=" * 70)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(CHECKPOINT)) if CHECKPOINT.exists() else []
    t0 = time.time()

    # --- Phase 1: SDXL Turbo ---
    turbo_done = len(list(OUTPUT_DIR.glob("turbo_*.png")))
    if turbo_done < TARGET_EACH:
        pipe_turbo = setup_turbo()
        for i in range(turbo_done, TARGET_EACH):
            try:
                img, prompt, seed = gen_turbo(pipe_turbo)
            except Exception as e:
                print(f"  [WARN] turbo {i}: {e}"); continue
            fname = f"turbo_{i:05d}.png"
            img.save(OUTPUT_DIR / fname)
            meta.append({"image_path": f"ai_generated/sdxl_turbo/{fname}",
                          "model": "sdxl-turbo", "prompt": prompt,
                          "seed": seed, "timestamp": datetime.now().isoformat()})
            done = i - turbo_done + 1
            if done % 50 == 0:
                json.dump(meta, open(CHECKPOINT, "w"), indent=2)
                rate = (time.time() - t0) / max(done, 1)
                print(f"  [turbo] {done}/{TARGET_EACH-turbo_done} | {rate:.1f}s/img | ETA: {rate*(TARGET_EACH-i-1)/60:.0f}min")
        del pipe_turbo
        torch.cuda.empty_cache()
        print(f"  [OK] Turbo phase complete")
    else:
        print(f"  [OK] Turbo: already {turbo_done}/{TARGET_EACH}. Skipping.")

    # --- Phase 2: SDXL Lightning ---
    light_done = len(list(OUTPUT_DIR.glob("light_*.png")))
    if light_done < TARGET_EACH:
        pipe_light = setup_lightning()
        t1 = time.time()
        for i in range(light_done, TARGET_EACH):
            try:
                img, prompt, seed = gen_lightning(pipe_light)
            except Exception as e:
                print(f"  [WARN] lightning {i}: {e}"); continue
            fname = f"light_{i:05d}.png"
            img.save(OUTPUT_DIR / fname)
            meta.append({"image_path": f"ai_generated/sdxl_turbo/{fname}",
                          "model": "sdxl-lightning", "prompt": prompt,
                          "seed": seed, "timestamp": datetime.now().isoformat()})
            done = i - light_done + 1
            if done % 50 == 0:
                json.dump(meta, open(CHECKPOINT, "w"), indent=2)
                rate = (time.time() - t1) / max(done, 1)
                print(f"  [lightning] {done}/{TARGET_EACH-light_done} | {rate:.1f}s/img | ETA: {rate*(TARGET_EACH-i-1)/60:.0f}min")
        del pipe_light
        torch.cuda.empty_cache()
        print(f"  [OK] Lightning phase complete")
    else:
        print(f"  [OK] Lightning: already {light_done}/{TARGET_EACH}. Skipping.")

    json.dump(meta, open(CHECKPOINT, "w"), indent=2)
    total = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\n[OK] COMPLETE: {total} images in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
