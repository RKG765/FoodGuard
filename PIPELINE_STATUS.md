# FoodGuard Pipeline Status
**Last Updated:** 2026-04-29 22:35 IST

## Current Dataset Counts
| Directory | Count | Target | Model | Status |
|---|---|---|---|---|
| `ai_generated/class1_raw` | 5,000 | 5,000 | RealVisXL V4.0 | ✅ Done |
| `ai_generated/class2_compressed` | 5,000 | 5,000 | RealVisXL V4.0 | ✅ Done |
| `ai_generated/class3_degraded` | 2,500 | 2,500 | RealVisXL V4.0 | ✅ Done |
| `ai_generated/class4_edited_real` | 5,501 | 2,500+ | Overlay fraud | ✅ Done |
| `ai_generated/flux_schnell` | 0 | 2,000 | **PixArt-Σ** (DiT) | ⏳ Run `generate_flux_schnell.py` |
| `ai_generated/kandinsky3` | 38 | 1,500 | Kandinsky 2.2 | ⏳ Run `generate_kandinsky.py` (resumes from 38) |
| `ai_generated/sdxl_turbo` | 646 | 1,500 | SDXL Turbo+Lightning | ⏳ Run `generate_sdxl_fast.py` (resumes from 646) |
| `ai_generated/stable_cascade` | 0 | 1,500 | Stable Cascade | ⏳ Run `generate_stable_cascade.py` |
| `ai_generated/indian` | 68 | — | SD3 (old) | ✅ Legacy |

## Execution Order (Run These Locally)

### Step 1: Generate Diversity AI Images (run in any order, all resume-safe)
```powershell
cd E:\BML\Semester-VI\Prj-3\scripts

# PixArt-Σ (DiT architecture, ~2.5GB download, ~8GB VRAM)
python generate_flux_schnell.py

# Kandinsky 2.2 (CLIP-Prior + UNet, prior cached, decoder cached after first run, ~8GB VRAM)
python generate_kandinsky.py

# SDXL Turbo + Lightning (resumes from 646/1500, ~7-8GB VRAM)
python generate_sdxl_fast.py

# Stable Cascade (3-stage Würstchen, ~10GB VRAM)
python generate_stable_cascade.py
```

### Step 2: Organize into 4-Class Folder Structure
```powershell
python scripts/organize_4class_dataset.py
```
Creates `dataset_4class/{train,val,test}/{real,perfect_ai,compressed_ai,edited_ai}/`

### Step 3: Train the Detector
```powershell
python train_4class_detector.py
```
- EfficientNet-B3, 30 epochs, AMP, early stopping
- Outputs: `checkpoints/food_detector/food_ai_detector.pth`

### Step 4: Run the App
```powershell
streamlit run app.py
```

## Bugs Fixed (This Session)
1. **FLUX.1-schnell 401 Unauthorized** → Replaced with PixArt-Σ (ungated, same DiT architecture)
2. **Kandinsky SSL_CERTIFICATE_VERIFY_FAILED** → Added SSL bypass (env vars + monkey-patch)
3. **SSL bypass added** to `generate_sdxl_fast.py` and `generate_stable_cascade.py`
4. **train_4class_detector.py FPR bug** → `evaluate_fpr()` was using `real_idx=0` but real is at index 3 (ImageFolder alphabetical). Fixed to use `REAL_CLASS_INDEX`. Same fix in `calibrate_threshold()`.
5. **build_detector_csv.py** → Added multi-model diversity dirs (PixArt, Kandinsky, SDXL-Turbo, Stable Cascade) so CSV includes all model fingerprints

## Architecture Overview
```
Real Images (Food-101 + UECFOOD256 + Aicrowd + Indian-Food)
    ↓
AI Images (RealVisXL + PixArt-Σ + Kandinsky + SDXL-Turbo/Lightning + Stable Cascade)
    ↓
organize_4class_dataset.py → dataset_4class/{train,val,test}/
    ↓
train_4class_detector.py → EfficientNet-B3 → checkpoints/
    ↓
app.py (Streamlit) / inference.py (CLI)
```

## Key Technical Notes
- **ImageFolder alphabetical order**: compressed_ai=0, edited_ai=1, perfect_ai=2, real=3
- **Class weights**: `[2.0, 1.0, 1.0, 1.0]` — real gets 2x weight to reduce FPR
- **Threshold calibration**: Finds P(real) threshold for ≤5% FPR on validation set
- **SSL bypass**: All generator scripts have SSL certificate bypass for campus proxy
- **All generators are resume-safe**: They count existing images and continue from where they stopped
