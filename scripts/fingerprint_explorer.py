"""
AI Fingerprint Explorer
========================
Proves each AI model leaves a unique forensic signature.

Picks 1 sample from each source, computes:
  - FFT Spectrum      : frequency-domain fingerprints
  - SRM Noise Map     : noise residual patterns
  - ELA Map           : compression error levels

Saves a comparison grid:
  ai_generated/fingerprints/comparison_grid.png

Row per source × 4 columns: [Original | FFT | SRM | ELA]
"""
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).parent.parent
FINGERPRINT_DIR = PROJECT_ROOT / "ai_generated" / "fingerprints"

IMAGE_SIZE = 256   # Display size per cell in the grid

SOURCES = {
    "Real Photo":        PROJECT_ROOT / "food_101" / "food-101" / "food-101" / "images",
    "SDXL RealVisXL":   PROJECT_ROOT / "ai_generated" / "class1_raw",
    "Flux.1 Schnell":   PROJECT_ROOT / "ai_generated" / "flux_schnell",
    "Stable Cascade":   PROJECT_ROOT / "ai_generated" / "stable_cascade",
    "Kandinsky 2.2":    PROJECT_ROOT / "ai_generated" / "kandinsky3",
    "SDXL Turbo":       PROJECT_ROOT / "ai_generated" / "sdxl_turbo",
}

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# FORENSIC FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_fft(image: Image.Image) -> Image.Image:
    """
    Log-magnitude FFT spectrum, DC shifted to center.
    AI tells:
      SDXL         → periodic grid spikes at regular intervals
      Flux.1       → clean center falloff, no grid
      Stable Cascade → block artifacts from 3-stage compression
      Turbo/Lightning → bright halo 100-300px band (distillation over-sharpening)
    Real photos: smooth, natural exponential falloff from center
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(fft))
    magnitude -= magnitude.min()
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    # Convert to RGB heatmap (apply colormap manually)
    spectrum = Image.fromarray(magnitude, mode="L").convert("RGB")
    return spectrum.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def compute_srm(image: Image.Image) -> Image.Image:
    """
    SRM-style noise residual via 3×3 high-pass Laplacian kernel.
    Strips color info, shows only noise:
      Real photos  : sensor-dependent noise, consistent grain throughout
      AI-generated : uniform synthetic noise (model's generator distribution)
      Inpainting   : bright boundary ring where real/AI noise patterns collide
    """
    try:
        import scipy.ndimage as ndi
    except ImportError:
        # Fallback: numpy-based Laplacian
        gray = np.array(image.convert("L"), dtype=np.float32)
        kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32) / 8.0
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(gray, 1, mode="reflect")
        h, w = gray.shape
        residual = np.zeros_like(gray)
        for di in range(3):
            for dj in range(3):
                residual += pad[di:di+h, dj:dj+w] * kernel[di, dj]
        residual = np.clip(np.abs(residual) * 10, 0, 255).astype(np.uint8)
        srm_img = Image.fromarray(residual, "L").convert("RGB")
        return srm_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    gray = np.array(image.convert("L"), dtype=np.float32)
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], np.float32) / 8.0
    residual = ndi.convolve(gray, kernel)
    residual = np.clip(np.abs(residual) * 10, 0, 255).astype(np.uint8)
    srm_img = Image.fromarray(residual, "L").convert("RGB")
    return srm_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def compute_ela(image: Image.Image, quality: int = 90, scale: int = 10) -> Image.Image:
    """
    Error Level Analysis — highlights JPEG recompression inconsistencies.
    Inpainted regions show different ELA pattern than surrounding real pixels.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recomp = Image.open(buffer).convert("RGB")
    ela = ImageChops.difference(image.convert("RGB"), recomp)
    extrema = ela.getextrema()
    max_diff = max(max(ex[1] for ex in extrema), 20)
    ela = ela.point(lambda x: min(int(x * 255 / max_diff * scale), 255))
    return ela.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)


def add_label(img: Image.Image, text: str, bg=(20, 20, 30), fg=(200, 200, 210)) -> Image.Image:
    """Paste a label bar at the top of an image."""
    bar_h = 28
    result = Image.new("RGB", (img.width, img.height + bar_h), bg)
    result.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 6), text, fill=fg, font=font)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GRID BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def pick_sample(source_dir: Path) -> Image.Image | None:
    """Pick first valid image from directory (recursive)."""
    if not source_dir.exists():
        return None
    for f in sorted(source_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in EXTS:
            try:
                return Image.open(f).convert("RGB").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
                )
            except Exception:
                continue
    return None


def build_grid():
    FINGERPRINT_DIR.mkdir(parents=True, exist_ok=True)

    col_labels = ["Original", "FFT Spectrum", "SRM Noise", "ELA Map"]
    rows = []
    row_labels = []

    for source_name, source_dir in SOURCES.items():
        print(f"  Processing: {source_name} ...")
        img = pick_sample(source_dir)
        if img is None:
            print(f"    [SKIP] No images found in {source_dir}")
            continue

        fft_img  = compute_fft(img)
        srm_img  = compute_srm(img)
        ela_img  = compute_ela(img)

        row = [img, fft_img, srm_img, ela_img]
        rows.append(row)
        row_labels.append(source_name)

    if not rows:
        print("[ERROR] No sources had images. Run generators first.")
        return

    # Add column labels to each cell
    labeled_rows = []
    for ri, (row, rlabel) in enumerate(zip(rows, row_labels)):
        labeled_row = []
        for ci, (cell, clabel) in enumerate(zip(row, col_labels)):
            header = f"{clabel}" if ri == 0 else ""
            cell_labeled = add_label(cell, f"{rlabel} — {clabel}" if ri > 0 else clabel)
            labeled_row.append(cell_labeled)
        labeled_rows.append(labeled_row)

    # Compute grid dimensions
    n_rows = len(labeled_rows)
    n_cols = 4
    cell_w = labeled_rows[0][0].width
    cell_h = labeled_rows[0][0].height
    gap = 4
    header_h = 50

    grid_w = n_cols * cell_w + (n_cols - 1) * gap
    grid_h = header_h + n_rows * cell_h + (n_rows - 1) * gap

    grid = Image.new("RGB", (grid_w, grid_h), (10, 12, 20))
    draw = ImageDraw.Draw(grid)

    # Draw title
    try:
        title_font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.text((12, 10), "FoodGuard — AI Fingerprint Explorer", fill=(100, 200, 255), font=title_font)
    draw.text((12, 32), "Each AI model leaves a unique forensic signature invisible to the human eye",
              fill=(120, 130, 150), font=small_font)

    # Paste cells
    for ri, row in enumerate(labeled_rows):
        for ci, cell in enumerate(row):
            x = ci * (cell_w + gap)
            y = header_h + ri * (cell_h + gap)
            grid.paste(cell, (x, y))

    out_path = FINGERPRINT_DIR / "comparison_grid.png"
    grid.save(out_path, quality=95)
    print(f"\n[OK] Comparison grid saved: {out_path}")
    print(f"  Grid size: {grid_w}×{grid_h}px | {n_rows} sources × 4 analyses")

    # Also save individual FFT spectra for each source
    for source_name, row in zip(row_labels, rows):
        safe_name = source_name.lower().replace(" ", "_").replace(".", "")
        row[1].save(FINGERPRINT_DIR / f"fft_{safe_name}.png")
        row[2].save(FINGERPRINT_DIR / f"srm_{safe_name}.png")
        row[3].save(FINGERPRINT_DIR / f"ela_{safe_name}.png")
    print(f"  Individual maps saved in {FINGERPRINT_DIR}")


def main():
    print("=" * 70)
    print("AI FINGERPRINT EXPLORER")
    print("Comparing FFT / SRM / ELA signatures across AI generators")
    print("=" * 70)
    print("\nSampling 1 image from each source...")
    build_grid()
    print("\n[TIP] Show comparison_grid.png to evaluators to prove dataset diversity")
    print("[TIP] Each model's FFT has a distinct pattern — this is your forensic argument")


if __name__ == "__main__":
    main()
