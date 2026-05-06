"""
FoodGuard — Streamlit Frontend
================================
Premium dark-themed UI for AI-generated food image detection.

Run:
    streamlit run app.py
"""

import io
import json
import time
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageChops, ImageDraw, ImageFilter
import timm
import numpy as np
from scipy.ndimage import label as scipy_label


# ──────────────────────────────────────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FoodGuard — AI Food Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — Premium Dark Theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Hero Section ── */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff 0%, #7b2ff7 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #8b95a5;
        font-weight: 300;
        letter-spacing: 0.02em;
    }

    /* ── Upload Area ── */
    .upload-zone {
        border: 2px dashed rgba(123, 47, 247, 0.4);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(123, 47, 247, 0.05);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    .upload-zone:hover {
        border-color: rgba(123, 47, 247, 0.8);
        background: rgba(123, 47, 247, 0.1);
    }

    /* ── Result Cards ── */
    .result-card {
        background: linear-gradient(145deg, rgba(30, 32, 44, 0.95), rgba(20, 22, 34, 0.95));
        border-radius: 16px;
        padding: 1.75rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .result-card-real {
        border-left: 4px solid #00e676;
    }
    .result-card-fake {
        border-left: 4px solid #ff1744;
    }

    /* ── Verdict ── */
    .verdict {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        letter-spacing: 0.02em;
    }
    .verdict-real {
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.15), rgba(0, 210, 255, 0.08));
        color: #00e676;
        border: 1px solid rgba(0, 230, 118, 0.25);
    }
    .verdict-fake {
        background: linear-gradient(135deg, rgba(255, 23, 68, 0.15), rgba(255, 107, 107, 0.08));
        color: #ff1744;
        border: 1px solid rgba(255, 23, 68, 0.25);
    }

    /* ── Probability Bars ── */
    .prob-container {
        margin: 0.6rem 0;
    }
    .prob-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }
    .prob-name {
        font-size: 0.85rem;
        font-weight: 500;
        color: #c8cdd5;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .prob-value {
        font-size: 0.85rem;
        font-weight: 600;
        color: #e8ecf1;
        font-variant-numeric: tabular-nums;
    }
    .prob-bar-bg {
        width: 100%;
        height: 10px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 5px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 1s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .bar-real { background: linear-gradient(90deg, #00c853, #00e676); }
    .bar-perfect_ai { background: linear-gradient(90deg, #d50000, #ff1744); }
    .bar-compressed_ai { background: linear-gradient(90deg, #ff6d00, #ffab00); }
    .bar-edited_ai { background: linear-gradient(90deg, #aa00ff, #d500f9); }

    /* ── Metric Pill ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    .metric-pill {
        flex: 1;
        min-width: 140px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .metric-pill-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e8ecf1;
    }
    .metric-pill-label {
        font-size: 0.72rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #161822 100%);
    }
    .sidebar-stat {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
    }
    .sidebar-stat-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #00d2ff;
    }
    .sidebar-stat-label {
        font-size: 0.7rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── ELA section ── */
    .ela-header {
        font-size: 1rem;
        font-weight: 600;
        color: #c8cdd5;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeInUp 0.6s ease-out forwards;
    }

    /* ── Divider ── */
    .subtle-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading (cached)
# ──────────────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints/food_detector")


@st.cache_resource(show_spinner=False)
def load_detector():
    """Load FoodGuard model once and cache it."""
    meta_path = CHECKPOINT_DIR / "metadata.json"
    ckpt_path = CHECKPOINT_DIR / "food_ai_detector.pth"

    with open(meta_path) as f:
        metadata = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(
        metadata["model"],
        pretrained=False,
        num_classes=metadata["num_classes"],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((metadata["image_size"], metadata["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return model, metadata, transform, device


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, image: Image.Image, metadata: dict, transform, device: str):
    """Run threshold-calibrated prediction on a single image."""
    input_tensor = transform(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    logits = model(input_tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000

    probs = F.softmax(logits, dim=1)[0]
    threshold = metadata["threshold"]
    class_names = metadata["class_names"]

    # Real class index from metadata (ImageFolder sorts alphabetically,
    # so real=3, NOT 0). This was the root cause of the "everything is
    # edited_ai 100%" bug.
    real_idx = metadata.get("real_class_index", class_names.index("real"))
    prob_real = probs[real_idx].item()

    if prob_real > threshold:
        prediction = "real"
        confidence = prob_real
    else:
        # Pick the highest-probability AI class (skip the real index)
        ai_indices = [i for i in range(len(class_names)) if i != real_idx]
        ai_probs = [(i, probs[i].item()) for i in ai_indices]
        best_ai_idx, best_ai_prob = max(ai_probs, key=lambda x: x[1])
        prediction = class_names[best_ai_idx]
        confidence = best_ai_prob

    probabilities = {name: prob.item() for name, prob in zip(class_names, probs)}
    is_fake = prediction != "real"

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "is_fake": is_fake,
        "inference_ms": elapsed_ms,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ELA (Error Level Analysis)
# ──────────────────────────────────────────────────────────────────────────────
def compute_ela(image: Image.Image, quality: int = 90, scale: int = 10) -> Image.Image:
    """Compute Error Level Analysis of an image."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    ela = ImageChops.difference(image, recompressed)
    extrema = ela.getextrema()
    max_diff = max(ex[1] for ex in extrema)
    # Floor max_diff to avoid over-amplifying noise on lossless PNGs
    max_diff = max(max_diff, 20)
    scale_factor = 255.0 / max_diff * scale
    ela = ela.point(lambda x: min(int(x * scale_factor), 255))
    return ela


# ──────────────────────────────────────────────────────────────────────────────
# Bounding Box via ELA sliding window
# ──────────────────────────────────────────────────────────────────────────────
def find_anomaly_bbox(
    image: Image.Image,
    ela_image: Image.Image,
    tile: int = 64,
    stride: int = 32,
    top_k: float = 0.05,
):
    """
    Scan the ELA heatmap with a sliding window.
    Return (x1, y1, x2, y2) of the highest-mean-intensity tile,
    expanded to cover the top-K% brightest region.

    Returns None when the image looks uniform (real photo).
    """
    ela_gray = np.array(ela_image.convert("L"), dtype=np.float32)
    h, w = ela_gray.shape

    # ── sliding window scan ──────────────────────────────────────────
    best_score = -1
    best_box   = (0, 0, w, h)
    for y in range(0, h - tile + 1, stride):
        for x in range(0, w - tile + 1, stride):
            patch = ela_gray[y:y+tile, x:x+tile]
            score = patch.mean()
            if score > best_score:
                best_score = score
                best_box   = (x, y, x + tile, y + tile)

    # ── expand: include all pixels above threshold ───────────────────
    threshold = np.percentile(ela_gray, (1 - top_k) * 100)
    mask = ela_gray >= threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, best_score

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    # Add padding
    pad = 8
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    # Uniformity check — skip box for truly uniform images (real photos)
    global_mean = ela_gray.mean()
    if best_score < global_mean * 1.4:          # < 40% above average = likely real
        return None, best_score

    return (x1, y1, x2, y2), best_score


def draw_bounding_box(
    image: Image.Image,
    bbox: tuple,
    label: str,
    score: float,
    color: str = "#ef4444",
) -> Image.Image:
    """
    Draw a labelled rectangle with corner accents on a copy of the image.

    Args:
        image (PIL.Image.Image): The original image to draw on.
        bbox (tuple): A tuple of (x1, y1, x2, y2) coordinates.
        label (str): The text label to display above the bounding box.
        score (float): The anomaly or confidence score to display next to the label.
        color (str, optional): The hex color code for the bounding box. Defaults to "#ef4444".

    Returns:
        PIL.Image.Image: A new image with the bounding box overlay applied.
    """
    annotated = image.copy().convert("RGBA")
    overlay   = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw      = ImageDraw.Draw(overlay)

    x1, y1, x2, y2 = bbox

    # Convert hex color to RGBA
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)

    # Semi-transparent fill
    draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 35))
    # Thick border
    lw = max(3, image.width // 150)
    draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 230), width=lw)

    # Corner accents
    cs = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    for cx, cy, dx, dy in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        draw.line([(cx, cy), (cx + dx*cs, cy)], fill=(r, g, b, 255), width=lw+1)
        draw.line([(cx, cy), (cx, cy + dy*cs)], fill=(r, g, b, 255), width=lw+1)

    annotated = Image.alpha_composite(annotated, overlay).convert("RGB")

    # Label tag above box
    draw2 = ImageDraw.Draw(annotated)
    tag   = f" {label}  {score:.0f} "
    tx, ty = x1, max(0, y1 - 22)
    draw2.rectangle([tx, ty, tx + len(tag)*7, ty + 20], fill=(r, g, b))
    draw2.text((tx + 4, ty + 3), tag, fill=(255, 255, 255))

    return annotated


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for EfficientNet-B3 (timm).

    Hooks the last convolutional block's output, backprops from the
    predicted class logit, and computes the weighted spatial heatmap.
    """

    def __init__(self, model: torch.nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None
        self._handle_fwd = None
        self._handle_bwd = None
        self._register_hooks()

    def _register_hooks(self):
        # EfficientNet-B3 in timm: last feature block is model.blocks[-1]
        target_layer = self.model.blocks[-1]

        def _save_activation(_, __, output):
            self.activations = output.detach()

        def _save_gradient(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self._handle_fwd = target_layer.register_forward_hook(_save_activation)
        self._handle_bwd = target_layer.register_full_backward_hook(_save_gradient)

    def remove_hooks(self):
        self._handle_fwd.remove()
        self._handle_bwd.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for a specific class index.

        Args:
            input_tensor (torch.Tensor): A preprocessed image tensor of shape (1, C, H, W).
            class_idx (int): The target class index to generate the activation map for.

        Returns:
            np.ndarray: A 2D float32 heatmap array of shape (H, W) normalized to [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)          # forward (hooks capture activations)
        score  = logits[0, class_idx]              # scalar for target class
        score.backward()                           # backward (hooks capture gradients)

        # Global average pool gradients over spatial dims → (C,)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam = torch.nn.functional.relu(cam)        # keep only positive contributions
        cam = cam.squeeze().cpu().numpy()          # (H, W)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def apply_gradcam_overlay(
    original: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """
    Resizes a CAM heatmap to the original image size and blends it as an overlay.

    Args:
        original (PIL.Image.Image): The original RGB image.
        cam (np.ndarray): The 2D float32 heatmap generated by GradCAM.
        alpha (float, optional): The blending weight for the heatmap. Defaults to 0.45.
        colormap (str, optional): The matplotlib colormap to apply. Defaults to "jet".

    Returns:
        PIL.Image.Image: A new image containing the original image blended with the heatmap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Resize CAM to image size
    cam_pil    = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        original.size, resample=Image.BICUBIC
    )
    cam_smooth = cam_pil.filter(ImageFilter.GaussianBlur(radius=6))
    cam_arr    = np.array(cam_smooth, dtype=np.float32) / 255.0

    # Apply jet colormap
    cmap      = cm.get_cmap(colormap)
    heatmap   = (cmap(cam_arr)[:, :, :3] * 255).astype(np.uint8)
    heatmap   = Image.fromarray(heatmap).convert("RGB")

    # Blend with original
    orig_rgb  = original.convert("RGB")
    blended   = Image.blend(orig_rgb, heatmap, alpha)
    return blended


def run_gradcam(
    model:        torch.nn.Module,
    transform,
    device:       str,
    pil_image:    Image.Image,
    class_idx:    int,
) -> Image.Image:
    """
    Executes the full Grad-CAM pipeline on an image.

    This function temporarily attaches hooks, runs a forward and backward pass,
    overlays the resulting heatmap on the image, and then detaches the hooks.
    It deliberately runs with gradient tracking enabled.

    Args:
        model (torch.nn.Module): The PyTorch model to interpret.
        transform (callable): The torchvision transform pipeline.
        device (str): The compute device ("cuda" or "cpu").
        pil_image (PIL.Image.Image): The input image to interpret.
        class_idx (int): The target class index to explain.

    Returns:
        PIL.Image.Image: The image with the Grad-CAM heatmap overlay.
    """
    model.train(False)           # eval mode but gradients enabled
    gcam    = GradCAM(model)
    tensor  = transform(pil_image).unsqueeze(0).to(device)
    tensor.requires_grad_(False)

    try:
        cam     = gcam.generate(tensor, class_idx)
        overlay = apply_gradcam_overlay(pil_image, cam)
    finally:
        gcam.remove_hooks()
        model.eval()             # restore eval mode

    return overlay


# ──────────────────────────────────────────────────────────────────────────────
# UI Helpers
# ──────────────────────────────────────────────────────────────────────────────
BAR_COLORS = {
    "real": "bar-real",
    "perfect_ai": "bar-perfect_ai",
    "compressed_ai": "bar-compressed_ai",
    "edited_ai": "bar-edited_ai",
}

CLASS_ICONS = {
    "real": "✅",
    "perfect_ai": "🤖",
    "compressed_ai": "📦",
    "edited_ai": "✂️",
}

CLASS_DESCRIPTIONS = {
    "real": "Genuine, unedited food photograph",
    "perfect_ai": "High-quality AI-generated image (text-to-image)",
    "compressed_ai": "AI-generated image degraded by JPEG compression & resizing",
    "edited_ai": "Real image tampered via AI inpainting (e.g., contaminant inserted)",
}


def render_probability_bars(probabilities: dict):
    """Render animated horizontal probability bars."""
    html_parts = []
    for name, prob in probabilities.items():
        pct = prob * 100
        bar_class = BAR_COLORS.get(name, "bar-real")
        icon = CLASS_ICONS.get(name, "")
        html_parts.append(f"""
        <div class="prob-container">
            <div class="prob-label">
                <span class="prob-name">{icon} {name.replace('_', ' ')}</span>
                <span class="prob-value">{pct:.2f}%</span>
            </div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill {bar_class}" style="width: {pct}%"></div>
            </div>
        </div>
        """)
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_verdict(result: dict):
    """Render the large verdict banner."""
    if result["is_fake"]:
        label = result["prediction"].replace("_", " ").upper()
        st.markdown(f"""
        <div class="verdict verdict-fake animate-in">
            🚨 AI DETECTED — {label}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="verdict verdict-real animate-in">
            ✅ AUTHENTIC — REAL FOOD IMAGE
        </div>
        """, unsafe_allow_html=True)


def render_metrics(result: dict):
    """Render confidence and inference time pills."""
    conf_pct = result["confidence"] * 100
    inf_ms = result["inference_ms"]
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.markdown(f"""
    <div class="metric-row animate-in">
        <div class="metric-pill">
            <div class="metric-pill-value">{conf_pct:.1f}%</div>
            <div class="metric-pill-label">Confidence</div>
        </div>
        <div class="metric-pill">
            <div class="metric-pill-value">{inf_ms:.0f}ms</div>
            <div class="metric-pill-label">Inference Time</div>
        </div>
        <div class="metric-pill">
            <div class="metric-pill-value">{device}</div>
            <div class="metric-pill-label">Compute</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar(metadata: dict):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <span style="font-size: 2.5rem;">🛡️</span>
            <h2 style="margin: 0.25rem 0 0; font-weight: 700;
                background: linear-gradient(135deg, #00d2ff, #7b2ff7);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                FoodGuard
            </h2>
            <p style="color: #6b7280; font-size: 0.78rem; margin-top: 2px;">
                AI Food Fraud Detection System
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

        st.markdown("#### 📊 Model Info")
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-value">{metadata['model'].replace('_', '-').title()}</div>
            <div class="sidebar-stat-label">Architecture</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-value">{metadata['num_classes']} Classes</div>
            <div class="sidebar-stat-label">Output Categories</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-value">{metadata['image_size']}×{metadata['image_size']}</div>
            <div class="sidebar-stat-label">Input Resolution</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-value">{metadata['threshold']:.2f}</div>
            <div class="sidebar-stat-label">Decision Threshold</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-value">{metadata.get('test_accuracy', 0) * 100:.2f}%</div>
            <div class="sidebar-stat-label">Test Accuracy</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-value">{metadata.get('test_fpr', 0) * 100:.1f}%</div>
            <div class="sidebar-stat-label">False Positive Rate</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

        st.markdown("#### 🏷️ Class Legend")
        for name in metadata["class_names"]:
            icon = CLASS_ICONS.get(name, "•")
            desc = CLASS_DESCRIPTIONS.get(name, "")
            color = "#00e676" if name == "real" else "#ff6b6b"
            st.markdown(f"""
            <div style="margin: 0.4rem 0; padding: 0.5rem 0.6rem;
                        background: rgba(255,255,255,0.03); border-radius: 8px;
                        border-left: 3px solid {color};">
                <div style="font-size: 0.82rem; font-weight: 600; color: #e0e0e0;">
                    {icon} {name.replace('_', ' ').title()}
                </div>
                <div style="font-size: 0.7rem; color: #6b7280; margin-top: 2px;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="text-align:center; color:#4a4f5e; font-size:0.68rem;">'
            'Built with PyTorch • EfficientNet-B3 • Streamlit</p>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Load model
    with st.spinner("🔄 Loading FoodGuard model..."):
        model, metadata, transform, device = load_detector()

    # Sidebar
    render_sidebar(metadata)

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🛡️ FoodGuard</div>
        <div class="hero-subtitle">
            Upload a food image to detect AI-generated fakes, compression artifacts, and inpainting fraud
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Drop a food image here",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        # Show empty-state placeholder
        st.markdown("""
        <div class="upload-zone">
            <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">📤</p>
            <p style="color: #8b95a5; font-size: 1rem; margin: 0;">
                Drag & drop a food image above, or click <b>Browse files</b>
            </p>
            <p style="color: #4a4f5e; font-size: 0.78rem; margin-top: 0.5rem;">
                Supports JPG · PNG · WebP · BMP
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Show sample results section
        st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

        cols = st.columns(4)
        sample_items = [
            ("✅ Real", "Genuine photo", "#00e676"),
            ("🤖 Perfect AI", "Text-to-image", "#ff1744"),
            ("📦 Compressed AI", "Re-encoded AI", "#ffab00"),
            ("✂️ Edited AI", "Inpainted fraud", "#d500f9"),
        ]
        for col, (title, desc, color) in zip(cols, sample_items):
            with col:
                st.markdown(f"""
                <div style="text-align:center; padding: 1.5rem 0.75rem;
                    background: rgba(255,255,255,0.02); border-radius: 12px;
                    border: 1px solid rgba(255,255,255,0.06);">
                    <div style="font-size: 0.9rem; font-weight: 600; color: {color};">
                        {title}
                    </div>
                    <div style="font-size: 0.72rem; color: #6b7280; margin-top: 4px;">
                        {desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        return

    # ── Process uploaded image ────────────────────────────────────────────
    image = Image.open(uploaded_file).convert("RGB")

    # Run prediction
    with st.spinner("🔍 Analyzing image..."):
        result = predict(model, image, metadata, transform, device)

    # ── Verdict ───────────────────────────────────────────────────────────
    render_verdict(result)

    # ── Metrics row ───────────────────────────────────────────────────────
    render_metrics(result)

    st.markdown('<div class="subtle-divider"></div>', unsafe_allow_html=True)

    # ── Main Content: Two Columns ─────────────────────────────────────────
    col_img, col_analysis = st.columns([1, 1], gap="large")

    with col_img:
        # Compute ELA and bounding box once
        ela_image = compute_ela(image)
        bbox, ela_score = find_anomaly_bbox(image, ela_image)

        is_ai = result["is_fake"]
        box_color = {
            "edited_ai":     "#a855f7",
            "perfect_ai":    "#3b82f6",
            "compressed_ai": "#f97316",
            "real":          "#22c55e",
        }.get(result["prediction"], "#ef4444")

        # ── Tab view: Original / Anomaly Box / ELA / Grad-CAM ────────
        tab_orig, tab_bbox, tab_ela, tab_gcam = st.tabs(
            ["🖼️ Original", "🎯 Anomaly Box", "🔬 ELA Map", "🧠 Grad-CAM"]
        )

        with tab_orig:
            st.image(image, use_container_width=True,
                     caption=f"{uploaded_file.name} — {image.size[0]}×{image.size[1]}px")

        with tab_bbox:
            if bbox is not None:
                pred_label = result["prediction"].replace("_", " ").upper()
                annotated  = draw_bounding_box(
                    image, bbox, pred_label, ela_score, color=box_color
                )
                st.image(annotated, use_container_width=True,
                         caption="Highest-anomaly region detected via ELA sliding window")
                x1, y1, x2, y2 = bbox
                bw, bh = x2 - x1, y2 - y1
                st.markdown(
                    f"""
                    <div style="font-size:0.75rem;color:#8b95a5;margin-top:4px;">
                    Box: ({x1}, {y1}) → ({x2}, {y2}) &nbsp;·&nbsp;
                    Size: {bw}×{bh}px &nbsp;·&nbsp;
                    ELA score: <b style='color:{box_color}'>{ela_score:.1f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.image(image, use_container_width=True)
                st.success(
                    "No high-anomaly region detected — ELA map is uniform. "
                    "This is consistent with a genuine food photograph."
                )

        with tab_ela:
            st.image(ela_image, use_container_width=True,
                     caption="ELA: bright = different compression history (potential tampering)")
            st.markdown(
                '<p style="font-size:0.72rem;color:#4a4f5e;margin-top:4px;">'  
                'Uniform grey = genuine. Bright patches = AI artefact or inpainted region.</p>',
                unsafe_allow_html=True,
            )

        with tab_gcam:
            st.markdown(
                '<p style="font-size:0.78rem;color:#8b95a5;margin-bottom:8px;">'  
                '<b>Grad-CAM</b> backprops from the predicted class logit through '
                'EfficientNet-B3\'s last convolutional block, showing which pixels '
                'most influenced the decision. Red/yellow = high activation.</p>',
                unsafe_allow_html=True,
            )
            with st.spinner("Computing Grad-CAM…"):
                # class index of predicted class
                class_names = metadata["class_names"]
                pred_class_idx = class_names.index(result["prediction"])
                try:
                    gcam_overlay = run_gradcam(
                        model, transform, device, image, pred_class_idx
                    )
                    st.image(
                        gcam_overlay,
                        use_container_width=True,
                        caption=(
                            f"Grad-CAM · Target class: "
                            f"{result['prediction'].replace('_', ' ').title()} · "
                            f"Confidence: {result['confidence']*100:.1f}%"
                        ),
                    )
                    st.markdown(
                        '<p style="font-size:0.72rem;color:#4a4f5e;margin-top:4px;">'  
                        'Warm (red/yellow) regions = model focus. '
                        'Cool (blue) regions = low influence.</p>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")

    with col_analysis:
        # Prediction details card
        pred_class = "result-card-fake" if result["is_fake"] else "result-card-real"
        icon = CLASS_ICONS.get(result["prediction"], "")
        desc = CLASS_DESCRIPTIONS.get(result["prediction"], "")

        st.markdown(f"""
        <div class="result-card {pred_class} animate-in">
            <div style="font-size: 0.72rem; text-transform: uppercase;
                        letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.5rem;">
                Classification Result
            </div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #e8ecf1;">
                {icon} {result['prediction'].replace('_', ' ').title()}
            </div>
            <div style="font-size: 0.82rem; color: #8b95a5; margin-top: 0.25rem;">
                {desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("""
        <div class="result-card animate-in">
            <div style="font-size: 0.72rem; text-transform: uppercase;
                        letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.75rem;">
                Class Probabilities
            </div>
        """, unsafe_allow_html=True)

        render_probability_bars(result["probabilities"])

        st.markdown("</div>", unsafe_allow_html=True)

        # Technical details
        w, h = image.size
        file_size_kb = uploaded_file.size / 1024

        st.markdown(f"""
        <div class="result-card animate-in">
            <div style="font-size: 0.72rem; text-transform: uppercase;
                        letter-spacing: 0.1em; color: #6b7280; margin-bottom: 0.75rem;">
                Image Metadata
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                <div>
                    <span style="font-size: 0.72rem; color: #6b7280;">Dimensions</span><br/>
                    <span style="font-size: 0.9rem; font-weight: 600; color: #c8cdd5;">
                        {w} × {h} px
                    </span>
                </div>
                <div>
                    <span style="font-size: 0.72rem; color: #6b7280;">File Size</span><br/>
                    <span style="font-size: 0.9rem; font-weight: 600; color: #c8cdd5;">
                        {file_size_kb:.1f} KB
                    </span>
                </div>
                <div>
                    <span style="font-size: 0.72rem; color: #6b7280;">Format</span><br/>
                    <span style="font-size: 0.9rem; font-weight: 600; color: #c8cdd5;">
                        {uploaded_file.type.split('/')[-1].upper()}
                    </span>
                </div>
                <div>
                    <span style="font-size: 0.72rem; color: #6b7280;">Threshold</span><br/>
                    <span style="font-size: 0.9rem; font-weight: 600; color: #c8cdd5;">
                        P(real) > {metadata['threshold']:.2f}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
