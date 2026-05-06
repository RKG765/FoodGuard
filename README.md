<p align="center">
  <h1 align="center">🛡️ FoodGuard</h1>
  <p align="center"><b>AI-Powered Food Fraud Detection System</b></p>
  <p align="center">
    Detect AI-generated, compressed, and tampered food images using deep learning forensics.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Model-EfficientNet--B3-green" />
  <img src="https://img.shields.io/badge/GPU-RTX%205070%20Ti-76b900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Academic-orange" />
</p>

---

## 📌 Overview

**FoodGuard** is a 4-class deep learning system that classifies food images as:

| Class | Description |
|-------|-------------|
| 🟢 **Real** | Genuine, unedited food photographs |
| 🔴 **Perfect AI** | High-quality AI-generated food images (no post-processing) |
| 🟡 **Compressed AI** | AI-generated images degraded by JPEG compression & resizing |
| 🟠 **Edited AI** | Real images tampered via AI inpainting (e.g., cockroach, mold inserted) |

> **Goal:** Achieve ≤ 5% False Positive Rate — genuine food photos must NOT be wrongly flagged.



## 🏗️ System Architecture

```mermaid
graph TB
    subgraph Data Collection
        K1["Kaggle: Food-101<br/>~101K images"]
        K2["Kaggle: Indian Food<br/>~4K images"]
        K3["Kaggle: Food Image Dataset<br/>~86K images"]
    end

    subgraph AI Generation
        G1["RealVisXL V4.0<br/>Text-to-Image"]
        G2["SDXL Inpainting<br/>Fraud Objects"]
    end

    K1 & K2 & K3 --> CSV["build_csv.py<br/>dataset_index.csv"]
    CSV --> ORG["organize_4class_dataset.py"]
    G1 --> C1["class1: Perfect AI<br/>600 imgs"]
    G1 --> C2["class2: Compressed AI<br/>600 imgs"]
    G1 --> C3["class3: Degraded AI<br/>400 imgs"]
    G2 --> C4["class4: Edited Real<br/>500+ imgs"]

    ORG --> DS["dataset_4class/<br/>train / val / test<br/>70% / 15% / 15%"]
    C1 & C2 & C3 & C4 --> DS

    DS --> TR["train_4class_detector.py<br/>EfficientNet-B3 + AMP"]
    TR --> CK["checkpoints/<br/>food_ai_detector.pth"]
    CK --> INF["inference.py<br/>Threshold-Calibrated Prediction"]

    style DS fill:#1a1a2e,stroke:#e94560,color:#fff
    style TR fill:#0f3460,stroke:#e94560,color:#fff
    style INF fill:#16213e,stroke:#00d2ff,color:#fff
```

---

## 🔬 Training Pipeline

```mermaid
flowchart LR
    A["📂 Load Dataset<br/>ImageFolder"] --> B["🔄 Transforms<br/>512×512, Normalize"]
    B --> C["🧠 EfficientNet-B3<br/>Pretrained ImageNet"]
    C --> D["📉 Weighted CE Loss<br/>[1.2, 1.0, 1.0, 1.0]"]
    D --> E["⚡ AdamW + AMP<br/>lr=3e-4"]
    E --> F["📊 Cosine Scheduler<br/>20 Epochs"]
    F --> G{"Val Accuracy<br/>Improved?"}
    G -- Yes --> H["💾 Save Best Model"]
    G -- No --> I["Continue Training"]
    H --> J["🎯 Threshold Calibration<br/>Target FPR ≤ 5%"]
    J --> K["✅ Final Evaluation<br/>Confusion Matrix + Metrics"]
```

---

## 🧬 Model Architecture

```mermaid
graph LR
    IMG["Input Image<br/>512 × 512 × 3"] --> EN["EfficientNet-B3<br/>~12M params"]
    EN --> GAP["Global Avg Pool<br/>1536-d"]
    GAP --> FC["Linear<br/>1536 → 4"]
    FC --> SM["Softmax"]
    SM --> R["P(real)"]
    SM --> P["P(perfect_ai)"]
    SM --> C["P(compressed_ai)"]
    SM --> E["P(edited_ai)"]

    R --> TH{"P(real) > θ ?"}
    TH -- Yes --> REAL["✅ REAL"]
    TH -- No --> AI["⚠️ AI Detected<br/>argmax of AI classes"]

    style REAL fill:#00c853,stroke:#00c853,color:#fff
    style AI fill:#ff1744,stroke:#ff1744,color:#fff
```

---

## 🗂️ Project Structure

```
FoodGuard/
├── config/
│   └── category_mapping.yaml      # Food category mapping
├── src/
│   ├── data/
│   │   ├── dataset_loader.py      # Generic food dataset loader
│   │   ├── detector_dataset.py    # 4-class detector dataset
│   │   ├── augmentations.py       # Training transforms
│   │   └── ela.py                 # Error Level Analysis (forensic)
│   └── models/
│       ├── food_classifier.py     # Base food classifier
│       ├── dual_stream_detector.py# RGB + FFT dual-stream model
│       ├── focal_loss.py          # Focal Loss implementation
│       └── trainer.py             # Training loop manager
├── scripts/
│   ├── build_csv.py               # Build master dataset CSV
│   ├── build_detector_csv.py      # Build 4-class detector CSV
│   ├── organize_4class_dataset.py # Organize into train/val/test
│   ├── generate_ai_images.py      # AI food image generation (SDXL)
│   ├── generate_fraud_inpainting.py # Fraud object inpainting
│   ├── generate_fraud_simple.py   # Overlay-based fraud fallback
│   └── validate_csv.py            # Dataset validation checks
├── train_4class_detector.py       # 🚀 Main training script
├── evaluate.py                    # Model evaluation & metrics
├── inference.py                   # Single-image inference
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 🪄 Run the Web App (Easiest Way!)

You don't need to train the model to try it! Just download the repository, run the Streamlit app, upload a food photo, and watch the forensic AI do its magic ✨.

```bash
streamlit run app.py
```

It will automatically load the best trained model. You can drag & drop images directly into your browser to see real-time forensic analysis, including Grad-CAM explainability and Error Level Analysis (ELA) bounding boxes.

### 3. Run Inference (Command Line)

If you prefer the terminal:

```bash
python inference.py path/to/food_image.jpg
```

### 4. Build Dataset & Train (For Developers)

If you want to train the model yourself or generate more AI images, you will use the files in the `scripts/` folder.

**Image Generation Scripts:**
All AI image generation happens in the `scripts/` directory. If you want to generate your own fake food images, use scripts like `generate_ai_images.py` and `generate_fraud_inpainting.py`.

**Prepare Data & Train:**
```bash
# Build unified CSV and organize datasets
python scripts/build_csv.py
python scripts/organize_4class_dataset.py

# Run training (uses AMP automatically)
python train_4class_detector.py
```

Checkpoints will be saved to `checkpoints/food_detector/`.

**Sample Output (Inference):**

```
============================================================
Image: test_burger.jpg
============================================================
Prediction:  REAL
Confidence:  94.32%
Is Fake:     NO

Class Probabilities:
  real           :  94.32% █████████████████████████████████████████████
  perfect_ai     :   3.21% █
  compressed_ai  :   1.87% 
  edited_ai      :   0.60% 
============================================================
```

---

## 🎯 AI Image Generation

FoodGuard generates its own training data using **RealVisXL V4.0** (`SG161222/RealVisXL_V4.0`):

```mermaid
graph TD
    subgraph "Text-to-Image Generation"
        P["Curated Food Prompts<br/>+ Quality Modifiers"] --> SDXL["RealVisXL V4.0<br/>25 steps, cfg=5-7.5"]
        SDXL --> RAW["Class 1: Raw AI<br/>512×512 PNG"]
        SDXL --> COMP["Class 2: Compressed<br/>JPEG q=40-85, resize"]
        SDXL --> DEG["Class 3: Degraded<br/>Blur + Noise"]
    end

    subgraph "Inpainting Generation"
        REAL["Real Food Image<br/>from clean pool"] --> MASK["Irregular Mask<br/>2-4% area, center-biased"]
        MASK --> INP["SDXL Inpainting<br/>cfg=4.5, strength=0.99"]
        INP --> EDIT["Class 4: Edited<br/>Fraud object inserted"]
    end

    style RAW fill:#e3f2fd,stroke:#1565c0,color:#000
    style COMP fill:#fff3e0,stroke:#e65100,color:#000
    style DEG fill:#fce4ec,stroke:#c62828,color:#000
    style EDIT fill:#ffebee,stroke:#b71c1c,color:#000
```

**Fraud Objects:** cockroach, housefly, mosquito, bee, ant, worm, human hair, mold, plastic fragment, paper piece, metal shard.

---

## 📊 Data Sources

> ⚠️ **Note:** Due to GitHub's file size limits, the actual image datasets are **NOT** included in this repository. 
> 
> 🔗 **[Download the AI-Generated Fake Food Dataset here (Kaggle / Google Drive)]** *(Link to be added)*

| Dataset | Source | Images | Cuisine Coverage |
|---------|--------|--------|-----------------|
| **Food-101** | Kaggle (ETH Zurich) | ~101,000 | Western, International |
| **Indian Food Dataset** | Kaggle | ~4,000 | Indian (biryani, paneer, etc.) |
| **Food Image Dataset** | Kaggle (UECFOOD256 + AIcrowd) | ~86,000 | Japanese, Mixed |
| **AI-Generated** | *See link above* | ~2,000 | Multi-cuisine |
| **AI-Inpainted Fraud** | *See link above* | ~550 | Multi-cuisine |

**Total Real Images:** ~191,000+ &nbsp;|&nbsp; **Sampled for Training:** 5,000 (balanced prototype)

---

## ⚙️ Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Backbone** | EfficientNet-B3 | Best accuracy-per-param; fits 12GB VRAM |
| **Image Size** | 512 × 512 | Preserves forensic artifacts vs 224×224 |
| **Batch Size** | 16 | Max for 512×512 on 12GB |
| **Optimizer** | AdamW | Better weight decay for fine-tuning |
| **Learning Rate** | 3e-4 | Standard for timm fine-tuning |
| **Scheduler** | Cosine Annealing (T=20) | Smooth decay, no sudden drops |
| **Loss** | Cross-Entropy | Weights: [1.2, 1.0, 1.0, 1.0] — penalizes FP on real |
| **AMP** | Enabled | ~2× speed, ~40% less VRAM |
| **Epochs** | 20 | With early stopping |
| **Target FPR** | ≤ 5% | Calibrated via threshold sweep |

---

## 🧪 Why 4 Classes, Not Binary?

```mermaid
graph TD
    B["Binary Classifier<br/>Real vs Fake"] --> L1["❌ Loses WHY it's fake"]
    B --> L2["❌ Misses subtle edits<br/>95% real, 5% inpainted"]

    F["4-Class Classifier"] --> W1["✅ Detects fully AI-generated"]
    F --> W2["✅ Handles compressed AI<br/>social media sharing"]
    F --> W3["✅ Catches subtle inpainting<br/>deliberate fraud"]
    F --> W4["✅ Maintains low FPR<br/>on genuine photos"]

    style B fill:#ffcdd2,stroke:#c62828,color:#000
    style F fill:#c8e6c9,stroke:#2e7d32,color:#000
```

---

## 📈 Training Results

```mermaid
flowchart TD
    subgraph Dataset["📦 Dataset (7,068 images)"]
        T["Train — 4,947"]
        V["Val — 1,060"]
        TS["Test — 1,061"]
    end

    subgraph Journey["🏋️ Training Journey (20 Epochs · RTX 5070 Ti)"]
        E1["Epoch 1\nVal Acc: 90.22%\nFPR: 4.44%"]
        E7["Epoch 7\nVal Acc: 99.53%\nFPR: 1.11%"]
        E18["⭐ Epoch 18 — Best\nVal Acc: 99.91%\nFPR: 1.11%"]
    end

    subgraph Calibration["🎯 Threshold Calibration"]
        TH["Optimal Threshold: 0.50\nFPR on Val: 1.11%\nTarget met: ≤ 5% ✓"]
    end

    subgraph TestResults["✅ Test Set Results"]
        direction LR
        ACC["Accuracy\n99.81%"]
        FPR["False Positive Rate\n0.00% 🎯"]
    end

    subgraph CM["🔢 Confusion Matrix"]
        R["Real\n90/90 ✓ (0 wrong)"]
        P["Perfect AI\n119/120 ✓"]
        C["Compressed AI\n101/101 ✓ (perfect)"]
        E["Edited AI\n749/750 ✓"]
    end

    Dataset --> Journey
    E1 --> E7 --> E18
    E18 --> Calibration
    Calibration --> TestResults
    TestResults --> CM

    style E18 fill:#00c853,stroke:#00c853,color:#fff
    style FPR fill:#00c853,stroke:#00c853,color:#fff
    style ACC fill:#0096ff,stroke:#0096ff,color:#fff
    style TH  fill:#1a1a2e,stroke:#00c853,color:#fff
```

---

## 🗺️ Roadmap

- [x] Real food data collection (3 Kaggle datasets, ~191K images)
- [x] AI image generation pipeline (RealVisXL V4.0)
- [x] Fraud inpainting pipeline (SDXL Inpainting)
- [x] Data processing & 4-class organization
- [x] Training script with AMP and threshold calibration
- [x] DGX Kubernetes Training Pipeline Setup (PVC, MIG GPU)
- [x] Executed model training on DGX Server
- [x] Threshold calibration for ≤5% FPR
- [x] Evaluation (confusion matrix, per-class metrics)
- [ ] Grad-CAM explainability visualization (Planned)
- [ ] Dual-stream model (RGB + FFT frequency analysis) (Planned)
- [ ] REST API deployment (FastAPI) (Planned)

---

## 🔧 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **PyTorch** | Deep learning framework |
| **timm** | EfficientNet model zoo |
| **HuggingFace Diffusers** | SDXL text-to-image & inpainting |
| **RealVisXL V4.0** | Photorealistic image generation |
| **scikit-learn** | Metrics & evaluation |
| **Pillow** | Image I/O and processing |
| **xformers** | Memory-efficient attention for SDXL |
| **CUDA AMP** | Mixed-precision training |
| **matplotlib / seaborn** | Visualization |

---

## 🌍 Real-World Applications

- **Food Delivery Apps** — Detect fraudulent complaint images (fake contaminants for refunds)
- **Restaurant Reviews** — Filter AI-manipulated food photos
- **Food Safety Agencies** — Verify authenticity of food complaint evidence
- **Social Media** — Flag AI-generated food content for transparency

---

## 👥 Team

| Member | Role |
|--------|------|
| **Raj** | Architecture, AI generation, training pipeline, inference |
| **Rahul** | Data validation, category mapping, evaluation scripts |
| **Aman** | Dataset management, augmentation testing, dual-stream model |

---

<p align="center">
  <b>Project</b><br/>
  Built with 🔬 PyTorch and ☕ caffeine
</p>
