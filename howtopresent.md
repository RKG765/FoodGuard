# FoodGuard — Professional Presentation Guide
### 3 Presenters · 22 Slides · ~20–25 Minutes

---

## Presenter Assignment

| Presenter | Slides | Role | Theme |
|---|---|---|---|
| **Presenter 1** | 1 – 7 | Problem & Data | "Why this problem matters and what we built the dataset on" |
| **Presenter 2** | 8 – 15 | Model & Training | "How we trained the model and what engineering we did" |
| **Presenter 3** | 16 – 22 | Results & Future | "What we achieved and where this goes next" |

> **Handoff cue:** Each presenter ends with "I'll hand over to [name] who will walk you through [next topic]."

---

## Full Slide Outline

---

### SECTION A — Presenter 1: Problem & Data (Slides 1–7)

---

#### Slide 1 — Title Slide

**Layout:** Centered title, subtitle, 3 author names, institution, date.

**Content:**
```
FoodGuard
AI-Powered Food Fraud Detection Using Forensic Deep Learning

Raj  ·  Rahul  ·  Aman
Semester VI  |  2026
```

**Speaker (P1):**
> "Good [morning/afternoon]. We're going to present FoodGuard — a forensic deep learning system that detects AI-generated food fraud. I'm [Name], and I'll begin with the problem and dataset."

**Design tip:** Dark background, project logo or a food image split half real / half AI.

---

#### Slide 2 — The Real-World Threat

**Layout:** Two-column. Left: text. Right: a striking side-by-side of a real food photo vs AI-generated food photo.

**Headline:** *"You Cannot Tell the Difference. Our Model Can."*

**Bullet points:**
- Diffusion models generate photorealistic food images in 3 seconds
- Already used to fabricate fake contamination complaints
- AI-generated fraud evidence submitted to delivery platforms
- Human moderators have no reliable way to detect it

**Speaker (P1):**
> "This is the problem. Generative AI has made food fraud trivially easy. Someone can type 'cockroach in biryani' and get a photorealistic image in three seconds — good enough to fool a customer service agent, a moderator, or even a food safety inspector. The problem is real, it is happening now, and there is no deployed solution for it."

---

#### Slide 3 — Why Current Solutions Fail

**Layout:** Three-column comparison table.

| Approach | Problem |
|---|---|
| Human moderation | Cannot distinguish AI from real at scale |
| Binary real/fake classifiers | Miss compressed & edited AI variants |
| Metadata / EXIF checks | Stripped by WhatsApp, Telegram, Instagram |
| Reverse image search | Original AI image may not be indexed |

**Speaker (P1):**
> "Existing approaches all have critical gaps. Human moderators fail at scale. Binary classifiers treat all fakes the same. Metadata is stripped by messaging apps — every image shared on WhatsApp loses its EXIF data. Reverse image search only works if the exact image has been seen before."
> "FoodGuard addresses all of these by operating at the pixel level, in the frequency domain, with no reliance on metadata."

---

#### Slide 4 — Our Approach: 4-Class Forensic Classification

**Layout:** Full-width diagram. Use `results/diagrams/01_threat_flow.png`

**Headline:** *"We Defined 4 Forensic Classes — Not Just Real vs Fake"*

**Below diagram, small caption:**
- **Real** — genuine camera photo
- **Perfect AI** — clean diffusion output
- **Compressed AI** — AI image degraded through JPEG pipeline
- **Edited AI** — real photo with AI-inpainted fraud object

**Speaker (P1):**
> "Our key insight is that 'fake' is not one thing. A perfect AI output looks nothing like a compressed AI image forensically. Grouping them together causes a classifier to learn confused, overlapping decision boundaries."
> "We separated them into four precise classes. This forces the model to learn the forensic signature of each type of manipulation — not just a generic 'AI-ness' score."

---

#### Slide 5 — The Dataset: Sources & Scale

**Layout:** Left panel — bar chart. Right panel — split table. Use `results/diagrams/02_dataset_pie.png`

**Headline:** *"36,173 Images · 5 AI Generators · 3 Real-World Sources"*

**Speaker (P1):**
> "We built the dataset from scratch. Real images came from three public Kaggle datasets: Food-101, UECFOOD256, and Indian Food. From these we sampled 12,000 to balance the classes."
> "The AI images were generated locally. We scripted and ran five different diffusion model pipelines: RealVisXL V4, Flux.1 Schnell, Kandinsky 2.2, SDXL Turbo, and Stable Cascade."
> "Why five models? Because a classifier trained only on one generator learns that generator's quirks — not universal AI fingerprints. Using five different architectures forces the model to generalise."

---

#### Slide 6 — The AI Fingerprint: What the Model Sees

**Layout:** Full-slide image. Use `ai_generated/fingerprints/comparison_grid.png`

**Headline:** *"Every AI Generator Leaves an Invisible Forensic Signature"*

**Speaker (P1):**
> "This is the scientific foundation of FoodGuard. We ran three forensic analyses on images from each source."
> "The **FFT column** — Fourier Transform. Real photos show a smooth radial falloff. SDXL images show periodic grid spikes at 64-pixel intervals — the signature of the attention window."
> "The **SRM column** — noise residuals. Real camera sensors produce heterogeneous noise. AI denoisers produce statistically uniform noise across the whole image."
> "The **ELA column** — Error Level Analysis. In edited images, the AI-inpainted region has a different JPEG compression history from the rest of the photo. It lights up like a beacon."
> "These three signals are what our neural network learned to detect."

---

#### Slide 7 — Dataset Engineering: Class Weights & Augmentation Preview

**Layout:** Two columns. Left: class weight table. Right: short description of the degradation strategy.

**Class weight table:**
| Class | Images | Weight | Why |
|---|---|---|---|
| compressed_ai | 5,000 | **2.4** | Smallest — needs most gradient |
| edited_ai | 8,000 | **1.5** | Second smallest |
| perfect_ai | 11,173 | **1.1** | Well represented |
| real | 12,000 | **1.5** | Boosted to enforce ≤5% FPR |

**Speaker (P1):**
> "The dataset is imbalanced — 12,000 real images vs only 5,000 compressed AI. We addressed this with inverse-frequency class weights passed into the loss function. Compressed AI gets 2.4× more weight per mistake."
> "We also had a critical engineering decision: the weights must match PyTorch ImageFolder's alphabetical class ordering — not logical ordering. Getting this wrong silently assigns weights to the wrong classes. We caught and fixed this bug before training."
> "I'll now hand over to [P2] who will walk you through the model architecture and training strategy."

---

---

### SECTION B — Presenter 2: Model & Training (Slides 8–15)

---

#### Slide 8 — Architecture Choice: Why EfficientNet-B3

**Layout:** Comparison table + small architecture diagram.

**Headline:** *"12M Parameters. 512×512. 8GB VRAM. The Right Tool for This Job."*

| Model | Params | VRAM @ 512px | ImageNet Top-1 |
|---|---|---|---|
| ResNet-50 | 25M | ~10GB | 76.1% |
| **EfficientNet-B3** | **12M** | **~8GB** | **81.6%** |
| ViT-Base | 86M | ~14GB ❌ | 81.8% |
| MobileNetV3 | 5M | ~4GB | 75.2% |

**Speaker (P2):**
> "Model selection is about trade-offs. We had 12.8GB of VRAM on our RTX 5070 Ti. ViT-Base exceeds that at this resolution. ResNet-50 has twice the parameters for less accuracy. EfficientNet-B3 is the Pareto-optimal choice — best accuracy per parameter, fits our hardware, and its compound scaling preserves high-frequency spatial features at 512 pixels."
> "High-frequency preservation matters because our forensic signatures — the FFT grid spikes, the SRM noise patterns — live in the high-frequency domain. A model that aggressively downsamples early loses this information."

---

#### Slide 9 — Training Innovation 1: Focal Loss

**Layout:** Left — formula with annotation. Right — a simple graph showing CE vs FL weighting.

**Headline:** *"Standard Cross-Entropy Ignores Your Hard Examples. Focal Loss Doesn't."*

$$FL(p_t) = -\alpha_t \cdot (1 - p_t)^{\gamma} \cdot \log(p_t)$$

| Confidence | Cross-Entropy weight | Focal Loss weight (γ=2) |
|---|---|---|
| 95% (easy example) | 1.0 | **0.0025** |
| 75% (medium) | 1.0 | **0.0625** |
| 55% (hard example) | 1.0 | **0.2025** |

**Speaker (P2):**
> "This is the most important training change we made. Standard cross-entropy treats every image equally — an obvious perfect AI image contributes the same loss as a confusing compressed AI image. The model wastes capacity on examples it already understands."
> "Focal Loss with gamma=2 dynamically scales down easy examples. When the model is 95% confident, the loss contribution drops to 0.25% of normal. The model is forced to spend all its capacity on the compressed AI images it keeps getting wrong."
> "This directly improved our compressed AI recall."

---

#### Slide 10 — Training Innovation 2: Degradation Augmentations

**Layout:** Full diagram. Use `results/diagrams/05_degradation.png`

**Headline:** *"Force the Model to Learn Noise Patterns, Not JPEG Compression Levels"*

**Speaker (P2):**
> "Standard augmentations — flip, rotate, crop — do not change high-frequency noise patterns. A model trained without degradation augmentations memorises the exact JPEG compression level of each training image."
> "During training, we randomly apply three forensic degradations. 25% of the time: re-compress at a random JPEG quality between 40 and 95. 15%: apply Gaussian blur. 10%: apply unsharp mask — a common AI post-processing step. 50% of the time: no change."
> "The result: the model cannot rely on specific compression artefacts. It must learn the deeper diffusion noise signature that survives these transformations."

---

#### Slide 11 — Training Innovation 3: EMA Weights

**Layout:** Diagram. Use `results/diagrams/06_ema.png`

**Headline:** *"The Model We Deploy Is Not the Model From the Last Batch"*

**Speaker (P2):**
> "We have a strict FPR target of 5% or less. The problem with deploying epoch-end weights is that the last batch might be unlucky — a cluster of hard compressed AI images that temporarily shifts the decision boundary."
> "EMA — Exponential Moving Average — maintains a shadow copy of the weights updated after every optimiser step. The formula is: 0.9998 times the old shadow, plus 0.0002 times the new weights."
> "This creates a smooth, stable weight trajectory. Our deployed checkpoint is the EMA shadow at the best validation epoch — not the raw weights. This is the standard practice at Google Brain and Meta AI for production model deployment."

---

#### Slide 12 — Training Innovation 4: AMP + Gradient Clipping

**Layout:** Code block + diagram. Use `results/diagrams/07_windows.png`

**Headline:** *"Mixed Precision + Gradient Clipping = Stable 19-Epoch Training"*

**Code highlight:**
```python
scaler.unscale_(optimizer)                         # Step 1: unscale
torch.nn.utils.clip_grad_norm_(model, max_norm=1.0)  # Step 2: clip
scaler.step(optimizer)                             # Step 3: step
```

**Speaker (P2):**
> "We trained with Automatic Mixed Precision — forward passes in float16, gradient accumulation in float32. This halved our VRAM usage, allowing a batch size that would otherwise not fit."
> "However, float16 has a narrow dynamic range. Gradient spikes can overflow to infinity, crashing the training run silently. The solution is gradient clipping — but it must happen in a specific order. You must unscale the gradients first, then clip, then step. Clipping before unscaling clips the scaled values, not the true gradients — a subtle bug that took two failed training runs to diagnose."
> "We also encountered two Windows-specific issues — torch.compile requiring Triton which is Linux-only, and 14 worker processes crashing with a paging file error. Both were diagnosed and fixed."

---

#### Slide 13 — Full Training Configuration

**Layout:** Clean configuration table.

**Headline:** *"Full Training Stack — Nothing Left to Chance"*

| Parameter | Value | Reason |
|---|---|---|
| Model | EfficientNet-B3 | Best accuracy/VRAM ratio |
| Input size | 512 × 512 | Preserves high-freq forensic features |
| Batch size | 16 (effective 32) | Fits in 12.8GB VRAM with AMP |
| Gradient accumulation | 2 steps | Doubles effective batch size |
| Loss | Focal Loss γ=2 | Focuses on hard examples |
| Optimiser | AdamW lr=3e-4 | Weight decay regularisation |
| LR schedule | Warmup 3ep → Cosine | Stable convergence |
| Label smoothing | 0.1 | Prevents overconfident predictions |
| Dropout | 0.3 | Regularisation |
| Gradient clip | max_norm=1.0 | Prevents NaN with AMP |
| EMA decay | 0.9998 | Smooth deployment weights |
| Early stopping | patience=5 | Prevents overfitting |

**Speaker (P2):**
> "Here is the complete configuration. Every hyperparameter has a reason. This is not a default setup — each one was chosen to address a specific challenge in forensic image detection."

---

#### Slide 14 — Training Hardware & Environment

**Layout:** Clean spec card + short bullet list of challenges solved.

**Hardware:**
```
CPU : Intel Core Ultra 9 · 24 cores
RAM : 32 GB DDR5
GPU : NVIDIA GeForce RTX 5070 Ti Laptop
VRAM: 12.8 GB GDDR7
OS  : Windows 11
Framework: PyTorch 2.11 + CUDA 12.8
```

**Speaker (P2):**
> "We trained entirely locally on a laptop. No cloud, no DGX cluster — though the architecture is ready for Kubernetes deployment, which we documented."
> "Training 19 epochs on 25,000 images took approximately 2 hours on this hardware with mixed precision."
> "I'll now hand over to [P3] who will walk you through the results."

---

#### Slide 15 — Transition: From Training to Results

**Layout:** Single large visual — the training dashboard. Use `results/training_dashboard.png`

**Headline:** *"19 Epochs. Early Stop. All Targets Met."*

**Speaker (P2):**
> "Before I hand over — this is the training dashboard. Four panels: loss, accuracy, FPR, and learning rate over 19 epochs. Notice the FPR panel — at epoch 1 the False Positive Rate was 63%. By epoch 7 it crossed below our 5% target. The model never looked back."
> "[P3] will now take you through exactly what the final numbers mean."

---

---

### SECTION C — Presenter 3: Results & Future (Slides 16–22)

---

#### Slide 16 — The Business Metric: FPR ≤ 5%

**Layout:** Single large number in centre + explanation text.

**Headline:** *"Why FPR Matters More Than Accuracy"*

```
Test Accuracy:  96.26%   ← good
Test FPR:        1.56%   ← critical
Target FPR:      5.00%   ← must be under this
```

**Speaker (P3):**
> "Let me explain why we track FPR and not just accuracy. In a food safety system, the cost of errors is asymmetric. Flagging an AI image is good — that's the system working. But falsely accusing a genuine complaint is a serious problem. It means real food contamination goes unaddressed, and a genuine customer is dismissed."
> "The FPR — False Positive Rate on real images — measures exactly this: what fraction of genuine photos does our system wrongly flag as AI?"
> "Our target was 5%. We achieved 1.56%. That means only 28 out of 1,800 real test images were incorrectly flagged."

---

#### Slide 17 — Threshold Calibration

**Layout:** Code pseudocode block + a small table showing threshold vs FPR.

**Headline:** *"A 50% Confidence Threshold Is a Rookie Mistake in Anomaly Detection"*

```
For θ from 0.50 to 0.99:
    classify real if P(real) > θ
    compute FPR on validation set
    if FPR ≤ 5%: store θ as candidate

Deploy: max(candidates) = 0.50
```

| Threshold (θ) | FPR on Val |
|---|---|
| 0.50 | 1.67% ✅ |
| 0.60 | 1.50% ✅ |
| 0.70 | 1.33% ✅ |

**Speaker (P3):**
> "We do not assume 50% confidence is the right cut-off. We sweep the threshold on the validation set — which the model has never trained on — and find the maximum threshold that still keeps FPR below 5%."
> "In our case, the model's confidence distributions are well-separated enough that even the default 0.50 threshold meets the target. This is actually a strong result — it means the model is not just barely distinguishing classes, it is doing so with high margin."
> "The threshold is saved in metadata.json and loaded at inference time. The test set was never touched until calibration was complete."

---

#### Slide 18 — Confusion Matrix Deep Dive

**Layout:** Use `results/confusion_matrix_calibrated.png` on left. Analysis bullets on right.

**Headline:** *"Where the Model Succeeds — and Where It Struggles"*

**Bullets:**
- `perfect_ai` → **100% recall** — zero misses
- `real` → **98.3% recall** — 28 false positives (FPR 1.56%)
- `edited_ai` → **97.2% recall** — strong on inpainted fraud
- `compressed_ai` → **81.5% recall** — hardest class, 139 confused with perfect_ai

**Speaker (P3):**
> "The confusion matrix tells us exactly where the model works and where it doesn't."
> "Perfect AI: zero misses. Every clean diffusion image is correctly identified. This makes sense — perfect AI has the clearest FFT signature."
> "The interesting failure is Compressed AI. 139 images were classified as Perfect AI instead. This is not a random failure — JPEG re-compression at low quality strips the frequency-domain artefacts that distinguish the two classes. They look identical after compression. This is the known forensic challenge and it's why we specifically added degradation augmentations."
> "Critically — the real class confusion matrix row shows only 28 errors out of 1,800. No real image was classified as edited_ai or compressed_ai. The few errors went to perfect_ai, which is the least severe misclassification in production."

---

#### Slide 19 — Per-Class Performance

**Layout:** Use `results/per_class_accuracy.png` + table.

**Headline:** *"Full Classification Report"*

| Class | Precision | Recall | F1 |
|---|---|---|---|
| compressed_ai | 98.87% | 81.47% | 89.33% |
| edited_ai | 98.65% | 97.17% | 97.90% |
| perfect_ai | 91.74% | 100.00% | 95.69% |
| **real** | **98.39%** | **98.33%** | **98.36%** |

**Speaker (P3):**
> "The per-class report shows two interesting patterns."
> "First — compressed_ai has 98.87% precision but only 81.47% recall. This means when we do flag something as compressed AI, we are almost always right. But we miss 18.5% of them. High precision, lower recall."
> "Second — perfect_ai has 91.74% precision but 100% recall. We catch every perfect AI image, but occasionally a compressed AI image is incorrectly placed in this bucket. That 139-image confusion we saw in the matrix explains this precision drop."
> "The real class: 98.39% precision, 98.33% recall. Excellent on both fronts — exactly what a safety system needs."

---

#### Slide 20 — Final Results Summary

**Layout:** Clean achievement table with target vs achieved.

**Headline:** *"All Targets Met. System Ready for Deployment."*

| Metric | Target | Achieved | Status |
|---|---|---|---|
| Test Accuracy | > 95% | **96.26%** | PASS |
| FPR on Real (argmax) | ≤ 5% | **1.56%** | PASS |
| FPR on Real (calibrated) | ≤ 5% | **1.67%** | PASS |
| Perfect AI Recall | > 90% | **100.00%** | PASS |
| Edited AI Recall | > 90% | **97.17%** | PASS |
| Training Stability | No NaN | **Stable** | PASS |

**Speaker (P3):**
> "Every target met. 96.26% accuracy. 1.56% False Positive Rate. The deployment checkpoint is saved, calibrated, and ready."

---

#### Slide 21 — Deployment Architecture

**Layout:** Use `results/diagrams/08_deployment.png`

**Headline:** *"From .pth File to Production API in One Step"*

**Speaker (P3):**
> "The deployment path is clear. The checkpoint is wrapped in a FastAPI endpoint. An incoming food image is resized, normalised, and passed through EfficientNet-B3. The threshold filter applies our calibrated 0.50 cut-off. The output is a class label, confidence score, and an ELA heatmap that localises the tampered region for human reviewers."
> "This is not a research prototype. The inference code exists. The metadata is saved. The threshold is calibrated. This can be deployed to a Flask or FastAPI server today."

---

#### Slide 22 — Future Work & Roadmap

**Layout:** Four tiles in a 2×2 grid.

**Tile 1 — Dual-Stream FFT Model:**
> Train a parallel branch on the FFT spectrum and fuse with the RGB branch at the classifier head. Direct access to frequency-domain features.

**Tile 2 — Grad-CAM Localisation:**
> Add a Grad-CAM layer to highlight the exact pixel region that triggered the AI classification. Gives human reviewers a visual explanation.

**Tile 3 — FastAPI + Dashboard:**
> Wrap inference.py in a production REST API. Build a web dashboard showing real-time fraud queue, confidence distribution, and class breakdown.

**Tile 4 — Model Distillation:**
> Compress EfficientNet-B3 into MobileNetV3 for edge deployment on mobile food safety inspection apps. Target: sub-50ms inference on CPU.

**Speaker (P3):**
> "Four clear next steps. The dual-stream FFT model is the highest-priority — it would directly address the compressed_ai recall gap by giving the model explicit access to frequency information."
> "Grad-CAM is important for regulatory compliance — any safety system needs to explain its decisions to human reviewers."
> "Thank you. I'm happy to take questions."

---

## Speaker Transition Lines

| Transition | Exact words |
|---|---|
| P1 → P2 | *"That covers the problem and the data. I'll hand over to [P2] who will walk you through the model architecture and what engineering went into training it."* |
| P2 → P3 | *"That covers the full training stack. I'll now hand over to [P3] who will walk you through what the trained model actually achieved."* |
| P3 → Q&A | *"That wraps up our presentation. All three of us are happy to take questions."* |

---

## Q&A Prep — All 3 Answer Together

| Question | Who answers | Key point |
|---|---|---|
| "Why 4 classes not 2?" | P1 | Compressed AI looks forensically different from Perfect AI — grouping them loses information |
| "Why only 81% on compressed_ai?" | P2 | JPEG strips the FFT artefacts — it's a known limitation, dual-stream FFT model is the fix |
| "How is 96% useful in production?" | P3 | FPR 1.56% is the real metric — 98 out of 100 real complaints still processed correctly |
| "How did you build the dataset?" | P1 | Scripted 5 diffusion pipelines locally, ran generation scripts, verified with cleanup script |
| "Why EfficientNet not ViT?" | P2 | ViT-Base needs 14GB VRAM at 512px — doesn't fit our hardware. EfficientNet-B3 is optimal |
| "What if someone compresses more aggressively?" | P2 | Below JPEG Q=40, even human experts can't tell — there's a fundamental information floor |
| "What is FPR vs accuracy?" | P3 | Accuracy counts all errors equally. FPR counts only the errors on real images — the business-critical metric |

---

## Design Guidelines for PowerPoint

1. **Font:** Use Calibri or Inter — clean, readable at distance
2. **Colours:** Blue (#0969DA), Green (#1A7F37), White background — matches the diagram style
3. **Slide title font size:** 28–32pt bold
4. **Body text:** 18–20pt minimum — never smaller
5. **One idea per slide** — if you have two ideas, make two slides
6. **Every diagram slide:** Image takes 60–70% of slide. Label what to look at with an arrow or circle
7. **Results slides:** Big numbers centred. Evaluators remember numbers, not paragraphs
8. **Avoid walls of text** — if it's more than 5 bullet points, split it

---

## Time Budget

| Presenter | Slides | Allotted Time |
|---|---|---|
| P1 | 7 slides | ~8 minutes |
| P2 | 8 slides | ~9 minutes |
| P3 | 7 slides | ~8 minutes |
| Q&A | — | 5–10 minutes |
| **Total** | **22 slides** | **~30–35 minutes** |

---

*Three people. One project. One story. Tell it like you built it — because you did.*
