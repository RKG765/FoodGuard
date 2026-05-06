"""Quick test to verify model predictions on test images."""
import torch, timm, json
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

meta = json.load(open("checkpoints/food_detector/metadata.json"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model(meta["model"], pretrained=False, num_classes=meta["num_classes"])
model.load_state_dict(torch.load("checkpoints/food_detector/food_ai_detector.pth", map_location=device))
model.to(device).eval()

tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_images = [
    ("dataset_4class/test/real/real_00000.jpg", "TRUE: real"),
    ("dataset_4class/test/real/real_00001.jpg", "TRUE: real"),
    ("dataset_4class/test/perfect_ai/perfect_ai_00000.png", "TRUE: perfect_ai"),
    ("dataset_4class/test/perfect_ai/perfect_ai_00001.png", "TRUE: perfect_ai"),
    ("dataset_4class/test/edited_ai/edited_ai_00000.png", "TRUE: edited_ai"),
    ("dataset_4class/test/compressed_ai/compressed_ai_00000.png", "TRUE: compressed_ai"),
]

class_names = meta["class_names"]
threshold = meta["threshold"]
real_idx = meta.get("real_class_index", class_names.index("real"))
print(f"Class names: {class_names}")
print(f"Real index: {real_idx}")
print(f"Threshold: {threshold}")
print()

for path, label in test_images:
    img = Image.open(path).convert("RGB")
    inp = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    prob_real = probs[real_idx].item()
    
    # Threshold decision
    if prob_real > threshold:
        decision = "real"
    else:
        ai_indices = [i for i in range(len(class_names)) if i != real_idx]
        ai_probs = [(i, probs[i].item()) for i in ai_indices]
        best_ai_idx, _ = max(ai_probs, key=lambda x: x[1])
        decision = class_names[best_ai_idx]

    status = "OK" if decision in label else "WRONG"
    print(f"[{status:5s}] {label:30s} => {decision:15s}  P(real)={prob_real:.4f}  argmax={class_names[pred_idx]}")
    for name, p in zip(class_names, probs):
        print(f"         {name}: {p:.6f}")
    print()
