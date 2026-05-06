"""
FoodGuard Inference Script
===========================

Load trained model and perform threshold-calibrated inference.

Usage:
    python inference.py path/to/image.jpg
"""

import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm


class FoodDetector:
    def __init__(self, checkpoint_dir="checkpoints/food_detector"):
        """Load trained model and metadata."""
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Load metadata
        with open(self.checkpoint_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        
        self.model_name = self.metadata["model"]
        self.num_classes = self.metadata["num_classes"]
        self.image_size = self.metadata["image_size"]
        self.threshold = self.metadata["threshold"]
        self.class_names = self.metadata["class_names"]
        
        # Load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=self.num_classes
        )
        self.model.load_state_dict(
            torch.load(self.checkpoint_dir / "food_ai_detector.pth",
                      map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded model: {self.model_name}")
        print(f"Threshold: {self.threshold:.3f}")
        print(f"Classes: {self.class_names}")
    
    @torch.no_grad()
    def predict(self, image_path):
        """Predict with threshold calibration.
        
        Returns:
            dict with: prediction, confidence, probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        
        # Real class index (ImageFolder sorts alphabetically:
        # compressed_ai=0, edited_ai=1, perfect_ai=2, real=3)
        real_idx = self.metadata.get("real_class_index",
                                     self.class_names.index("real"))
        prob_real = probs[real_idx].item()
        
        if prob_real > self.threshold:
            prediction = "real"
            confidence = prob_real
        else:
            # AI detected - choose highest AI class (skip real index)
            ai_indices = [i for i in range(len(self.class_names)) if i != real_idx]
            ai_probs = [(i, probs[i].item()) for i in ai_indices]
            best_ai_idx, best_ai_prob = max(ai_probs, key=lambda x: x[1])
            prediction = self.class_names[best_ai_idx]
            confidence = best_ai_prob
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {name: prob.item() 
                           for name, prob in zip(self.class_names, probs)},
            "is_fake": prediction != "real"
        }


def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load detector
    detector = FoodDetector()
    
    # Predict
    result = detector.predict(image_path)
    
    print("\n" + "=" * 60)
    print(f"Image: {image_path}")
    print("=" * 60)
    print(f"Prediction:  {result['prediction'].upper()}")
    print(f"Confidence:  {result['confidence']*100:.2f}%")
    print(f"Is Fake:     {'YES' if result['is_fake'] else 'NO'}")
    print("\nClass Probabilities:")
    for class_name, prob in result['probabilities'].items():
        bar = "█" * int(prob * 50)
        print(f"  {class_name:15s}: {prob*100:6.2f}% {bar}")
    print("=" * 60)


if __name__ == "__main__":
    main()
