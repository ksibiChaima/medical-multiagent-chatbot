import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict

class BrainTumorModel(torch.nn.Module):
    """Modèle CNN simplifié - adaptez selon VOTRE modèle"""
    def __init__(self):
        super().__init__()
        # Remplacez par votre architecture
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 56 * 56, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MLAgent:
    """Agent pour tumeurs cérébrales"""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = BrainTumorModel()
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✓ Modèle brain tumor chargé")
        except Exception as e:
            print(f"⚠️ Erreur: {e}")
            self.demo_mode = True
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.classes = ["Glioma", "Meningioma", "Pituitary", "Pas de tumeur"]
    
    def analyze(self, image_path: str) -> Dict:
        """Analyse IRM cérébrale"""
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        result = self.classes[predicted.item()]
        conf = confidence.item()
        
        return {
            "agent": "ML Agent (Brain Tumor)",
            "tumor_type": result,
            "confidence": f"{conf*100:.1f}%",
            "recommendation": "Consulter neurologue" if result != "Pas de tumeur" else "✓ Pas d'anomalie",
            "all_scores": {
                cls: f"{prob.item()*100:.1f}%"
                for cls, prob in zip(self.classes, probabilities[0])
            }
        }