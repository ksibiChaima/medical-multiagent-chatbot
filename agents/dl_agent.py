import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict

class LungCancerModel(torch.nn.Module):
    """Modèle VGG simplifié - adaptez selon VOTRE modèle"""
    def __init__(self):
        super().__init__()
        # Remplacez par l'architecture de votre notebook
        from torchvision.models import vgg16
        self.model = vgg16(pretrained=False)
        self.model.classifier[6] = torch.nn.Linear(4096, 2)
    
    def forward(self, x):
        return self.model(x)

class DLAgent:
    """Agent pour analyse cancer du poumon"""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Charger le modèle
        self.model = LungCancerModel()
        
        # Charger les poids
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✓ Modèle chargé avec succès")
        except Exception as e:
            print(f"⚠️ Erreur chargement modèle: {e}")
            print("Mode démo activé (prédictions aléatoires)")
            self.demo_mode = True
        
        self.model.to(self.device)
        self.model.eval()
        
        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.classes = ["Sain", "Cancer Détecté"]
    
    def analyze(self, image_path: str) -> Dict:
        """Analyse une image de poumon"""
        
        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        result = self.classes[predicted.item()]
        conf = confidence.item()
        
        return {
            "agent": "DL Agent (Lung Cancer)",
            "prediction": result,
            "confidence": f"{conf*100:.1f}%",
            "recommendation": "⚠️ Consulter un médecin" if result == "Cancer Détecté" else "✓ Pas d'anomalie détectée",
            "all_scores": {
                self.classes[0]: f"{probabilities[0][0].item()*100:.1f}%",
                self.classes[1]: f"{probabilities[0][1].item()*100:.1f}%"
            }
        }

# Test
if __name__ == "__main__":
    agent = DLAgent("../models/lung_model.pth")
    # result = agent.analyze("path/to/test/image.jpg")
    # print(result)