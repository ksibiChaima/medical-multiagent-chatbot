import os
from dotenv import load_dotenv
from agents.router import RouterAgent
from agents.dl_agent import DLAgent
from agents.ml_agent import MLAgent
from agents.rag_agent import RAGAgent

class MedicalChatbot:
    """Orchestrateur principal du chatbot"""
    
    def __init__(self):
        # Charger les variables d'environnement
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        print("🏥 Initialisation du Medical Chatbot...")
        
        # Initialiser les agents
        self.router = RouterAgent(api_key)
        self.dl_agent = DLAgent(os.getenv("LUNG_MODEL_PATH", "models/lung_model.pth"))
        self.ml_agent = MLAgent(os.getenv("BRAIN_MODEL_PATH", "models/brain_model.pth"))
        self.rag_agent = RAGAgent(api_key)
        
        print("✓ Tous les agents sont prêts!\n")
    
    def process_query(self, query: str, image_path: str = None):
        """Traite une requête utilisateur"""
        
        print(f"\n{'='*60}")
        print(f"QUESTION: {query}")
        if image_path:
            print(f"IMAGE: {image_path}")
        print(f"{'='*60}\n")
        
        # 1. Router détermine l'agent
        has_image = image_path is not None
        agent_name = self.router.route(query, has_image)
        
        print(f"🎯 Routage → {agent_name.upper()}\n")
        
        # 2. Exécuter l'agent approprié
        try:
            if agent_name == "dl_agent":
                if not image_path:
                    return {"error": "DL Agent nécessite une image"}
                result = self.dl_agent.analyze(image_path)
            
            elif agent_name == "ml_agent":
                if not image_path:
                    return {"error": "ML Agent nécessite une image"}
                result = self.ml_agent.analyze(image_path)
            
            else:  # rag_agent
                result = self.rag_agent.answer(query)
            
            # 3. Afficher le résultat
            self._display_result(result)
            return result
        
        except Exception as e:
            error_msg = f"❌ Erreur: {str(e)}"
            print(error_msg)
            return {"error": error_msg}
    
    def _display_result(self, result: dict):
        """Affiche joliment le résultat"""
        print(f"\n📊 RÉSULTAT:")
        print(f"{'─'*60}")
        for key, value in result.items():
            if key != "all_scores":
                print(f"{key.upper()}: {value}")
        
        if "all_scores" in result:
            print(f"\nSCORES DÉTAILLÉS:")
            for cls, score in result["all_scores"].items():
                print(f"  • {cls}: {score}")
        print(f"{'─'*60}\n")

# Point d'entrée
if __name__ == "__main__":
    chatbot = MedicalChatbot()
    
    # TESTS
    print("\n" + "="*60)
    print("🧪 MODE TEST")
    print("="*60)
    
    # Test 1: Question textuelle
    chatbot.process_query("Quels sont les symptômes du diabète?")
    
    # Test 2: Analyse d'image (décommentez quand vous avez des images)
    # chatbot.process_query(
    #     "Analyse cette image de poumon",
    #     image_path="data/test_images/lung_xray.jpg"
    # )
    
    # Test 3: IRM cérébrale
    # chatbot.process_query(
    #     "Identifie le type de tumeur dans cette IRM",
    #     image_path="data/test_images/brain_mri.jpg"
    # )