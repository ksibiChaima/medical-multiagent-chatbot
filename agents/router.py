from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class RouterAgent:
    """Détermine quel agent utiliser"""
    
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un routeur médical. Détermine quel agent utiliser:

RÈGLES:
- Si la question mentionne une IMAGE + "poumon/lung/chest" → réponds "dl_agent"
- Si la question mentionne une IMAGE + "cerveau/brain/tumor/IRM/MRI" → réponds "ml_agent"  
- Sinon (question textuelle) → réponds "rag_agent"

Réponds UNIQUEMENT avec: dl_agent, ml_agent, ou rag_agent"""),
            ("human", "Question: {query}\nImage fournie: {has_image}")
        ])
    
    def route(self, query: str, has_image: bool = False) -> str:
        """Retourne le nom de l'agent à utiliser"""
        response = self.llm.invoke(
            self.prompt.format_messages(
                query=query,
                has_image="Oui" if has_image else "Non"
            )
        )
        
        agent = response.content.strip().lower()
        
        # Sécurité
        if agent not in ["dl_agent", "ml_agent", "rag_agent"]:
            return "rag_agent"
        
        return agent

# Test
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    router = RouterAgent(os.getenv("OPENAI_API_KEY"))
    
    # Tests
    print(router.route("J'ai une image de poumon à analyser", has_image=True))
    # → dl_agent
    
    print(router.route("Analyse cette IRM cérébrale", has_image=True))
    # → ml_agent
    
    print(router.route("Quels sont les symptômes du diabète?"))
    # → rag_agent