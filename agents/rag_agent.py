from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from typing import Dict
import os

class RAGAgent:
    """Agent RAG simple sans Neo4j"""
    
    def __init__(self, api_key: str, docs_folder: str = "data/medical_docs"):
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.3)
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Charger les documents
        self.vectorstore = self._load_documents(docs_folder)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un assistant médical expert. Utilise le contexte fourni pour répondre.

CONTEXTE:
{context}

RÈGLES:
- Réponds de manière précise et empathique
- Cite les sources si disponibles
- Recommande toujours de consulter un professionnel pour un diagnostic
- Si tu ne sais pas, dis-le clairement"""),
            ("human", "{question}")
        ])
    
    def _load_documents(self, docs_folder: str):
        """Charge les documents médicaux"""
        documents = []
        
        # Créer des documents de base si le dossier est vide
        if not os.path.exists(docs_folder) or not os.listdir(docs_folder):
            print("⚠️ Pas de documents trouvés, utilisation de connaissances de base")
            base_knowledge = [
                "Le diabète est une maladie chronique caractérisée par un taux élevé de sucre dans le sang.",
                "Les symptômes du diabète incluent: soif excessive, fatigue, vision floue.",
                "L'hypertension artérielle est définie par une pression supérieure à 140/90 mmHg.",
                "Les maladies cardiovasculaires sont la première cause de mortalité dans le monde."
            ]
            documents = [Document(page_content=text) for text in base_knowledge]
        else:
            # Charger les fichiers .txt du dossier
            for filename in os.listdir(docs_folder):
                if filename.endswith('.txt'):
                    with open(os.path.join(docs_folder, filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": filename}
                        ))
        
        # Splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(documents)
        
        # Créer le vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        print(f"✓ {len(splits)} chunks chargés dans le vectorstore")
        return vectorstore
    
    def answer(self, question: str) -> Dict:
        """Répond à une question médicale"""
        
        # Recherche de documents pertinents
        docs = self.vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Générer la réponse
        response = self.llm.invoke(
            self.prompt.format_messages(
                context=context,
                question=question
            )
        )
        
        return {
            "agent": "RAG Agent (Medical Q&A)",
            "answer": response.content,
            "sources": [doc.metadata.get("source", "Base de connaissances") for doc in docs],
            "confidence": "Basé sur documents médicaux"
        }

# Test
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = RAGAgent(os.getenv("OPENAI_API_KEY"))
    result = agent.answer("Quels sont les symptômes du diabète?")
    print(result)