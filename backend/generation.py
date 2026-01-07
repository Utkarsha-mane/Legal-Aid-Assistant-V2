import requests
import json
from typing import Dict, Any


class LegalAnswerGenerator:
    """
    Generates grounded answers using Ollama's LLaMA3 model.
    Implements strict prompt engineering to prevent hallucinations.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3:8b"):
        """
        Initialize LLM generator.
        
        Args:
            base_url: Ollama API endpoint
            model: Generation model name
        """
        self.base_url = base_url
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        
        # System prompt enforces strict grounding
        self.system_prompt = """You are a Legal Aid Assistant. Your role is to answer legal questions based ONLY on the provided document context.

STRICT RULES:
1. Answer ONLY from the provided context
2. If the answer is not in the context, respond EXACTLY: "The information is not available in the uploaded document."
3. Do NOT use external legal knowledge
4. Quote relevant parts when possible
5. Be precise and cite case names/sections when available
6. Maintain legal terminology from the source
7. If context is insufficient for a complete answer, state what is available and what is missing

Context will be provided between [CONTEXT START] and [CONTEXT END] markers."""
    
    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """
        Generate grounded answer from query and retrieved context.
        
        Args:
            query: User's question
            context: Retrieved document chunks
            
        Returns:
            Dictionary with answer and metadata
        """
        if not context or context.strip() == "":
            return {
                'success': True,
                'answer': "The information is not available in the uploaded document.",
                'grounded': False
            }
        
        # Construct prompt with explicit grounding instructions
        prompt = f"""{self.system_prompt}

[CONTEXT START]
{context}
[CONTEXT END]

User Question: {query}

Answer (based strictly on the context above):"""
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for deterministic, grounded responses
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            print("Generating answer from LLaMA3...")
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                return {
                    'success': True,
                    'answer': answer,
                    'grounded': True,
                    'model': self.model
                }
            else:
                return {
                    'success': False,
                    'error': f"Generation failed: {response.status_code}",
                    'answer': "Error generating response"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'answer': f"Error: {str(e)}"
            }
    
    def check_model_availability(self) -> bool:
        """
        Verify that the generation model is available in Ollama.
        
        Returns:
            True if model is available
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(m['name'].startswith(self.model) for m in models)
        except Exception as e:
            print(f"Error checking model availability: {e}")
        return False