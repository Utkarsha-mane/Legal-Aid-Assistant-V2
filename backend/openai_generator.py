import openai
from typing import Dict, Any, List
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OpenAILegalAnswerGenerator:
    """
    Generates grounded answers using OpenAI's GPT-4 model.
    Implements strict prompt engineering to prevent hallucinations.
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI generator.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
            model: Model name (gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview, gpt-4-1106-preview)
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        # Set the API key
        openai.api_key = self.api_key
        
        # Model configuration
        self.model = model
        
        # Available models
        self.available_models = {
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-4-1106": "gpt-4-1106-preview",
            "gpt-4-0125": "gpt-4-0125-preview",
            "gpt-3.5-turbo": "gpt-3.5-turbo"
        }
        
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
8. Always provide specific references to the source material when answering

You are designed to help users understand legal documents, but ONLY based on what is explicitly stated in the provided context."""
        
        logger.info(f"Initialized OpenAI generator with model: {self.model}")
    
    def generate_answer(self, query: str, context: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate grounded answer from query and retrieved context.
        
        Args:
            query: User's question
            context: Retrieved document chunks
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with answer and metadata
        """
        # Validate inputs
        if not query or not query.strip():
            return {
                'success': False,
                'error': 'Empty query provided',
                'answer': 'Please provide a valid question.'
            }
        
        if not context or context.strip() == "":
            return {
                'success': True,
                'answer': "The information is not available in the uploaded document.",
                'grounded': False
            }
        
        try:
            # Construct messages for ChatGPT
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""Context from the legal document:

{context}

---

Question: {query}

Please provide a detailed answer based strictly on the context above. If the information is not in the context, say so explicitly."""}
            ]
            
            logger.info(f"Generating answer for query: {query[:50]}...")
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent, factual responses
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            
            # Get usage stats
            usage = response.usage
            
            logger.info(f"Generated answer successfully. Tokens used: {usage.total_tokens}")
            
            return {
                'success': True,
                'answer': answer,
                'grounded': True,
                'model': self.model,
                'tokens_used': {
                    'prompt': usage.prompt_tokens,
                    'completion': usage.completion_tokens,
                    'total': usage.total_tokens
                },
                'context_length': len(context)
            }
            
        except openai.error.AuthenticationError:
            error_msg = "Invalid OpenAI API key. Please check your API key."
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'answer': error_msg
            }
        
        except openai.error.RateLimitError:
            error_msg = "OpenAI API rate limit exceeded. Please try again later."
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'answer': error_msg
            }
        
        except openai.error.InvalidRequestError as e:
            error_msg = f"Invalid request: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'answer': error_msg
            }
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'answer': f"Error: {str(e)}"
            }
    
    def generate_summary(self, chunks: List[str], max_chunks: int = 10) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the document.
        
        Args:
            chunks: List of document chunks
            max_chunks: Maximum number of chunks to use
            
        Returns:
            Dictionary with summary and metadata
        """
        # Limit chunks to avoid token overflow
        text_chunks = chunks[:max_chunks]
        combined_text = "\n\n".join(text_chunks)
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"""Please provide a comprehensive summary of the following legal document.

Include:
1. Main parties involved
2. Type of case/document
3. Key legal issues and arguments
4. Important dates and events
5. Final outcome or current status (if available)
6. Significant legal precedents or statutes cited

Document text:

{combined_text}

Provide a well-structured summary:"""}
            ]
            
            logger.info("Generating document summary...")
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=1500,
                temperature=0.2,
                top_p=0.95
            )
            
            summary = response.choices[0].message.content.strip()
            usage = response.usage
            
            logger.info(f"Summary generated. Tokens used: {usage.total_tokens}")
            
            return {
                'success': True,
                'summary': summary,
                'tokens_used': {
                    'prompt': usage.prompt_tokens,
                    'completion': usage.completion_tokens,
                    'total': usage.total_tokens
                },
                'chunks_processed': len(text_chunks)
            }
            
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'summary': error_msg
            }
    
    def check_api_connection(self) -> bool:
        """
        Test the OpenAI API connection.
        
        Returns:
            True if connection successful
        """
        try:
            # Simple test call
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            logger.info("OpenAI API connection successful")
            return True
        except Exception as e:
            logger.error(f"OpenAI API connection failed: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """
        List available OpenAI models.
        
        Returns:
            List of model IDs
        """
        try:
            models = openai.Model.list()
            model_ids = [model.id for model in models.data if 'gpt' in model.id]
            logger.info(f"Available models: {model_ids}")
            return model_ids
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def test_generation(self) -> Dict[str, Any]:
        """
        Test the generation capability with a simple query.
        
        Returns:
            Test result dictionary
        """
        test_context = """This is a test legal document. 
        
Case Name: Smith v. Johnson
Court: Supreme Court
Date: January 15, 2024

The plaintiff filed a motion for summary judgment, which was denied by the lower court."""
        
        test_query = "What motion did the plaintiff file?"
        
        logger.info("Running generation test...")
        result = self.generate_answer(test_query, test_context)
        
        return {
            "test_passed": result.get('success', False),
            "result": result
        }