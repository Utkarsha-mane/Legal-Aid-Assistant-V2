import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LegalAnswerGenerator:
    """
    Generates grounded answers using Google Gemini via Vertex AI.
    Implements strict prompt engineering to prevent hallucinations.
    """

    def __init__(self, model: str = "gemini-1.5-flash"):
        """
        Initialize Vertex AI-based LLM generator.

        Args:
            model: Gemini model name available on Vertex AI (default: gemini-1.5-flash)
        """
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not project_id:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT not found. "
                "Please set it in your backend/.env file:\n"
                "  GOOGLE_CLOUD_PROJECT=your-gcp-project-id\n"
                "  GOOGLE_CLOUD_LOCATION=us-central1\n"
                "Also ensure Application Default Credentials are configured:\n"
                "  gcloud auth application-default login"
            )

        # Import Vertex AI SDK
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel, GenerationConfig
        except ImportError:
            raise ImportError(
                "google-cloud-aiplatform is not installed. "
                "Run: pip install google-cloud-aiplatform"
            )

        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)

        self.model_name = model
        self.project_id = project_id
        self.location = location

        self.system_instruction = (
            "You are a Legal Aid Assistant. Your role is to answer legal questions "
            "based ONLY on the provided document context.\n\n"
            "STRICT RULES:\n"
            "1. Answer ONLY from the provided context\n"
            "2. If the answer is not in the context, respond EXACTLY: "
            "\"The information is not available in the uploaded document.\"\n"
            "3. Do NOT use external legal knowledge\n"
            "4. Quote relevant parts when possible\n"
            "5. Be precise and cite case names/sections when available\n"
            "6. Maintain legal terminology from the source\n"
            "7. If context is insufficient for a complete answer, state what is "
            "available and what is missing\n\n"
            "Context will be provided between [CONTEXT START] and [CONTEXT END] markers."
        )

        self.model = GenerativeModel(
            model_name=model,
            system_instruction=self.system_instruction,
        )

        self._GenerationConfig = GenerationConfig

        print(f"[OK] Vertex AI Gemini model initialized: {model} (project={project_id}, location={location})")

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
                "success": True,
                "answer": "The information is not available in the uploaded document.",
                "grounded": False,
            }

        user_message = (
            f"[CONTEXT START]\n{context}\n[CONTEXT END]\n\n"
            f"User Question: {query}\n\n"
            "Answer (based strictly on the context above):"
        )

        try:
            print(f"[INFO] Sending request to Vertex AI Gemini ({self.model_name})...")

            generation_config = self._GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
                top_p=0.9,
            )

            response = self.model.generate_content(
                user_message,
                generation_config=generation_config,
            )

            answer = response.text.strip()
            print(f"[OK] Answer generated ({len(answer)} chars)")

            return {
                "success": True,
                "answer": answer,
                "grounded": True,
                "model": self.model_name,
            }

        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Vertex AI generation failed: {error_msg}")

            if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                friendly = (
                    "Permission denied. Ensure your service account has the "
                    "'Vertex AI User' role on project: " + self.project_id
                )
            elif "quota" in error_msg.lower() or "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                friendly = "Vertex AI quota exceeded. Please wait a moment and try again."
            elif "not found" in error_msg.lower() or "404" in error_msg:
                friendly = (
                    f"Model '{self.model_name}' not found on Vertex AI in region '{self.location}'. "
                    "Try 'gemini-1.5-flash' or 'gemini-1.5-pro', or change GOOGLE_CLOUD_LOCATION."
                )
            elif "credentials" in error_msg.lower() or "UNAUTHENTICATED" in error_msg:
                friendly = (
                    "Authentication failed. Run: gcloud auth application-default login\n"
                    "Or set GOOGLE_APPLICATION_CREDENTIALS to your service account key path."
                )
            else:
                friendly = f"Vertex AI error: {error_msg}"

            return {
                "success": False,
                "error": friendly,
                "answer": friendly,
            }

    def check_model_availability(self) -> bool:
        """
        Verify that Vertex AI credentials are valid and the model is accessible.

        Returns:
            True if available, False otherwise
        """
        try:
            generation_config = self._GenerationConfig(max_output_tokens=5)
            self.model.generate_content("ping", generation_config=generation_config)
            return True
        except Exception as e:
            print(f"[ERROR] Vertex AI availability check failed: {e}")
            return False