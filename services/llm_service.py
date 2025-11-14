# services/llm_service.py
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

class LLMService:
    """LLM service using Gemini"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        logger.info(f"LLM Service initialized with model: {model_name}")
    
    def load_model(self):
        """Configure and load Gemini model"""
        try:
            logger.info("Configuring Gemini API...")
            genai.configure(api_key=self.api_key)
            
            # Initialize the model
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("Gemini model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Gemini model: {e}")
            raise
    
    async def generate_response(self, text: str) -> str:
        """
        Generate response from text using Gemini
        
        Args:
            text: Input text (transcription)
            
        Returns:
            str: Generated response
        """
        if self.model is None:
            self.load_model()
        
        try:
            logger.info(f"Generating response for: {text}")
            
            # Generate response
            response = self.model.generate_content(text)
            generated_text = response.text.strip()
            
            logger.info(f"Generated response: {generated_text}")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
