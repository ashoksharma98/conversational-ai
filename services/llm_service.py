# services/llm_service.py
import logging
import asyncio
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
    
    async def generate_response(self, text: str, progress_callback=None) -> str:
        """
        Generate response from text using Gemini
        
        Args:
            text: Input text (transcription)
            progress_callback: Optional callback for progress updates
            
        Returns:
            str: Generated response
        """
        if self.model is None:
            self.load_model()
        
        try:
            logger.info(f"Generating response for: {text}")

            # Create a task to send progress updates while generating
            async def send_progress():
                while True:
                    await asyncio.sleep(3)  # Send update every 3 seconds
                    if progress_callback:
                        await progress_callback()

            # Start progress task
            progress_task = asyncio.create_task(send_progress())

            try:
                # Generate response (run in thread pool to avoid blocking)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    self.model.generate_content,
                    text
                )
                generated_text = response.text.strip()
            finally:
                # Cancel progress task
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

            logger.info(f"Generated response: {generated_text}")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
