# services/llm_service.py
import logging
import asyncio
import google.generativeai as genai
from google import genai as gemini

logger = logging.getLogger(__name__)

# class LLMService:
#     """LLM service using Gemini"""
#
#     def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
#         self.api_key = api_key
#         self.model_name = model_name
#         self.model = None
#         logger.info(f"LLM Service initialized with model: {model_name}")
#
#     def load_model(self):
#         """Configure and load Gemini model"""
#         try:
#             logger.info("Configuring Gemini API...")
#             genai.configure(api_key=self.api_key)
#
#             # Initialize the model
#             self.model = genai.GenerativeModel(self.model_name)
#             logger.info("Gemini model loaded successfully")
#         except Exception as e:
#             logger.error(f"Error loading Gemini model: {e}")
#             raise
#
#     async def generate_response(self, text: str, progress_callback=None) -> str:
#         """
#         Generate response from text using Gemini
#
#         Args:
#             text: Input text (transcription)
#             progress_callback: Optional callback for progress updates
#
#         Returns:
#             str: Generated response
#         """
#         if self.model is None:
#             self.load_model()
#
#         try:
#             logger.info(f"Generating response for: {text}")
#
#             # Create a task to send progress updates while generating
#             async def send_progress():
#                 while True:
#                     await asyncio.sleep(3)  # Send update every 3 seconds
#                     if progress_callback:
#                         await progress_callback()
#
#             # Start progress task
#             progress_task = asyncio.create_task(send_progress())
#
#             try:
#                 # Generate response (run in thread pool to avoid blocking)
#                 loop = asyncio.get_event_loop()
#                 response = await loop.run_in_executor(
#                     None,
#                     self.model.generate_content,
#                     text
#                 )
#                 generated_text = response.text.strip()
#             finally:
#                 # Cancel progress task
#                 progress_task.cancel()
#                 try:
#                     await progress_task
#                 except asyncio.CancelledError:
#                     pass
#
#             logger.info(f"Generated response: {generated_text}")
#             return generated_text
#
#         except Exception as e:
#             logger.error(f"Error generating response: {e}")
#             raise


class LLMService:
    def __init__(
            self,
            api_key: str,
            model_name: str = "gemini-2.5-flash",
            temperature: float = 0.7,
            system_instruction: str | None = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.system_instruction = system_instruction
        self.model = None

    def _get_model(self):
        if self.model is None:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    top_p=0.95,
                    max_output_tokens=8192,
                ),
                system_instruction=self.system_instruction or
                                   "You are a friendly, natural voice assistant. Respond concisely and conversationally."
            )
        return self.model

    # async def stream_response(self, user_text: str):
    #     """
    #     Streams tokens one-by-one from Gemini.
    #     Properly handles sync generator in async context.
    #     """
    #     if not user_text.strip():
    #         return
    #
    #     model = self._get_model()
    #     logger.info(f"LLM ← {user_text!r}")
    #
    #     try:
    #         # Run the blocking generate_content in a thread pool
    #         loop = asyncio.get_event_loop()
    #
    #         # Create the stream in executor
    #         stream = await loop.run_in_executor(
    #             None,
    #             lambda: model.generate_content([user_text], stream=True)
    #         )
    #
    #         chunk_count = 0
    #
    #         # Iterate over chunks (this is synchronous, so wrap each iteration)
    #         for chunk in stream:
    #             chunk_count += 1
    #             logger.debug(f"Raw chunk {chunk_count}: {chunk.text!r}")
    #
    #             if chunk.text:
    #                 yield chunk.text
    #
    #             # Yield control to event loop periodically
    #             await asyncio.sleep(0)
    #
    #         logger.info(f"Stream finished with {chunk_count} chunks")
    #
    #     except Exception as e:
    #         logger.error(f"Streaming failed: {e}", exc_info=True)
    #         yield "Oops, my thoughts got stuck. Let's try that again?"

    async def stream_response(self, user_text: str):
        if not user_text.strip():
            logger.warning("Empty input to LLM")
            return

        model = self._get_model()
        logger.info(f"LLM ← {user_text!r}")

        def _generate_sync():
            try:
                logger.info("Starting Gemini generate_content(stream=True)")
                response = model.generate_content(
                    [user_text],
                    stream=True,
                    safety_settings={
                        # Be explicit — unblock everything that is safe for a voice assistant
                        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    }
                )

                chunk_no = 0
                for chunk in response:
                    chunk_no += 1
                    logger.info(f"Chunk {chunk_no}: {chunk.text!r}")
                    if chunk.text:
                        yield chunk.text

                logger.info(f"Gemini streaming finished — {chunk_no} chunks")
                if chunk_no == 0:
                    yield "I'm not sure how to answer that."

            except Exception as e:
                logger.error(f"Gemini EXCEPTION: {e!r}", exc_info=True)
                # Re-raise so we see it
                raise

        try:
            # Run the entire sync generator in a thread and yield tokens immediately
            for token in await asyncio.to_thread(list, _generate_sync()):
                yield token

        except Exception as e:
            logger.error(f"LLM streaming completely failed: {e}")
            yield "Sorry, I couldn't think of an answer right now."


class GeminiService:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",  # Updated to real 2025 model (fastest); fallback: "gemini-1.5-flash"
        temperature: float = 0.7,
        system_instruction: str | None = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.system_instruction = system_instruction
        self.client = None

    def _get_client(self):
        if self.client is None:
            # New client setup (faster cold start)
            self.client = gemini.Client(api_key=self.api_key)
            logger.info(f"New Gemini client initialized for {self.model_name}")
        return self.client

    async def stream_response(self, user_text: str):
        """
        Streams buffered text chunks (3-5 words) using new Google API.
        Yields: Natural sentence fragments for TTS (e.g., "The capital is" → then "New Delhi.").
        """
        if not user_text.strip():
            logger.warning("Empty input to LLM")
            return

        client = self._get_client()
        logger.info(f"LLM ← {user_text!r} (using new genai.Client API)")

        try:
            # New API call: Direct streaming with async support
            response = client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=[user_text],
                config=gemini.types.GenerateContentConfig(
                    system_instruction="You are a friendly, natural voice assistant. Respond concisely and conversationally.",
                    temperature=self.temperature,
                    top_p=0.95,
                    max_output_tokens=8192,
                ),
            )

            full_text = ""
            buffer = ""
            word_count = 0
            chunk_count = 0

            # Native async iteration (no to_thread needed!)
            async for chunk in await response:
                if not chunk.text:
                    continue

                # Accumulate raw tokens
                full_text += chunk.text
                words = chunk.text.split()  # Simple word split
                for word in words:
                    buffer += word + " "
                    word_count += 1

                    # Smart chunking: Yield every 3-5 words OR on punctuation (natural pauses)
                    if word_count >= 3 or any(p in word for p in '.!?'):
                        chunk_count += 1
                        logger.debug(f"LLM chunk {chunk_count}: {buffer.strip()!r}")
                        yield buffer.strip()  # Send buffered chunk to TTS
                        buffer = ""
                        word_count = 0

                        # Yield control for real-time (prevents blocking)
                        await asyncio.sleep(0)

            # Final buffer if any
            if buffer.strip():
                yield buffer.strip()

            logger.info(f"LLM streaming done: {full_text!r} ({chunk_count} chunks)")

        except Exception as e:
            logger.error(f"New Gemini API failed: {e}", exc_info=True)
            yield "Sorry, I couldn't respond right now—let's try again."
