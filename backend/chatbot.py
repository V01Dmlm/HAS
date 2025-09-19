# backend/chatbot.py
import logging
import os
import torch
from ctransformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChatBot")

class ChatBot:
    def __init__(self, model_path: str = None, model_name: str = None,
                 max_threads: int = 4, use_cuda: bool = True, summarizer=None):
        """
        Robust Mistral/GPT-NeoX wrapper for production.
        You can pass either:
          - model_path="full/path/to/model.gguf"
          - model_name="model.gguf"  # auto looks in ./models/
        """

        # Determine the actual path
        model_path = model_path or model_name
        if not model_path:
            raise ValueError("You must provide either model_path or model_name.")

        # If it’s not an absolute path, assume it's in ./models/
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), "models", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'")

        self.summarizer = summarizer
        self.use_cuda = use_cuda and torch.cuda.is_available()
        os.environ["CT_THREADS"] = str(max_threads)
        os.environ["CT_USE_CUDA"] = "1" if self.use_cuda else "0"

        logger.info(f"Loading model from '{model_path}' (GPU: {self.use_cuda})...")
        self.model_type = "mistral" if "mistral" in model_path.lower() else "gpt_neox"

        try:
            self.llm = AutoModelForCausalLM.from_pretrained(model_path, model_type=self.model_type)
        except RuntimeError:
            logger.warning("GPU unavailable or failed, falling back to CPU...")
            os.environ["CT_USE_CUDA"] = "0"
            self.llm = AutoModelForCausalLM.from_pretrained(model_path, model_type=self.model_type)
            self.use_cuda = False

        logger.info(f"{self.model_type} model loaded successfully.")
        self.max_context_len = 400 if self.model_type == "mistral" else 300
        self.chunk_size = 200
        self.chunk_overlap = 50

    def _chunk_text(self, text: str, max_chars=None, overlap=None):
        max_chars = max_chars or self.chunk_size
        overlap = overlap or self.chunk_overlap
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start += max_chars - overlap
        logger.debug(f"Text chunked into {len(chunks)} pieces")
        return chunks

    def greet(self) -> str:
        prompt = (
            "You are HAS, a friendly AI assistant helping with PDFs.\n"
            "Tone: casual, playful, slightly sarcastic.\n"
            "Give a short greeting as if talking to a friend.\n"
            "Response:"
        )
        try:
            greeting = self.llm(prompt, max_new_tokens=100, stop=["\n"])
            if isinstance(greeting, list):
                greeting = " ".join(str(g) for g in greeting).strip()
            else:
                greeting = str(greeting).strip()
            return greeting
        except Exception as e:
            logger.error(f"Failed to generate greeting: {e}")
            return "Hey! I'm HAS 😎 Ready to mess with some PDFs?"

    def ask(self, query: str, context: str = "", max_new_tokens: int = 256) -> str:
        if len(context) > self.max_context_len:
            context = context[-self.max_context_len:]

        context_chunks = self._chunk_text(context)
        answers = []

        for i, chunk in enumerate(context_chunks):
            prompt = (
                f"You are HAS, an AI assistant that answers questions based on the given context.\n"
                f"Tone: helpful, slightly sarcastic, playful.\n"
                f"Do not make up info; if unknown, admit it.\n"
                f"Context: {chunk}\n"
                f"Question: {query}\n"
                f"Answer concisely, clearly, with witty sarcasm:"
            )
            try:
                chunk_answer = self.llm(prompt, max_new_tokens=max_new_tokens,
                                        stop=["\nQuestion:", "\nContext:"])
                if isinstance(chunk_answer, list):
                    chunk_answer = " ".join(str(c) for c in chunk_answer).strip()
                else:
                    chunk_answer = str(chunk_answer).strip()
                answers.append(chunk_answer)
                logger.debug(f"Chunk {i+1}/{len(context_chunks)} processed")
            except Exception as e:
                logger.error(f"LLM call failed on chunk {i+1}: {e}")
                answers.append("⚠️ Error generating chunk response.")

        final_answer = " ".join(answers).strip()

        if self.summarizer and len(final_answer) > 1000:
            try:
                final_answer = self.summarizer(final_answer)
            except Exception as e:
                logger.warning(f"Summarizer failed: {e}")

        return final_answer
