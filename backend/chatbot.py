# backend/chatbot.py
import logging
import os
import torch
from ctransformers import AutoModelForCausalLM
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChatBot")


class ChatBot:
    def __init__(self, model_path: str = None, model_name: str = None,
                 max_threads: int = 4, use_cuda: bool = True, summarizer=None):
        """
        ChatBot wrapper around local mistral/neoX models.
        - Always tries to answer using provided PDF context.
        - Falls back to general answer only if no context is available.
        - Prioritizes short, clear, coherent answers.
        """
        # Resolve path
        model_path = model_path or model_name
        if not model_path:
            raise ValueError("You must provide either model_path or model_name.")

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
            logger.warning("GPU unavailable, falling back to CPU...")
            os.environ["CT_USE_CUDA"] = "0"
            self.llm = AutoModelForCausalLM.from_pretrained(model_path, model_type=self.model_type)
            self.use_cuda = False

        logger.info(f"{self.model_type} model loaded successfully.")
        self.max_context_len = 800
        self.chunk_size = 200
        self.chunk_overlap = 50

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks for context handling."""
        chunks, start = [], 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _deduplicate_sentences(self, text: str) -> str:
        """Remove duplicate sentences for clarity."""
        seen, result = set(), []
        for sentence in text.split(". "):
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                result.append(sentence)
        return ". ".join(result)

    def ask(self, query: str, context: str = "", max_new_tokens: int = 256) -> str:
        """
        Generate answer.
        - Always uses PDF context if provided.
        - If context empty, answer more generally.
        """
        if len(context) > self.max_context_len:
            context = context[-self.max_context_len:]

        use_context = bool(context.strip())
        if not use_context:
            prompt = (
                f"You are HAS, an AI assistant.\n"
                f"Question: {query}\n"
                f"Answer clearly and concisely:"
            )
        else:
            prompt = (
                f"You are HAS, an AI assistant answering strictly based on the given PDF context.\n"
                f"Tone: professional, concise, clear.\n"
                f"Do not make things up — if the context lacks the answer, say so.\n\n"
                f"Context: {context}\n"
                f"Question: {query}\n"
                f"Answer:"
            )

        try:
            output = self.llm(prompt, max_new_tokens=max_new_tokens,
                              stop=["\nQuestion:", "\nContext:"])
            if isinstance(output, list):
                output = " ".join(str(x) for x in output)
            answer = self._deduplicate_sentences(str(output).strip())
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = "⚠️ Error generating response."

        if len(answer) > 800:
            answer = answer[:800] + "…"
        return answer

    def greet(self) -> str:
        return "Hello! I'm HAS, your PDF assistant. How can I help today?"
