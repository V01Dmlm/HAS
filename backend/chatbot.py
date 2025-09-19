# backend/chatbot.py
import logging
import os
import torch
from ctransformers import AutoModelForCausalLM
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChatBot")


class ChatBot:
    """
    PDF-RAG-focused assistant:
    - Answers using only PDF context.
    - Falls back to generic model knowledge only for vague/general queries.
    """
    def __init__(self, model_path: str = None, model_name: str = None,
                 max_threads: int = 4, use_cuda: bool = True, summarizer=None):
        model_path = model_path or model_name
        if not model_path:
            raise ValueError("Provide model_path or model_name.")

        if not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), "models", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at '{model_path}'")

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
        self.max_context_len = 400 if self.model_type == "mistral" else 300
        self.chunk_size = 200
        self.chunk_overlap = 50

    # ----------------- Context Handling -----------------
    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _deduplicate_sentences(self, text: str) -> str:
        seen = set()
        result = []
        for sentence in text.split(". "):
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                result.append(sentence)
        return ". ".join(result)

    # ----------------- Core Logic -----------------
    def ask(self, query: str, context: str = "", max_new_tokens: int = 256) -> str:
        """
        Ask a question using only the provided PDF context.
        - If context is empty and query is general, allow model's generic knowledge.
        """
        # If context is empty but query is very vague, allow generic knowledge
        is_general_query = not context or len(context.strip()) < 50

        # If context exists, strictly prioritize it
        if context:
            if len(context) > self.max_context_len:
                context = context[-self.max_context_len:]
            chunks = self._chunk_text(context)
        else:
            chunks = [""]  # single empty chunk for generic fallback

        answers = []

        for i, chunk in enumerate(chunks):
            # Prompt construction
            prompt = (
                "You are a professional AI assistant (HAS) answering questions "
                "based ONLY on the provided context. Do not fabricate information. "
                "If the answer is unknown, say you cannot answer.\n"
                f"Context: {chunk}\n"
                f"Question: {query}\n"
                "Answer:"
            )

            # If general query and context is empty, allow general knowledge
            if is_general_query and not chunk.strip():
                prompt = (
                    "You are a professional AI assistant (HAS). Answer the question "
                    "using your general knowledge, in a concise and factual manner.\n"
                    f"Question: {query}\n"
                    "Answer:"
                )

            try:
                chunk_answer = self.llm(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    stop=["\nQuestion:", "\nContext:"]
                )
                # Handle list or string outputs
                if isinstance(chunk_answer, list):
                    chunk_answer = " ".join(str(c) for c in chunk_answer).strip()
                else:
                    chunk_answer = str(chunk_answer).strip()
                answers.append(chunk_answer)
            except Exception as e:
                logger.error(f"LLM call failed on chunk {i+1}: {e}")
                answers.append("⚠️ Error generating chunk response.")

        merged = " ".join(answers)
        merged = self._deduplicate_sentences(merged)

        # Summarize if very long
        if self.summarizer and len(merged) > 1000:
            try:
                merged = self.summarizer(merged)
            except Exception as e:
                logger.warning(f"Summarizer failed: {e}")

        # Trim final answer to avoid overly long responses
        if len(merged) > 800:
            merged = merged[:800] + "…"

        return merged.strip()

    def greet(self) -> str:
        return "Hello! I'm HAS, your PDF assistant. How can I help today?"
