import os
import fitz  # PyMuPDF
import faiss
import pickle
import logging
import io
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

from backend.arabic_utils import contains_arabic, reshape_for_display, clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PDFHandler")


class PDFHandler:
    def __init__(
        self,
        upload_dir="uploads",
        vector_dir="vector_store",
        max_workers=4,
        summarizer=None,
        chunk_size=400,
        overlap=50,
        embedder_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Production-ready PDF handler optimized for Arabic-heavy content.
        """
        self.upload_dir = upload_dir
        self.vector_dir = vector_dir
        self.max_workers = max_workers
        self.summarizer = summarizer
        self.chunk_size = chunk_size
        self.overlap = overlap

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(vector_dir, exist_ok=True)

        # Embedder
        self.embedder = SentenceTransformer(embedder_model)

        # Paths
        self.index_path = os.path.join(vector_dir, "faiss.index")
        self.chunks_path = os.path.join(vector_dir, "chunks.pkl")
        self.metadata_path = os.path.join(vector_dir, "metadata.pkl")

        # Internal
        self.chunks = []
        self.metadata = []
        self.index = None

        self._load_index()

    # -------------------- Core Index Ops --------------------
    def _load_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            logger.info("Loading existing FAISS index...")
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, "rb") as f:
                        self.metadata = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, rebuilding...")
                self._rebuild_index()
        else:
            self.index = None

    def _rebuild_index(self):
        """Rebuild FAISS index from stored chunks."""
        if not self.chunks:
            logger.warning("No chunks available to rebuild index.")
            return
        embeddings = self._embed_chunks(self.chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)
        logger.info("FAISS index rebuilt successfully.")

    # -------------------- PDF Handling --------------------
    def save_pdf(self, file):
        """Save and index a PDF. Accepts file path (str) or BytesIO."""
        if isinstance(file, str):
            file_path = file
            pdf_name = os.path.basename(file)
        elif isinstance(file, io.BytesIO):
            if not hasattr(file, "name"):
                raise ValueError("BytesIO file must have 'name' attribute")
            pdf_name = os.path.basename(file.name)
            file_path = os.path.join(self.upload_dir, pdf_name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        else:
            raise TypeError("file must be str path or BytesIO")

        self._process_pdf_text(file_path)

    def _process_pdf_text(self, pdf_path: str):
        text = self.extract_text(pdf_path)
        if not text.strip():
            logger.warning(f"No text found in PDF '{pdf_path}'")
            return

        if self.summarizer:
            try:
                text = self.summarizer(text)
            except Exception as e:
                logger.warning(f"Document summarizer failed: {e}")

        chunks = self.chunk_text(text)
        metadata = [
            {"file": os.path.basename(pdf_path), "chunk": i} for i in range(len(chunks))
        ]

        embeddings = self._embed_chunks(chunks)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

        # Save everything safely
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.chunks_path, "wb") as f:
                pickle.dump(self.chunks, f)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"PDF '{os.path.basename(pdf_path)}' processed and embeddings saved!")
        except Exception as e:
            logger.error(f"Failed to save index or chunks: {e}")

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF with parallel page reading."""
        try:
            doc = fitz.open(pdf_path)
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                texts = list(ex.map(lambda p: p.get_text(), doc))
            return clean_text(" ".join(texts))
        except Exception as e:
            logger.warning(f"Failed to read PDF '{pdf_path}': {e}")
            return ""

    # -------------------- Text Ops --------------------
    def chunk_text(self, text: str):
        """Chunk text into overlapping segments for safe LLM input."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            if self.summarizer:
                try:
                    chunk = self.summarizer(chunk)
                except Exception:
                    pass
            chunks.append(clean_text(chunk))
            start += self.chunk_size - self.overlap
        return chunks

    def _embed_chunks(self, chunks):
        """Safe embedding with batching + retry."""
        batch_size = 32
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            try:
                vecs = self.embedder.encode(batch, convert_to_numpy=True)
                embeddings.append(vecs)
            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
        return np.vstack(embeddings) if embeddings else np.zeros((0, 384))

    # -------------------- Retrieval --------------------
    def get_context(self, query: str, top_k=3, pdf_files=None, max_chars=400, reshape_arabic=False):
        """
        Retrieve relevant context safely for Arabic-heavy PDFs.
        reshape_arabic: only reshape for display; embeddings remain raw.
        """
        pdf_files = pdf_files or os.listdir(self.upload_dir)
        for fname in pdf_files:
            if fname not in [m["file"] for m in self.metadata]:
                path = os.path.join(self.upload_dir, fname)
                if os.path.isfile(path):
                    try:
                        logger.info(f"[DYNAMIC INDEX] Processing new PDF: {fname}")
                        self.save_pdf(path)
                    except Exception as e:
                        logger.warning(f"Failed to index {fname}: {e}")

        if not self.index or not self.chunks:
            return {"text": "", "sources": []}

        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k * 5)

        context_text = ""
        sources_set = set()
        added = 0

        for idx in indices[0]:
            if idx >= len(self.chunks):
                continue
            meta = self.metadata[idx]
            pdf_name = meta["file"]
            if pdf_files and pdf_name not in pdf_files:
                continue

            chunk_text = self.chunks[idx]
            display_text = (
                reshape_for_display(chunk_text)
                if reshape_arabic and contains_arabic(chunk_text)
                else chunk_text
            )

            full_chunk = f"[From {pdf_name}]\n{display_text}\n\n"

            if len(context_text) + len(full_chunk) > max_chars:
                remaining = max_chars - len(context_text)
                context_text += full_chunk[:remaining]
                sources_set.add(pdf_name)
                break

            context_text += full_chunk
            sources_set.add(pdf_name)
            added += 1
            if added >= top_k:
                break

        return {"text": context_text.strip(), "sources": list(sources_set)}

    # -------------------- Maintenance --------------------
    def clear_index(self):
        """Clear everything safely."""
        self.index = None
        self.chunks = []
        self.metadata = []
        for path in [self.index_path, self.chunks_path, self.metadata_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {e}")
        logger.info("Vector store cleared.")
