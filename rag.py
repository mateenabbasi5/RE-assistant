import os
import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

DOC_DIR = os.path.join(os.path.dirname(__file__), "data", "docs")

# keep chunk sizes reasonable
FLAN_MODEL_ID = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_ID)


def _chunk_text(text: str, max_tokens: int = 200) -> List[str]:
    """
    Split a long text into smaller chunks based on sentences,
    keeping each chunk under max_tokens (approximate).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""

    for sent in sentences:
        candidate = (current + " " + sent).strip()
        if not candidate:
            continue

        token_count = len(tokenizer.encode(candidate, add_special_tokens=False))
        if token_count <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    return chunks


def _load_documents() -> List[Tuple[str, str]]:
    """
    Load all .txt files from DOC_DIR and split them into chunks.
    Returns a list of (filename, chunk_text).
    """
    docs: List[Tuple[str, str]] = []

    if not os.path.isdir(DOC_DIR):
        return docs

    for fname in os.listdir(DOC_DIR):
        if not fname.lower().endswith(".txt"):
            continue

        full_path = os.path.join(DOC_DIR, fname)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            continue

        chunks = _chunk_text(text)
        for chunk in chunks:
            docs.append((fname, chunk))

    return docs

# Load and vectorize docs
_document_chunks: List[Tuple[str, str]] = _load_documents()
_corpus_texts: List[str] = [chunk for _, chunk in _document_chunks]

if _corpus_texts:
    _vectorizer = TfidfVectorizer(max_features=5000)
    _doc_vectors = _vectorizer.fit_transform(_corpus_texts)
else:
    _vectorizer = None
    _doc_vectors = None


def retrieve_context(query: str, k: int = 3) -> str:
    """
    Given a user query, return up to k most relevant chunks from the docs
    as a single string, each prefixed with [Source: filename].
    """
    if not _vectorizer or _doc_vectors is None or not _corpus_texts:
        return ""

    query_vec = _vectorizer.transform([query])
    sims = cosine_similarity(query_vec, _doc_vectors)[0]

    # Top-k indices (highest similarity first)
    top_idxs = sims.argsort()[::-1][:k]

    pieces = []
    for idx in top_idxs:
        fname, chunk = _document_chunks[int(idx)]
        pieces.append(f"[Source: {fname}]\n{chunk}")

    return "\n\n".join(pieces)
