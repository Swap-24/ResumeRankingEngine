"""
candidate_embedder.py — Batch-embed candidate semantic texts using BGE-small.

Designed for CPU-only inference with sentence-transformers.
Uses batched encoding for throughput; ~1-2 min for 5K candidates on modern CPU.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from models.features import CandidateFeatures
from models.job_spec import JobSpec

# ---------------------------------------------------------------------------
# Model — loaded once at module import, shared across all calls
# ---------------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

print(f"[Embedder] Loading model: {MODEL_NAME}")
_model = SentenceTransformer(MODEL_NAME)
print("[Embedder] Model ready.")


# ---------------------------------------------------------------------------
# JD embedding
# ---------------------------------------------------------------------------

def embed_jd(job: JobSpec) -> np.ndarray:
    """
    Produce a single JD query vector.
    Blend of:
      - 70%: full JD text (captures semantic intent broadly)
      - 30%: skills-only text (focuses on keyword alignment)

    The BGE model is trained with a query prefix for retrieval tasks.
    """
    # Full JD text
    full_vec = _encode_query(job.full_text[:2048])  # truncate to avoid token overflow

    # Skills-focused text
    skills_text = " ".join(job.required_skills + job.preferred_skills)
    if skills_text.strip():
        skills_vec = _encode_query(skills_text)
        blended = 0.7 * full_vec + 0.3 * skills_vec
    else:
        blended = full_vec

    return _normalize(blended)


def embed_jd_intent(job: JobSpec) -> np.ndarray:
    """
    Embed just the role intent text (title + opening sentences).
    Used for the career-alignment anti-keyword-stuffer check.
    """
    return _normalize(_encode_query(job.intent_text[:512]))


# ---------------------------------------------------------------------------
# Candidate embedding
# ---------------------------------------------------------------------------

def embed_candidates(
    features: list[CandidateFeatures],
    batch_size: int = 128,
) -> np.ndarray:
    """
    Batch-embed all candidate semantic texts.

    Returns
    -------
    np.ndarray of shape (N, D) — normalized L2 embeddings, float32.
    """
    texts = [f.semantic_text[:3000] for f in features]  # hard cap per candidate

    print(f"[Embedder] Encoding {len(texts)} candidates (batch_size={batch_size})...")

    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return embeddings.astype(np.float32)


def embed_career_only(
    features: list[CandidateFeatures],
    batch_size: int = 128,
) -> np.ndarray:
    """
    Embed only the career history portion of each candidate's text.
    Used for the career-alignment check (anti-keyword-stuffer).
    """
    texts = []
    for f in features:
        # Extract only the [CAREER] sections from semantic_text
        career_lines = [
            line for line in f.semantic_text.split("\n")
            if line.startswith("[CAREER]") or line.startswith("[TITLE]")
        ]
        texts.append(" ".join(career_lines)[:2000] if career_lines else f.current_title)

    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _encode_query(text: str) -> np.ndarray:
    """
    Encode a single query string with BGE's recommended instruction prefix.
    BGE-small uses 'Represent this sentence for searching relevant passages:'
    for asymmetric retrieval.
    """
    instruction = "Represent this sentence for searching relevant passages: "
    return _model.encode(
        instruction + text,
        normalize_embeddings=False,
        convert_to_numpy=True,
    )


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return vec / norm
