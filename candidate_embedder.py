
import numpy as np
from sentence_transformers import SentenceTransformer

from models.features import CandidateFeatures
from models.job_spec import JobSpec



MODEL_NAME = "BAAI/bge-small-en-v1.5"

print(f"[Embedder] Loading model: {MODEL_NAME}")
_model = SentenceTransformer(MODEL_NAME)
print("[Embedder] Model ready.")



def embed_jd(job: JobSpec) -> np.ndarray:
    full_vec = _encode_query(job.full_text[:2048])  # truncate to avoid token overflow

    skills_text = " ".join(job.required_skills + job.preferred_skills)
    if skills_text.strip():
        skills_vec = _encode_query(skills_text)
        blended = 0.7 * full_vec + 0.3 * skills_vec
    else:
        blended = full_vec

    return _normalize(blended)


def embed_jd_intent(job: JobSpec) -> np.ndarray:
    return _normalize(_encode_query(job.intent_text[:512]))



def embed_candidates(
    features: list[CandidateFeatures],
    batch_size: int = 128,
) -> np.ndarray:
    texts = [f.semantic_text[:3000] for f in features]  

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




def _encode_query(text: str) -> np.ndarray:
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
