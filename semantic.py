from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

def embed(text: str):

    if not text.strip():
        return None

    return model.encode(

        text,

        normalize_embeddings=True,

        convert_to_numpy=True
    )

def build_resume_text(candidate):

    resume = candidate.resume

    pieces = []

    if resume.summary:
        pieces.append(resume.summary)

    if resume.skills:
        pieces.append(" ".join(resume.skills))

    for exp in resume.career_history:

        pieces.append(exp.get("title", ""))

        pieces.append(exp.get("description", ""))

    for project in resume.projects:

        pieces.append(project.get("title", ""))

        pieces.append(project.get("description", ""))

    return "\n".join(pieces)

def build_job_text(job):

    pieces = [

        job.title,

        job.summary,

        " ".join(job.required_skills),

        " ".join(job.preferred_skills)
    ]

    return "\n".join(pieces)

def similarity(vec1, vec2):

    if vec1 is None or vec2 is None:
        return 0.0

    return float(np.dot(vec1, vec2))