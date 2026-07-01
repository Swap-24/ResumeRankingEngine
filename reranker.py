from heapq import nlargest

from models.candidate import Candidate
from models.job import Job
from scorer import overall_score

def rank_candidates(
    candidates: list[Candidate],
    job: Job,
    top_k: int = 100
):

    scored = []

    for candidate in candidates:

        score = overall_score(candidate, job)

        scored.append((score, candidate))

    return nlargest(
        top_k,
        scored,
        key=lambda x: x[0]
    )