from loader import load_candidates
from models.job import Job
from reranker import rank_candidates


def main():

    job = Job(
        title="AI Engineer",
        required_skills=[
            "Python",
            "Machine Learning",
            "TensorFlow"
        ],
        min_experience=5,
        max_experience=9,
        location="Pune"
    )

    candidates = load_candidates(
        "candidates.jsonl.gz"
    )

    top = rank_candidates(
        candidates,
        job,
        top_k=100
    )

    for rank, (score, candidate) in enumerate(top, start=1):

        print(
            rank,
            candidate.candidate_id,
            score
        )


if __name__ == "__main__":
    main()