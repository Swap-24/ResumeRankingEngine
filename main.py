from loader import load_job
from semantic import embed, build_job_text
from retriever import retrieve
from reranker import rerank
from writer import write_submission


def main():

    print("Loading job...")
    job = load_job(r"C:\Users\KIIT0001\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\job_description.docx")

    print("Building job text...")
    job_text = build_job_text(job)

    print("Embedding job...")
    job_embedding = embed(job_text)

    print("Retrieving candidates...")
    retrieved = retrieve(job_embedding, top_k=3000)

    print(f"Retrieved {len(retrieved)} candidates")

    print("Reranking...")
    final = rerank(retrieved, job)

    print(f"Final candidates: {len(final)}")

    print("Writing submission...")
    write_submission(final, "submission.csv")

    print("Done!")


if __name__ == "__main__":
    main()