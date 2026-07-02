import sys
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import os
import time
from datetime import date

from jd_parser          import parse_jd
from heuristic_filter   import stream_and_filter
from candidate_embedder import embed_jd, embed_jd_intent, embed_candidates, embed_career_only
from ranker             import rank_candidates
from output_writer      import write_submission


_DEFAULT_JD = os.environ.get(
    "JD_PATH",
    r"C:\Users\KIIT0001\Downloads\[PUB] India_runs_data_and_ai_challenge"
    r"\[PUB] India_runs_data_and_ai_challenge"
    r"\India_runs_data_and_ai_challenge\job_description.docx",
)
_DEFAULT_CANDIDATES = os.environ.get(
    "CANDIDATES_PATH",
    r"C:\Users\KIIT0001\Downloads\[PUB] India_runs_data_and_ai_challenge"
    r"\[PUB] India_runs_data_and_ai_challenge"
    r"\India_runs_data_and_ai_challenge\candidates.jsonl",
)
_DEFAULT_OUT = "submission.csv"

def main():
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranking Engine"
    )
    parser.add_argument("--jd",         default=_DEFAULT_JD,         help="Path to JD .docx file")
    parser.add_argument("--candidates", default=_DEFAULT_CANDIDATES,  help="Path to candidates JSONL (or .gz)")
    parser.add_argument("--out",        default=_DEFAULT_OUT,         help="Output CSV filename")
    parser.add_argument("--top-k",      type=int, default=100,        help="Number of candidates to output")
    args = parser.parse_args()

    wall_start = time.time()
    today = date.today()

    
    print("\n" + "="*60)
    print("Stage 2: Parsing Job Description...")
    print("="*60)
    t0 = time.time()
    job = parse_jd(args.jd)
    print(f"  Title           : {job.title}")
    print(f"  Required skills : {job.required_skills[:10]}")
    print(f"  Preferred skills: {job.preferred_skills[:5]}")
    print(f"  YOE range       : {job.min_experience}–{job.max_experience} years")
    dq_preview = [d.encode('ascii', 'replace').decode('ascii')[:60] for d in job.disqualifiers[:3]]
    print(f"  Disqualifiers   : {dq_preview}")
    print(f"  JD parsed in {time.time()-t0:.1f}s")

  
    print("Stage 1: Heuristic Filter (streaming 100K candidates)...")
    print("="*60)
    t0 = time.time()
    survivors = stream_and_filter(
        candidates_path=args.candidates,
        job=job,
        today=today,
        max_candidates=1500,
    )
    print(f"  Stage 1 complete: {len(survivors)} survivors selected in {time.time()-t0:.1f}s")

   
    print("\n" + "="*60)
    print("Stage 3: Semantic NLP Ranker...")
    print("="*60)

    print("  Embedding JD...")
    t0 = time.time()
    jd_vec        = embed_jd(job)
    jd_intent_vec = embed_jd_intent(job)
    print(f"  JD embedded in {time.time()-t0:.2f}s")

    print(f"  Embedding {len(survivors)} candidate profiles...")
    t0 = time.time()
    cand_embeddings   = embed_candidates(survivors, batch_size=128)
    print(f"  Candidate embeddings done in {time.time()-t0:.1f}s")

    print("  Scoring all candidates...")
    t0 = time.time()
    ranked = rank_candidates(
        features=survivors,
        candidate_embeddings=cand_embeddings,
        jd_embedding=jd_vec,
        jd_intent_embedding=jd_intent_vec,
        job=job,
        top_k=args.top_k,
    )
    print(f"  Scoring complete in {time.time()-t0:.2f}s")

    
    print("\n" + "="*60)
    print("Stage 4: Writing submission CSV...")
    print("="*60)
    write_submission(ranked, job, args.out)
    wall_elapsed = time.time() - wall_start
    print(f"\n{'='*60}")
    print(f"✅  Done in {wall_elapsed:.1f}s  ({wall_elapsed/60:.2f} min)")
    print(f"    Output: {args.out}")
    print("="*60)

    if wall_elapsed > 300:
        print("⚠️  WARNING: Exceeded 5-minute budget!")


if __name__ == "__main__":
    main()