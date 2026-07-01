"""
main.py — Redrob Intelligent Candidate Ranking Engine

Entry point for the two-stage JD-agnostic pipeline.

NOTE: UTF-8 stdout is forced early to handle Unicode chars (arrows, em-dashes
etc.) that appear in the JD text without crashing on Windows CP1252 consoles.

Usage:
    python main.py \\
        --jd         <path/to/job_description.docx> \\
        --candidates <path/to/candidates.jsonl[.gz]> \\
        --out        <output_filename.csv>

Defaults (for local dev):
    --jd        uses JD_PATH env variable or the hardcoded dev path below
    --candidates uses CANDIDATES_PATH env variable or the hardcoded dev path
    --out        submission.csv
"""

import sys
import io
# Force UTF-8 stdout/stderr on Windows to handle Unicode chars (→ etc.) in JD text
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import argparse
import os
import time
from datetime import date

# ---- Stage modules --------------------------------------------------------
from jd_parser          import parse_jd
from heuristic_filter   import stream_and_filter
from candidate_embedder import embed_jd, embed_jd_intent, embed_candidates, embed_career_only
from ranker             import rank_candidates
from output_writer      import write_submission


# ---- Default dev paths (override via CLI args or env vars) ----------------
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


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Stage 2: Parse the JD (fast — before streaming, so we can use
    # JD role type for potential Stage 1 pruning in future)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("Stage 2: Parsing Job Description...")
    print("="*60)
    t0 = time.time()
    job = parse_jd(args.jd)
    print(f"  Title           : {job.title}")
    print(f"  Required skills : {job.required_skills[:10]}")
    print(f"  Preferred skills: {job.preferred_skills[:5]}")
    print(f"  YOE range       : {job.min_experience}–{job.max_experience} years")
    # Strip non-ASCII for safe console printing on Windows
    dq_preview = [d.encode('ascii', 'replace').decode('ascii')[:60] for d in job.disqualifiers[:3]]
    print(f"  Disqualifiers   : {dq_preview}")
    print(f"  JD parsed in {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # Stage 1: Heuristic filter (O(N) streaming)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
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

    # -----------------------------------------------------------------------
    # Stage 3: Semantic NLP Ranker
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("Stage 3: Semantic NLP Ranker...")
    print("="*60)

    # 3a. Embed JD
    print("  Embedding JD...")
    t0 = time.time()
    jd_vec        = embed_jd(job)
    jd_intent_vec = embed_jd_intent(job)
    print(f"  JD embedded in {time.time()-t0:.2f}s")

    # 3b. Embed all survivors (full semantic text)
    print(f"  Embedding {len(survivors)} candidate profiles...")
    t0 = time.time()
    cand_embeddings   = embed_candidates(survivors, batch_size=128)
    print(f"  Candidate embeddings done in {time.time()-t0:.1f}s")

    # 3c. Score and rank
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

    # -----------------------------------------------------------------------
    # Stage 4: Write submission
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("Stage 4: Writing submission CSV...")
    print("="*60)
    write_submission(ranked, job, args.out)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    wall_elapsed = time.time() - wall_start
    print(f"\n{'='*60}")
    print(f"✅  Done in {wall_elapsed:.1f}s  ({wall_elapsed/60:.2f} min)")
    print(f"    Output: {args.out}")
    print("="*60)

    if wall_elapsed > 300:
        print("⚠️  WARNING: Exceeded 5-minute budget!")


if __name__ == "__main__":
    main()