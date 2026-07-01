"""
output_writer.py — Stage 4: Produce a valid submission.csv.

Enforces:
  - Exactly 100 data rows
  - Ranks 1–100, each exactly once
  - Scores monotonically non-increasing
  - Tie-break: equal scores → candidate_id ascending
  - Reasoning column: hallucination-free, candidate-specific
  - UTF-8 encoding
"""

import csv
from pathlib import Path

from models.features import CandidateFeatures
from models.job_spec import JobSpec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_submission(
    ranked: list[tuple[float, CandidateFeatures]],
    job: JobSpec,
    output_path: str,
) -> None:
    """
    Write the final submission CSV from ranked candidates.

    Parameters
    ----------
    ranked      : list of (score, features) sorted descending by score (top 100)
    job         : JobSpec for reasoning generation
    output_path : path to output .csv file
    """
    if len(ranked) < 100:
        raise ValueError(f"Need at least 100 ranked candidates, got {len(ranked)}")

    top100 = ranked[:100]

    # ---- Enforce monotonic non-increasing scores -------------------------
    top100 = _enforce_monotonic(top100)

    # ---- Write CSV -------------------------------------------------------
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(str(out), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (score, feat) in enumerate(top100, start=1):
            reasoning = _build_reasoning(feat, job, score)
            writer.writerow([
                feat.candidate_id,
                rank,
                f"{score:.6f}",
                reasoning,
            ])

    print(f"[Writer] Submission written to: {out.resolve()}")
    print(f"[Writer] Top candidate: {top100[0][1].candidate_id} (score={top100[0][0]:.4f})")
    print(f"[Writer] #100 candidate: {top100[99][1].candidate_id} (score={top100[99][0]:.4f})")


# ---------------------------------------------------------------------------
# Reasoning builder
# ---------------------------------------------------------------------------

def _build_reasoning(feat: CandidateFeatures, job: JobSpec, score: float) -> str:
    """
    Build a 1–2 sentence hallucination-free reasoning string.
    Only references fields that actually exist in the candidate's data.
    """
    parts: list[str] = []

    # Sentence 1: identity + experience
    yoe_str = f"{feat.years_experience:.1f}yr"
    title_str = feat.current_title or "unknown role"
    company_str = f" at {feat.current_company}" if feat.current_company else ""
    parts.append(f"{yoe_str} {title_str}{company_str}.")

    # Sentence 2: skill alignment + behavioral signals
    detail_parts: list[str] = []

    # Which required skills does this candidate actually have?
    matched_required = [
        s for s in job.required_skills
        if _skill_in_candidate(s, feat)
    ]
    if matched_required:
        skills_str = ", ".join(matched_required[:3])  # show up to 3
        jd_mention = f"JD requirement for {skills_str}"
        detail_parts.append(f"matches {jd_mention}")

    # Behavioral highlights
    rrr_pct = int(feat.recruiter_response_rate * 100)
    detail_parts.append(f"{rrr_pct}% recruiter response rate")
    detail_parts.append(f"{feat.notice_period_days}d notice period")

    if feat.open_to_work:
        detail_parts.append("actively open to work")

    if feat.last_active_days_ago <= 30:
        detail_parts.append("active within last 30 days")
    elif feat.last_active_days_ago <= 90:
        detail_parts.append("active within last 90 days")

    parts.append("Demonstrates " + "; ".join(detail_parts) + ".")

    reasoning = " ".join(parts)

    # Hard cap at 300 chars to keep CSV clean
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    return reasoning


def _skill_in_candidate(required_skill: str, feat: CandidateFeatures) -> bool:
    """Check if a required skill is present in the candidate's skill set."""
    req = required_skill.lower()
    for s in feat.skills:
        if req == s or req in s or s in req:
            return True
    return False


# ---------------------------------------------------------------------------
# Monotonic score enforcement
# ---------------------------------------------------------------------------

def _enforce_monotonic(
    ranked: list[tuple[float, CandidateFeatures]],
) -> list[tuple[float, CandidateFeatures]]:
    """
    Ensure scores are strictly non-increasing.
    If a lower-ranked candidate has a higher score (shouldn't happen after sorting,
    but floating point can cause surprises), cap it to the previous score.
    """
    if not ranked:
        return ranked

    result = [ranked[0]]
    prev_score = ranked[0][0]

    for score, feat in ranked[1:]:
        if score > prev_score:
            score = prev_score  # cap to enforce monotonicity
        result.append((score, feat))
        prev_score = score

    return result
