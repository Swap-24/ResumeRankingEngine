"""
ranker.py — Stage 3: Multi-signal scoring and final top-100 selection.

Scoring formula (all components normalized 0.0–1.0):
  final_score = (
      semantic_score        * 0.45   # cosine sim (JD embedding vs candidate)
    + career_align_score    * 0.15   # career/title vs JD intent (anti-keyword-stuffer)
    + skill_overlap_score   * 0.15   # required skill coverage (weighted by duration)
    + behavioral_score      * 0.20   # redrob_signals multiplier
    + yoe_score             * 0.05   # YOE within JD bounds
  )

Penalties applied multiplicatively:
  - Honeypot: × 0.05  (near-disqualification)
  - Career misalignment (title/history irrelevant to JD): × 0.40
  - Disqualifier phrase matched in career text: × 0.30 per phrase
"""

import numpy as np
from models.features import CandidateFeatures
from models.job_spec import JobSpec


# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------

W_SEMANTIC      = 0.45
W_CAREER_ALIGN  = 0.15
W_SKILL_OVERLAP = 0.15
W_BEHAVIORAL    = 0.20
W_YOE           = 0.05

# Career alignment penalty threshold — below this cosine sim → apply penalty
CAREER_ALIGN_THRESHOLD = 0.22
CAREER_ALIGN_PENALTY   = 0.40

# Honeypot penalty multiplier
HONEYPOT_PENALTY = 0.05

# Disqualifier penalty per matched phrase
DISQUALIFIER_PENALTY = 0.30


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_candidates(
    features: list[CandidateFeatures],
    candidate_embeddings: np.ndarray,    # shape (N, D)
    jd_embedding: np.ndarray,            # shape (D,)
    jd_intent_embedding: np.ndarray,     # shape (D,)
    job: JobSpec,
    top_k: int = 100,
) -> list[tuple[float, CandidateFeatures]]:
    """
    Score all candidates and return the top_k sorted descending by score.
    Tie-break: equal scores → sort by candidate_id ascending.
    """
    N = len(features)
    assert candidate_embeddings.shape[0] == N

    # ---- 1. Semantic scores (batch dot product) --------------------------
    # Embeddings are L2-normalized → dot product == cosine similarity
    semantic_scores = candidate_embeddings @ jd_embedding           # (N,)
    career_scores   = candidate_embeddings @ jd_intent_embedding     # (N,)

    # Clip to [0, 1] — cosine similarity should already be positive for
    # reasonable candidates, but BGE can produce small negatives
    semantic_scores = np.clip(semantic_scores, 0.0, 1.0)
    career_scores   = np.clip(career_scores,   0.0, 1.0)

    # ---- 2. Precompute JD required skill set -----------------------------
    required_set = set(job.required_skills)
    preferred_set = set(job.preferred_skills)
    disqualifier_phrases = [d.lower() for d in job.disqualifiers]

    # ---- 3. Score each candidate -----------------------------------------
    scored: list[tuple[float, CandidateFeatures]] = []

    for i, feat in enumerate(features):
        sem   = float(semantic_scores[i])
        c_aln = float(career_scores[i])

        skill  = _skill_overlap_score(feat, required_set, preferred_set)
        behav  = _behavioral_score(feat)
        yoe    = _yoe_score(feat.years_experience, job.min_experience, job.max_experience)

        raw_score = (
            W_SEMANTIC      * sem
          + W_CAREER_ALIGN  * c_aln
          + W_SKILL_OVERLAP * skill
          + W_BEHAVIORAL    * behav
          + W_YOE           * yoe
        )

        # ---- Apply penalties ----
        penalty = 1.0

        # Honeypot penalty
        if feat.is_honeypot:
            penalty *= HONEYPOT_PENALTY

        # Career misalignment penalty (anti-keyword-stuffer)
        if c_aln < CAREER_ALIGN_THRESHOLD:
            penalty *= CAREER_ALIGN_PENALTY

        # Disqualifier phrase penalty
        career_text_lower = feat.semantic_text.lower()
        for phrase in disqualifier_phrases:
            if phrase and phrase in career_text_lower:
                penalty *= DISQUALIFIER_PENALTY
                break  # one match is enough

        final = raw_score * penalty
        scored.append((final, feat))

    # ---- 4. Sort: descending score, ascending candidate_id on ties -------
    scored.sort(key=lambda x: (-x[0], x[1].candidate_id))

    return scored[:top_k]


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def _skill_overlap_score(
    feat: CandidateFeatures,
    required: set[str],
    preferred: set[str],
) -> float:
    """
    Skill overlap score with duration-weighted matching.

    Each required skill is worth 1.0 if matched, scaled by the candidate's
    skill_weight (proficiency × capped_duration / max_duration).
    Preferred skills contribute at half weight.
    """
    if not required and not preferred:
        return 0.5  # neutral if JD has no skill requirements

    total_possible = len(required) + 0.5 * len(preferred)
    if total_possible == 0:
        return 0.5

    earned = 0.0

    for skill_name in required:
        # Fuzzy match: check if required skill is a substring of any candidate skill
        # or vice versa (handles "Python" matching "Python 3", "postgres" → "postgresql")
        matched_weight = _fuzzy_skill_match(skill_name, feat)
        earned += matched_weight

    for skill_name in preferred:
        matched_weight = _fuzzy_skill_match(skill_name, feat)
        earned += 0.5 * matched_weight

    return min(earned / total_possible, 1.0)


def _fuzzy_skill_match(required_skill: str, feat: CandidateFeatures) -> float:
    """
    Check if a required skill matches any of the candidate's skills.
    Returns the candidate's skill_weight for the best match (0.0 if no match).

    Matching strategy (no external libraries needed):
    1. Exact match
    2. Substring containment (both directions)
    3. First-token match (e.g. "python" matches "python 3.10")
    """
    req = required_skill.lower()
    req_tokens = set(req.split())

    best_weight = 0.0

    for cand_skill in feat.skills:
        weight = feat.skill_weights.get(cand_skill, 0.0)
        cand_tokens = set(cand_skill.split())

        # Exact match
        if req == cand_skill:
            return weight

        # Substring match (either direction)
        if req in cand_skill or cand_skill in req:
            best_weight = max(best_weight, weight * 0.9)
            continue

        # Token overlap (Jaccard-like: shared tokens / union tokens)
        union = req_tokens | cand_tokens
        inter = req_tokens & cand_tokens
        if union and len(inter) / len(union) >= 0.5:
            best_weight = max(best_weight, weight * 0.75)

    return best_weight


def _behavioral_score(feat: CandidateFeatures) -> float:
    """
    Composite behavioral score from redrob_signals (0.0–1.0).

    Weights:
      recruiter_response_rate   35%
      activity_score            20%  (inverse of days_inactive)
      profile_completeness      15%
      notice_period_score       15%  (shorter = better)
      interview_completion_rate 10%
      offer_acceptance_rate      5%
    """
    # Activity score: linear decay from 0 → 365 days inactive
    activity = max(0.0, 1.0 - feat.last_active_days_ago / 365.0)

    # Notice period score: 0 days = 1.0, 180 days = 0.0
    notice = max(0.0, 1.0 - feat.notice_period_days / 180.0)

    # Profile completeness: /100 → 0.0–1.0
    completeness = min(feat.profile_completeness / 100.0, 1.0)

    score = (
        0.35 * feat.recruiter_response_rate
      + 0.20 * activity
      + 0.15 * completeness
      + 0.15 * notice
      + 0.10 * feat.interview_completion_rate
      + 0.05 * feat.offer_acceptance_rate
    )

    # Small bonus for open_to_work flag
    if feat.open_to_work:
        score = min(score + 0.05, 1.0)

    return score


def _yoe_score(yoe: float, min_exp: float, max_exp: float) -> float:
    """
    YOE alignment score (0.0–1.0).
    Full credit if within the JD's target range.
    Partial credit if below (under-experienced).
    Slight penalty if above (over-qualified).
    """
    if min_exp <= yoe <= max_exp:
        return 1.0
    if yoe < min_exp:
        if min_exp == 0:
            return 1.0
        return max(0.0, yoe / min_exp)
    # yoe > max_exp
    overshoot = yoe - max_exp
    return max(0.5, 1.0 - overshoot * 0.05)  # gentle penalty, floor at 0.5
