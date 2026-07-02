import numpy as np
from models.features import CandidateFeatures
from models.job_spec import JobSpec



W_SEMANTIC      = 0.45
W_CAREER_ALIGN  = 0.15
W_SKILL_OVERLAP = 0.15
W_BEHAVIORAL    = 0.20
W_YOE           = 0.05

CAREER_ALIGN_THRESHOLD = 0.22
CAREER_ALIGN_PENALTY   = 0.40

HONEYPOT_PENALTY = 0.05

DISQUALIFIER_PENALTY = 0.30




def rank_candidates(
    features: list[CandidateFeatures],
    candidate_embeddings: np.ndarray,   
    jd_embedding: np.ndarray,            
    jd_intent_embedding: np.ndarray,     
    job: JobSpec,
    top_k: int = 100,
) -> list[tuple[float, CandidateFeatures]]:
    N = len(features)
    assert candidate_embeddings.shape[0] == N

    semantic_scores = candidate_embeddings @ jd_embedding           
    career_scores   = candidate_embeddings @ jd_intent_embedding     

    semantic_scores = np.clip(semantic_scores, 0.0, 1.0)
    career_scores   = np.clip(career_scores,   0.0, 1.0)

    required_set = set(job.required_skills)
    preferred_set = set(job.preferred_skills)
    disqualifier_phrases = [d.lower() for d in job.disqualifiers]

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

        penalty = 1.0

        if feat.is_honeypot:
            penalty *= HONEYPOT_PENALTY

        if c_aln < CAREER_ALIGN_THRESHOLD:
            penalty *= CAREER_ALIGN_PENALTY

        career_text_lower = feat.semantic_text.lower()
        for phrase in disqualifier_phrases:
            if phrase and phrase in career_text_lower:
                penalty *= DISQUALIFIER_PENALTY
                break  

        final = raw_score * penalty
        scored.append((final, feat))

    scored.sort(key=lambda x: (-x[0], x[1].candidate_id))

    return scored[:top_k]




def _skill_overlap_score(feat: CandidateFeatures, required: set[str], preferred: set[str],) -> float:
    if not required and not preferred:
        return 0.5  

    total_possible = len(required) + 0.5 * len(preferred)
    if total_possible == 0:
        return 0.5

    earned = 0.0
    candidate_tokens = {
        skill: set(skill.split())
        for skill in feat.skills
    }

    for skill_name in required:
        matched_weight = _fuzzy_skill_match(skill_name, feat, candidate_tokens)
        earned += matched_weight

    for skill_name in preferred:
        matched_weight = _fuzzy_skill_match(skill_name, feat, candidate_tokens)
        earned += 0.5 * matched_weight

    return min(earned / total_possible, 1.0)


def _fuzzy_skill_match(
    required_skill: str,
    feat: CandidateFeatures,
    candidate_tokens: dict[str, set[str]] | None = None,
) -> float:
    req = required_skill.lower()
    req_tokens = set(req.split())
    if candidate_tokens is None:
        candidate_tokens = {
            skill: set(skill.split())
            for skill in feat.skills
        }

    best_weight = 0.0

    for cand_skill in feat.skills:
        weight = feat.skill_weights.get(cand_skill, 0.0)
        cand_tokens = candidate_tokens[cand_skill]

        if req == cand_skill:
            return weight

        if req in cand_skill or cand_skill in req:
            best_weight = max(best_weight, weight * 0.9)
            continue

        union = req_tokens | cand_tokens
        inter = req_tokens & cand_tokens
        if union and len(inter) / len(union) >= 0.5:
            best_weight = max(best_weight, weight * 0.75)

    return best_weight


def _behavioral_score(feat: CandidateFeatures) -> float:
    activity = max(0.0, 1.0 - feat.last_active_days_ago / 365.0)

    notice = max(0.0, 1.0 - feat.notice_period_days / 180.0)

    completeness = min(feat.profile_completeness / 100.0, 1.0)

    score = (
        0.35 * feat.recruiter_response_rate
      + 0.20 * activity
      + 0.15 * completeness
      + 0.15 * notice
      + 0.10 * feat.interview_completion_rate
      + 0.05 * feat.offer_acceptance_rate
    )

    if feat.open_to_work:
        score = min(score + 0.05, 1.0)

    return score


def _yoe_score(yoe: float, min_exp: float, max_exp: float) -> float:
    if min_exp <= yoe <= max_exp:
        return 1.0
    if yoe < min_exp:
        if min_exp == 0:
            return 1.0
        return max(0.0, yoe / min_exp)
    overshoot = yoe - max_exp
    return max(0.5, 1.0 - overshoot * 0.05)  
