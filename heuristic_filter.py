
import gzip
import json
from datetime import date, datetime
from pathlib import Path

from models.candidate import Candidate
from models.features import CandidateFeatures
from models.job_spec import JobSpec


_PROFICIENCY_WEIGHT = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2,
}

_EDUCATION_TIER_RANK = {
    "tier_1": 4,
    "tier_2": 3,
    "tier_3": 2,
    "tier_4": 1,
    "": 0,
}



def stream_and_filter(
    candidates_path: str,
    job: JobSpec,
    today: date | None = None,
    max_candidates: int = 2000,
) -> list[CandidateFeatures]:
    
    if today is None:
        today = date.today()

    path = Path(candidates_path)
    open_fn = gzip.open if path.suffix == ".gz" else open

    scored_candidates = []
    total = 0
    hard_discarded = 0
    soft_discarded = 0

    print(f"[Filter] Streaming and scoring candidates against '{job.title}'...")

    with open_fn(str(path), "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                hard_discarded += 1
                continue

            is_honeypot, _ = _check_honeypots(raw)

            h_score = _compute_heuristic_score(raw, job, today, is_honeypot)

            try:
                features = _extract_features(raw, today)
            except Exception:
                hard_discarded += 1
                continue

            features.is_honeypot = is_honeypot
            scored_candidates.append((h_score, features))

    scored_candidates.sort(key=lambda x: (-x[0], x[1].candidate_id))

    survivors = [feat for _, feat in scored_candidates[:max_candidates]]

    print(
        f"[Filter] Total candidates: {total} | Selected top {len(survivors)} "
        f"for semantic ranking | Hard-discarded: {hard_discarded}"
    )
    return survivors


def _compute_heuristic_score(raw: dict, job: JobSpec, today: date, is_honeypot: bool) -> float:
    if is_honeypot:
        return -999999.0  

    score = 0.0

    profile = raw.get("profile", {})
    signals = raw.get("redrob_signals", {})
    skills = raw.get("skills", [])

    yoe = float(profile.get("years_of_experience", 0))
    if job.min_experience - 1.5 <= yoe <= job.max_experience + 3.0:
        score += 15.0
    else:
        deviation = min(abs(yoe - job.min_experience), abs(yoe - job.max_experience))
        score -= deviation * 3.0

    jd_skills = set(job.required_skills) | set(job.preferred_skills)
    jd_words = set()
    for s in jd_skills:
        for w in s.replace("-", " ").split():
            w = w.strip().lower()
            if len(w) > 2 and w not in ["experience", "systems", "development", "production", "frameworks", "infrastructure", "role", "team", "with", "for", "the"]:
                jd_words.add(w)

    matched_skills = 0
    for s in skills:
        name = s.get("name", "").strip().lower()
        if not name:
            continue
        for w in name.replace("-", " ").split():
            if w in jd_words:
                matched_skills += 1
                break
    score += matched_skills * 2.5

    title_words = set(job.title.lower().replace("-", " ").replace("/", " ").split())
    title_words = {
        w for w in title_words 
        if len(w) > 2 and w not in ["senior", "founding", "lead", "junior", "staff", "head", "manager", "team", "engineer"]
    }
    
    current_title = profile.get("current_title", "").lower()
    title_match = False
    for w in current_title.replace("-", " ").replace("/", " ").split():
        if w in title_words:
            title_match = True
            break
    if title_match:
        score += 12.0

    rrr = float(signals.get("recruiter_response_rate", 0.0))
    score += rrr * 10.0

    open_to_work = bool(signals.get("open_to_work_flag", False))
    if open_to_work:
        score += 5.0

    comp = float(signals.get("profile_completeness_score", 0.0))
    score += (comp / 100.0) * 5.0

    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            days_inactive = (today - date.fromisoformat(last_active_str)).days
            if days_inactive <= 90:
                score += 5.0
            elif days_inactive > 180:
                score -= 5.0
        except ValueError:
            pass

    return score


def _check_honeypots(raw: dict) -> tuple[bool, str]:
    profile = raw.get("profile", {})
    signals = raw.get("redrob_signals", {})
    skills = raw.get("skills", [])
    career = raw.get("career_history", [])

    declared_yoe = float(profile.get("years_of_experience", 0))

    zero_duration_experts = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 1) == 0
    ]
    if len(zero_duration_experts) >= 3:
        return True, f"{len(zero_duration_experts)} expert/advanced skills with 0 months usage"

    suspicious_experts = [
        s for s in skills
        if s.get("proficiency") == "expert"
        and s.get("duration_months", 0) < 3
        and s.get("endorsements", 0) == 0
    ]
    if len(suspicious_experts) >= 5:
        return True, f"{len(suspicious_experts)} expert skills with <3 months and 0 endorsements"

    total_career_months = sum(
        j.get("duration_months", 0) for j in career
        if isinstance(j.get("duration_months"), (int, float))
    )
    if declared_yoe > 2 and total_career_months > 0:
        max_plausible_yoe = total_career_months / 12 * 1.15  
        if declared_yoe > max_plausible_yoe + 3:
            return True, (
                f"Declared YOE={declared_yoe:.1f} but total career "
                f"months={total_career_months} (max plausible={max_plausible_yoe:.1f})"
            )

    signup_str = signals.get("signup_date", "")
    last_active_str = signals.get("last_active_date", "")
    if signup_str and last_active_str:
        try:
            signup = date.fromisoformat(signup_str)
            last_active = date.fromisoformat(last_active_str)
            if signup > last_active:
                return True, f"signup_date ({signup_str}) > last_active_date ({last_active_str})"
        except ValueError:
            pass

    rrr = float(signals.get("recruiter_response_rate", 0))
    icr = float(signals.get("interview_completion_rate", 0))
    oar = float(signals.get("offer_acceptance_rate", 0))
    if rrr == 1.0 and icr == 1.0 and oar == 1.0 and declared_yoe < 2:
        return True, "All behavioral rates = 1.0 with <2 YOE — statistically impossible"

    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count > 12:
        return True, f"Expert in {expert_count} skills — implausibly broad expertise"

    return False, ""


def _check_deadweight(
    raw: dict,
    today: date,
    min_response_rate: float,
    max_inactive_days: int,
    min_completeness: float,
) -> tuple[bool, str]:
    signals = raw.get("redrob_signals", {})
    profile = raw.get("profile", {})

    rrr = float(signals.get("recruiter_response_rate", 0.0))
    completeness = float(signals.get("profile_completeness_score", 0.0))
    open_to_work = bool(signals.get("open_to_work_flag", False))
    last_active_str = signals.get("last_active_date", "")

    days_inactive = 9999
    if last_active_str:
        try:
            last_active = date.fromisoformat(last_active_str)
            days_inactive = (today - last_active).days
        except ValueError:
            pass

    if rrr < min_response_rate and not open_to_work and days_inactive > 90:
        return True, f"RRR={rrr:.2f}, not open to work, inactive {days_inactive}d"

    if completeness < min_completeness:
        return True, f"Profile completeness={completeness:.1f}% below threshold"

    if days_inactive > max_inactive_days and not open_to_work:
        return True, f"Inactive for {days_inactive} days and not open to work"

    return False, ""


def _extract_features(raw: dict, today: date) -> CandidateFeatures:
    profile = raw.get("profile", {})
    signals = raw.get("redrob_signals", {})
    career = raw.get("career_history", [])
    skills_raw = raw.get("skills", [])
    education = raw.get("education", [])

    semantic_text = _build_semantic_text(profile, career, skills_raw)

    skills_set: set[str] = set()
    skill_weights: dict[str, float] = {}
    assessment_scores: dict[str, float] = signals.get("skill_assessment_scores", {})

    career_text_lower = " ".join(
        (j.get("description", "") + " " + j.get("title", "")).lower()
        for j in career
    )

    for s in skills_raw:
        name = s.get("name", "").strip().lower()
        if not name:
            continue
        prof = s.get("proficiency", "beginner").lower()
        dur = min(float(s.get("duration_months", 0)), 36.0)  # cap at 36 months
        prof_w = _PROFICIENCY_WEIGHT.get(prof, 0.2)

        weight = prof_w * (dur / 36.0)

        if name in career_text_lower or name.replace(" ", "") in career_text_lower:
            weight *= 1.5

        skills_set.add(name)
        skill_weights[name] = min(weight, 1.5)  # cap

    edu_tier = ""
    if education:
        edu_tier = education[0].get("tier", "")

    last_active_str = signals.get("last_active_date", "")
    days_ago = 999
    if last_active_str:
        try:
            days_ago = (today - date.fromisoformat(last_active_str)).days
        except ValueError:
            pass

    return CandidateFeatures(
        candidate_id=raw.get("candidate_id", ""),
        semantic_text=semantic_text,
        years_experience=float(profile.get("years_of_experience", 0)),
        current_title=profile.get("current_title", ""),
        current_company=profile.get("current_company", ""),
        company_size=profile.get("current_company_size", ""),
        industry=profile.get("current_industry", ""),
        education_tier=edu_tier,
        location=profile.get("location", ""),
        skills=skills_set,
        skill_weights=skill_weights,
        skill_assessment_scores={k.lower(): v for k, v in assessment_scores.items()},
        github_score=float(signals.get("github_activity_score", 0)),
        recruiter_response_rate=float(signals.get("recruiter_response_rate", 0)),
        avg_response_time_hours=float(signals.get("avg_response_time_hours", 999)),
        interview_completion_rate=float(signals.get("interview_completion_rate", 0)),
        offer_acceptance_rate=float(signals.get("offer_acceptance_rate", 0)),
        profile_completeness=float(signals.get("profile_completeness_score", 0)),
        notice_period_days=int(signals.get("notice_period_days", 90)),
        open_to_work=bool(signals.get("open_to_work_flag", False)),
        willing_to_relocate=bool(signals.get("willing_to_relocate", False)),
        last_active_days_ago=days_ago,
        profile_views_30d=int(signals.get("profile_views_received_30d", 0)),
        saved_by_recruiters_30d=int(signals.get("saved_by_recruiters_30d", 0)),
        connection_count=int(signals.get("connection_count", 0)),
        is_honeypot=False,
    )


def _build_semantic_text(profile: dict, career: list, skills_raw: list) -> str:
    parts: list[str] = []
    if profile.get("current_title"):
        parts.append(f"[TITLE] {profile['current_title']}")
    if profile.get("headline"):
        parts.append(f"[HEADLINE] {profile['headline']}")
    if profile.get("summary"):
        parts.append(f"[SUMMARY] {profile['summary']}")
    career_parts: list[str] = []
    for job in career:
        role_text = f"{job.get('title', '')} at {job.get('company', '')}. {job.get('description', '')}"
        career_parts.append(role_text.strip())

    career_blob = "\n".join(career_parts)
    if career_blob:
        parts.append(f"[CAREER] {career_blob}")

    proficiency_order = {"expert": 0, "advanced": 1, "intermediate": 2, "beginner": 3}
    sorted_skills = sorted(
        skills_raw,
        key=lambda s: proficiency_order.get(s.get("proficiency", "beginner").lower(), 3),
    )
    skill_names = [s.get("name", "") for s in sorted_skills if s.get("name")]
    if skill_names:
        parts.append(f"[SKILLS] {' '.join(skill_names)}")

    return "\n".join(parts)
