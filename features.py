from dataclasses import dataclass
from models.candidate import Candidate


@dataclass(slots=True)
class CandidateFeatures:
    candidate_id: str
    semantic_text: str
    years_experience: float
    skills: set[str]
    skill_assessment_scores: dict[str, float]
    current_title: str
    current_company: str
    company_size: str
    industry: str
    education_tier: str
    github_score: float
    recruiter_response_rate: float
    interview_completion_rate: float
    offer_acceptance_rate: float
    profile_views: int
    saved_by_recruiters: int
    profile_completeness: float
    open_to_work: bool
    willing_to_relocate: bool

def extract_features(candidate: Candidate) -> CandidateFeatures:

    profile = candidate.raw["profile"]
    signals = candidate.raw["redrob_signals"]

    semantic_parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]

    # Career History
    for job in candidate.raw.get("career_history", []):

        semantic_parts.append(job.get("title", ""))
        semantic_parts.append(job.get("description", ""))

    # Skills
    skills = set()

    assessment_scores = {}

    for skill in candidate.raw.get("skills", []):

        name = skill["name"].lower()

        skills.add(name)

        if name in signals.get("skill_assessment_scores", {}):

            assessment_scores[name] = \
                signals["skill_assessment_scores"][name]

        semantic_parts.append(name)

    education = candidate.raw.get("education", [])

    tier = ""

    if education:
        tier = education[0].get("tier", "")

    semantic_text = "\n".join(semantic_parts)

    return CandidateFeatures(

        candidate_id=candidate.candidate_id,

        semantic_text=semantic_text,

        years_experience=profile.get(
            "years_of_experience",
            0
        ),

        skills=skills,

        skill_assessment_scores=assessment_scores,

        current_title=profile.get(
            "current_title",
            ""
        ),

        current_company=profile.get(
            "current_company",
            ""
        ),

        company_size=profile.get(
            "current_company_size",
            ""
        ),

        industry=profile.get(
            "current_industry",
            ""
        ),

        education_tier=tier,

        github_score=signals.get(
            "github_activity_score",
            0
        ),

        recruiter_response_rate=signals.get(
            "recruiter_response_rate",
            0
        ),

        interview_completion_rate=signals.get(
            "interview_completion_rate",
            0
        ),

        offer_acceptance_rate=signals.get(
            "offer_acceptance_rate",
            0
        ),

        profile_views=signals.get(
            "profile_views_received_30d",
            0
        ),

        saved_by_recruiters=signals.get(
            "saved_by_recruiters_30d",
            0
        ),

        profile_completeness=signals.get(
            "profile_completeness_score",
            0
        ),

        open_to_work=signals.get(
            "open_to_work_flag",
            False
        ),

        willing_to_relocate=signals.get(
            "willing_to_relocate",
            False
        ),
    )