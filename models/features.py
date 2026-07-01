from dataclasses import dataclass, field


@dataclass(slots=True)
class CandidateFeatures:
    """
    Extracted + pre-processed features for a single candidate.
    Built once during Stage 1 / Stage 3 and reused for scoring.
    """

    # Identity
    candidate_id: str

    # Semantic text blob used for embedding (career-weighted)
    semantic_text: str

    # Profile signals
    years_experience: float
    current_title: str
    current_company: str
    company_size: str
    industry: str
    education_tier: str                          # e.g. "tier_1", "tier_3"
    location: str

    # Skills — name → (proficiency_weight, duration_months)
    # proficiency_weight: expert=1.0, advanced=0.8, intermediate=0.5, beginner=0.2
    skills: set[str]                              # lowercased skill names
    skill_weights: dict[str, float] = field(default_factory=dict)  # name → weight × capped_months
    skill_assessment_scores: dict[str, float] = field(default_factory=dict)

    # Behavioral / Redrob signals
    github_score: float = 0.0                    # 0–100
    recruiter_response_rate: float = 0.0         # 0.0–1.0
    avg_response_time_hours: float = 999.0
    interview_completion_rate: float = 0.0       # 0.0–1.0
    offer_acceptance_rate: float = 0.0           # 0.0–1.0
    profile_completeness: float = 0.0            # 0–100
    notice_period_days: int = 90
    open_to_work: bool = False
    willing_to_relocate: bool = False
    last_active_days_ago: int = 999              # computed from last_active_date
    profile_views_30d: int = 0
    saved_by_recruiters_30d: int = 0
    connection_count: int = 0

    # Honeypot flag — set by Stage 1 heuristic filter
    is_honeypot: bool = False