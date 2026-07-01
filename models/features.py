from dataclasses import dataclass, field


@dataclass(slots=True)
class CandidateFeatures:

    
    candidate_id: str

    semantic_text: str

    years_experience: float
    current_title: str
    current_company: str
    company_size: str
    industry: str
    education_tier: str                          
    location: str

    skills: set[str]                             
    skill_weights: dict[str, float] = field(default_factory=dict)  
    skill_assessment_scores: dict[str, float] = field(default_factory=dict)

    github_score: float = 0.0                    
    recruiter_response_rate: float = 0.0         
    avg_response_time_hours: float = 999.0
    interview_completion_rate: float = 0.0       
    offer_acceptance_rate: float = 0.0           
    profile_completeness: float = 0.0           
    notice_period_days: int = 90
    open_to_work: bool = False
    willing_to_relocate: bool = False
    last_active_days_ago: int = 999              
    profile_views_30d: int = 0
    saved_by_recruiters_30d: int = 0
    connection_count: int = 0

    is_honeypot: bool = False