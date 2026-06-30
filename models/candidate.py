from dataclasses import dataclass
from .resume import Resume

@dataclass
class Candidate:
    candidate_id: str
    resume: Resume

    location: str = ""
    github_activity: float = 0.0
    recruiter_response_rate: float = 0.0
    open_to_work: bool = False