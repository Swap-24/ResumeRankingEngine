from dataclasses import dataclass, field

@dataclass(slots=True)
class Job:

    title: str
    company: str = ""
    location: str = ""
    summary: str = ""
    min_experience: float = 0
    max_experience: float = 100
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    culture: list[str] = field(default_factory=list)
    semantic_requirements: str = ""
    semantic_responsibilities: str = ""
    semantic_company: str = ""