from dataclasses import dataclass, field

@dataclass
class Job:
    title: str = ""
    summary: str = ""

    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)

    min_experience: float = 0
    max_experience: float = 100

    location: str = ""