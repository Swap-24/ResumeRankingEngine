from dataclasses import dataclass, field


@dataclass(slots=True)
class JobSpec:
    """Structured representation of a parsed Job Description."""

    title: str
    full_text: str                          # Raw JD text — used for embedding
    intent_text: str                        # Title + first sentences — role intent embedding
    required_skills: list[str] = field(default_factory=list)   # lowercased
    preferred_skills: list[str] = field(default_factory=list)  # lowercased
    min_experience: float = 0.0
    max_experience: float = 50.0
    disqualifiers: list[str] = field(default_factory=list)     # phrases that penalize
