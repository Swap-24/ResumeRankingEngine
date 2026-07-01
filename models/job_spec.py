from dataclasses import dataclass, field


@dataclass(slots=True)
class JobSpec:
    
    title: str
    full_text: str                          
    intent_text: str                        
    required_skills: list[str] = field(default_factory=list)   
    preferred_skills: list[str] = field(default_factory=list)  
    min_experience: float = 0.0
    max_experience: float = 50.0
    disqualifiers: list[str] = field(default_factory=list)    