from dataclasses import dataclass


@dataclass(slots=True)
class Candidate:

    candidate_id: str

    raw: dict