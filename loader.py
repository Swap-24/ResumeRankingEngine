import json
import gzip
from pathlib import Path

from models.candidate import Candidate
from models.resume import Resume


def load_candidates(path: str) -> list[Candidate]:
    """
    Reads a .jsonl or .jsonl.gz file and returns
    a list of Candidate objects.
    """

    candidates = []

    # Decide how to open the file
    open_function = gzip.open if path.endswith(".gz") else open

    with open_function(path, "rt", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            # Ignore blank lines
            if not line:
                continue

            raw_candidate = json.loads(line)

            candidate = parse_candidate(raw_candidate)

            candidates.append(candidate)

    return candidates

def parse_candidate(raw: dict) -> Candidate:

    profile = raw.get("profile", {})
    signals = raw.get("redrob_signals", {})

    resume = Resume(
        summary=profile.get("summary", ""),

        skills=[
            skill.get("name", "")
            for skill in raw.get("skills", [])
        ],

        career_history=raw.get("career_history", []),

        education=raw.get("education", []),

        projects=raw.get("projects", [])
    )

    candidate = Candidate(

        candidate_id=raw.get("candidate_id", ""),

        resume=resume,

        location=profile.get("location", ""),

        github_activity=signals.get(
            "github_activity_score",
            0
        ),

        recruiter_response_rate=signals.get(
            "recruiter_response_rate",
            0
        ),

        open_to_work=signals.get(
            "open_to_work_flag",
            False
        )
    )

    return candidate