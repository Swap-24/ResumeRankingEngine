import json
import gzip
import json
from models.job import Job
from pathlib import Path
from models.candidate import Candidate
from models.resume import Resume


def load_candidates(path: str) -> list[Candidate]:

    candidates = []

    open_function = gzip.open if path.endswith(".gz") else open

    with open_function(path, "rt", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            raw_candidate = json.loads(line)

            candidate = parse_candidate(raw_candidate)

            candidates.append(candidate)

    return candidates

def parse_candidate(raw: dict) -> Candidate:
    return Candidate(candidate_id=raw.get("candidate_id", ""),raw=raw)



def load_job(path: str) -> Job:

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return Job(
        title=raw.get("title", ""),
        summary=raw.get("summary", ""),
        required_skills=raw.get("required_skills", []),
        preferred_skills=raw.get("preferred_skills", []),
        min_experience=raw.get("min_experience", 0),
        max_experience=raw.get("max_experience", 100),
        location=raw.get("location", "")
    )