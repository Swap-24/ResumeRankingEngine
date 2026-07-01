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

    return Candidate(
        candidate_id=raw["candidate_id"],
        raw=raw
    )