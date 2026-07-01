

import re
from pathlib import Path
from docx import Document
from models.job_spec import JobSpec

_REQUIRED_HEADING_TOKENS = {
    "absolutely need", "must have", "mandatory", "required", "non-negotiable",
    "core skill", "minimum qualif", "basic qualif", "key skill", "essential",
    "things you need", "you need", "need to have",
}

# Preferred skills
_PREFERRED_HEADING_TOKENS = {
    "nice to have", "preferred", "bonus", "good to have", "like you to have",
    "advantageous", "desirable", "would like", "like to see",
}

# Disqualifiers
_DISQUALIFIER_HEADING_TOKENS = {
    "do not want", "not want", "disqualif", "will not", "not looking",
    "not suitable", "explicitly do not", "we don't want",
}

# YOE regex
_YOE_RANGE  = re.compile(r"(\d+)\s*[–\-to]+\s*(\d+)\s*(?:years?|yrs?)", re.IGNORECASE)
_YOE_MIN    = re.compile(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)", re.IGNORECASE)


def parse_jd(path: str) -> JobSpec:
    doc_path = Path(path)
    if not doc_path.exists():
        raise FileNotFoundError(f"JD file not found: {path}")

    doc = Document(str(doc_path))
    paragraphs = doc.paragraphs  # preserve style info

    full_text = "\n".join(p.text.strip() for p in paragraphs if p.text.strip())
    title = _extract_title(paragraphs)

    min_exp, max_exp = _extract_yoe(full_text)

    intent_sentences = _extract_sentences(full_text)[:6]
    intent_text = f"{title}. " + " ".join(intent_sentences)

    required_skills, preferred_skills, disqualifiers = _parse_skill_sections(paragraphs)

    if not required_skills:
        required_skills = _fallback_skill_extract(full_text)

    return JobSpec(
        title=title,
        full_text=full_text,
        intent_text=intent_text,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_experience=min_exp,
        max_experience=max_exp,
        disqualifiers=disqualifiers,
    )


def _extract_title(paragraphs) -> str:
    """Return the first non-empty paragraph styled as Title or Heading."""
    for p in paragraphs:
        if not p.text.strip():
            continue
        style = p.style.name.lower()
        if "title" in style or "heading" in style:
            # Clean up: strip "Job Description:" prefix if present
            raw = p.text.strip()
            raw = re.sub(r"^job\s+description\s*:\s*", "", raw, flags=re.IGNORECASE)
            return raw
    # Fallback: first non-empty paragraph
    for p in paragraphs:
        if p.text.strip():
            return p.text.strip()[:120]
    return "Unknown Role"


def _parse_skill_sections(paragraphs) -> tuple[list[str], list[str], list[str]]:
    required: list[str] = []
    preferred: list[str] = []
    disqualifiers: list[str] = []

    mode = "none"

    for p in paragraphs:
        text = p.text.strip()
        if not text:
            continue

        style = p.style.name.lower()
        text_lower = text.lower()

        if "heading" in style:
            if _matches_any(text_lower, _REQUIRED_HEADING_TOKENS):
                mode = "required"
            elif _matches_any(text_lower, _PREFERRED_HEADING_TOKENS):
                mode = "preferred"
            elif _matches_any(text_lower, _DISQUALIFIER_HEADING_TOKENS):
                mode = "disqualifier"
            else:
                mode = "none"
            continue

        if mode == "none":
            continue

        is_list = "list" in style or "bullet" in style or "number" in style
        is_short_normal = style == "normal" and len(text) < 250

        if not (is_list or is_short_normal):
            continue

        if mode == "required":
            tokens = _extract_skill_tokens_from_bullet(text)
            required.extend(tokens)
        elif mode == "preferred":
            tokens = _extract_skill_tokens_from_bullet(text)
            preferred.extend(tokens)
        elif mode == "disqualifier":
            disqualifiers.append(_summarize_disqualifier(text))

    return (
        _deduplicate(required),
        _deduplicate(preferred),
        _deduplicate(disqualifiers),
    )


def _extract_skill_tokens_from_bullet(text: str) -> list[str]:
    tokens: list[str] = []

    parens = re.findall(r"\(([^)]+)\)", text)
    for paren in parens:
        inner_tokens = [t.strip().lower() for t in re.split(r"[,/]", paren)]
        tokens.extend(t for t in inner_tokens if _is_skill_like(t))

    clean = re.sub(r"\([^)]*\)", "", text).strip()
    clean = re.sub(r"^[\u2022\u2013\u2014\-\*\>\•·]\s*", "", clean).strip()

    parts = re.split(r"[,;/]|\band\b|\bor\b", clean, flags=re.IGNORECASE)
    for part in parts:
        part = part.strip().lower()
        phrase = _extract_leading_skill(part)
        if phrase and _is_skill_like(phrase):
            tokens.append(phrase)

    return tokens


def _extract_leading_skill(text: str) -> str:
    text = re.sub(
        r"^(production|hands.on|strong|prior|experience with|exposure to|"
        r"background in|solid|deep|working knowledge of|familiarity with)\s+",
        "", text, flags=re.IGNORECASE
    ).strip()
    words = text.split()[:5]
    phrase = " ".join(words).rstrip(".,;:")
    return phrase[:60]


def _is_skill_like(text: str) -> bool:
    if not text or len(text) < 2 or len(text) > 60:
        return False
    stopwords = {
        "the", "this", "that", "we", "our", "your", "you", "are", "is", "will",
        "have", "has", "for", "and", "or", "but", "not", "with", "in", "on", "at",
        "by", "to", "of", "a", "an", "be", "as", "if", "it", "its", "do", "can",
        "may", "must", "should", "job", "role", "team", "company", "position",
        "candidate", "people", "someone", "things", "what", "how", "when", "where",
        "why", "work", "working", "use", "using", "based", "systems", "platform",
        "etc", "similar", "such",
    }
    first_word = text.split()[0].lower().strip(".,;:")
    if first_word in stopwords:
        return False
    return bool(re.search(r"[a-z0-9]{2,}", text))


def _summarize_disqualifier(text: str) -> str:
    text = text.lower()
    text = re.sub(r"^[\u2022\u2013\u2014\-\*\>\•·]\s*", "", text).strip()

    for prefix in ["if you've", "if you", "people who", "candidates who", "those who"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    return text[:150].strip()


def _extract_yoe(text: str) -> tuple[float, float]:
    m = _YOE_RANGE.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    matches = _YOE_MIN.findall(text)
    if matches:
        val = float(matches[0])
        return val, val + 10.0
    return 0.0, 50.0


def _extract_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _fallback_skill_extract(text: str) -> list[str]:
    """Fallback for plain-text JDs: extract capitalized noun phrases."""
    pattern = re.compile(r"\b([A-Z][a-zA-Z0-9+#\-\.]*(?:\s+[A-Z][a-zA-Z0-9+#\-\.]*){0,3})\b")
    candidates = pattern.findall(text)
    stopwords = {
        "The", "This", "We", "Our", "You", "Are", "Is", "Will", "Have", "For",
        "And", "Or", "In", "On", "At", "If", "It", "Do", "Can", "Must", "Should",
        "Job", "Role", "Team", "Company", "Position", "Candidate", "Good", "Bad",
    }
    skills = []
    for c in candidates:
        first_word = c.split()[0]
        if first_word not in stopwords and len(c) >= 3:
            skills.append(c.lower())
    return _deduplicate(skills)[:30]


def _matches_any(text: str, token_set: set[str]) -> bool:
    return any(token in text for token in token_set)


def _deduplicate(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
