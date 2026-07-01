import csv
from pathlib import Path
from models.features import CandidateFeatures
from models.job_spec import JobSpec


def write_submission(
    ranked: list[tuple[float, CandidateFeatures]],
    job: JobSpec,
    output_path: str,
) -> None:
    if len(ranked) < 100:
        raise ValueError(f"Need at least 100 ranked candidates, got {len(ranked)}")

    top100 = ranked[:100]

    top100 = _enforce_monotonic(top100)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(str(out), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (score, feat) in enumerate(top100, start=1):
            reasoning = _build_reasoning(feat, job, score)
            writer.writerow([
                feat.candidate_id,
                rank,
                f"{score:.6f}",
                reasoning,
            ])

    print(f"[Writer] Submission written to: {out.resolve()}")
    print(f"[Writer] Top candidate: {top100[0][1].candidate_id} (score={top100[0][0]:.4f})")
    print(f"[Writer] #100 candidate: {top100[99][1].candidate_id} (score={top100[99][0]:.4f})")




def _build_reasoning(feat: CandidateFeatures, job: JobSpec, score: float) -> str:
    yoe = feat.years_experience
    title = feat.current_title or "Engineer"
    company = feat.current_company

    matched_skills = [
        s for s in (job.required_skills + job.preferred_skills)
        if _skill_in_candidate(s, feat)
    ]
    
    seen_s = set()
    unique_skills = []
    for s in matched_skills:
        clean = s.strip().lower()
        if clean and clean not in seen_s:
            seen_s.add(clean)
            unique_skills.append(clean)

    top_skills = unique_skills[:3]
    skills_phrase = ""
    if top_skills:
        h = hash(feat.candidate_id)
        phrasings = [
            f"offers hands-on experience with {', '.join(top_skills)}",
            f"brings strong expertise in {', '.join(top_skills)}",
            f"is highly proficient in {', '.join(top_skills)}",
            f"has solid experience utilizing {', '.join(top_skills)}",
            f"demonstrates deep knowledge of {', '.join(top_skills)}",
            f"shows a strong background in {', '.join(top_skills)}"
        ]
        skills_phrase = phrasings[abs(h) % len(phrasings)]

    h_idx = hash(feat.candidate_id + "title")
    if company:
        title_phrasings = [
            f"Currently working as a {title} at {company} with {yoe:.1f} years of experience",
            f"A {title} at {company} offering {yoe:.1f} years of experience",
            f"Brings {yoe:.1f} years of professional experience, currently serving as a {title} at {company}",
            f"Served as {title} at {company} for several years, compiling {yoe:.1f} total YOE"
        ]
    else:
        title_phrasings = [
            f"An experienced {title} with {yoe:.1f} years of background",
            f"Brings {yoe:.1f} years of experience working as a {title}",
            f"An active {title} with {yoe:.1f} years of industry experience"
        ]
    intro_phrase = title_phrasings[abs(h_idx) % len(title_phrasings)]

    rrr = int(feat.recruiter_response_rate * 100)
    notice = feat.notice_period_days
    open_work = feat.open_to_work
    
    notice_str = f"{notice}-day notice" if notice > 0 else "immediate availability"
    h_behav = hash(feat.candidate_id + "behav")
    
    if rrr >= 70:
        if open_work:
            behavioral_phrases = [
                f"They are actively open to work with a {notice_str} and an outstanding {rrr}% recruiter response rate.",
                f"The candidate is highly responsive ({rrr}% response rate), open to new opportunities, and has a {notice_str}.",
                f"Features a stellar {rrr}% response rate, is open to work, and has a {notice_str} period."
            ]
        else:
            behavioral_phrases = [
                f"Maintains a high recruiter response rate of {rrr}% with a {notice_str} period.",
                f"Shows strong responsiveness on the platform ({rrr}% response rate) and has a {notice_str}.",
                f"Highly responsive user ({rrr}% response rate) available on a {notice_str} timeline."
            ]
    else:
        if open_work:
            behavioral_phrases = [
                f"Open to new roles with a {notice_str} and a {rrr}% response rate.",
                f"Actively seeking new opportunities with a {notice_str} timeline ({rrr}% RRR).",
                f"Is open to work on a {notice_str} basis, demonstrating a {rrr}% platform response rate."
            ]
        else:
            behavioral_phrases = [
                f"Available on a {notice_str} with a {rrr}% recruiter response rate.",
                f"Has a {notice_str} period and maintains a {rrr}% response rate.",
                f"Currently has a {notice_str} timeline and a platform response rate of {rrr}%."
            ]
    
    behav_phrase = behavioral_phrases[abs(h_behav) % len(behavioral_phrases)]

    h_comb = hash(feat.candidate_id + "comb")
    if skills_phrase:
        if abs(h_comb) % 2 == 0:
            reason = f"{intro_phrase}, who {skills_phrase}. {behav_phrase}"
        else:
            reason = f"{intro_phrase} and {skills_phrase}. {behav_phrase}"
    else:
        reason = f"{intro_phrase}. {behav_phrase}"

    reason = " ".join(reason.split())
    
    sentences = [s.strip() for s in reason.split(". ") if s.strip()]
    capitalized_sentences = []
    for s in sentences:
        if s:
            capitalized_sentences.append(s[0].upper() + s[1:])
    reason = ". ".join(capitalized_sentences)
    if not reason.endswith("."):
        reason += "."

    if len(reason) > 300:
        reason = reason[:297] + "..."

    return reason


def _skill_in_candidate(required_skill: str, feat: CandidateFeatures) -> bool:
    req = required_skill.lower()
    for s in feat.skills:
        if req == s or req in s or s in req:
            return True
    return False


def _enforce_monotonic(
    ranked: list[tuple[float, CandidateFeatures]],
) -> list[tuple[float, CandidateFeatures]]:
    if not ranked:
        return ranked

    result = [ranked[0]]
    prev_score = ranked[0][0]

    for score, feat in ranked[1:]:
        if score > prev_score:
            score = prev_score  
        result.append((score, feat))
        prev_score = score

    return result
