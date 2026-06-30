from models.candidate import Candidate
from models.job import Job

def experience_score(candidate: Candidate, job: Job) -> float:

    years = candidate.years_of_experience

    if years < job.min_experience:
        return 0

    if years > job.max_experience:
        return 15

    return 20

def skill_score(candidate: Candidate, job: Job) -> float:

    candidate_skills = {
        skill.lower()
        for skill in candidate.resume.skills
    }

    required = {
        skill.lower()
        for skill in job.required_skills
    }

    if not required:
        return 0

    matched = len(candidate_skills & required)

    return (matched / len(required)) * 40

def location_score(candidate: Candidate, job: Job) -> float:

    if not job.location:
        return 0

    if candidate.location.lower() == job.location.lower():
        return 10

    return 0

def behavior_score(candidate: Candidate) -> float:

    score = 0

    if candidate.open_to_work:
        score += 5

    score += candidate.github_activity / 20

    score += candidate.recruiter_response_rate * 5

    return score

def overall_score(candidate: Candidate, job: Job) -> float:

    return (

        experience_score(candidate, job)

        + skill_score(candidate, job)

        + location_score(candidate, job)

        + behavior_score(candidate)

    )