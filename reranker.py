def rerank(results, job):

    ranked = []

    required = {skill.lower() for skill in job.required_skills}

    for semantic_score, feature in results:

        score = semantic_score * 0.45

        if feature.years_experience >= job.min_experience:
            score += 10

        overlap = len(feature.skills & required)

        if required:

            score += (overlap / len(required)) * 15

        score += (feature.github_score / 100) * 5

        score += (feature.recruiter_response_rate) * 5

        score += (feature.interview_completion_rate) * 5

        score += (feature.offer_acceptance_rate) * 5

        if feature.open_to_work:
            score += 5

        ranked.append((score, feature))

    ranked.sort(reverse=True, key=lambda x: x[0])

    return ranked[:100]