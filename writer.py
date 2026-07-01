import csv


def write_submission(results, filename="submission.csv"):

    with open(filename,"w",newline="",encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["candidate_id","score"])

        for score, feature in results:

            writer.writerow([feature.candidate_id,round(score, 4)])