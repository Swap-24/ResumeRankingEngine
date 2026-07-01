import pickle
import numpy as np

from semantic import ( embed, similarity)

resume_embeddings = np.load("artifacts/resume_embeddings.npy", mmap_mode="r")

candidate_ids = np.load("artifacts/candidate_ids.npy")

with open("artifacts/features.pkl","rb") as f:
    features = pickle.load(f)

feature_lookup = {feature.candidate_id: feature for feature in features}

def retrieve(job_embedding, top_k=3000):

    scores = resume_embeddings @ job_embedding

    indices = np.argpartition(scores, -top_k)[-top_k:]

    indices = indices[np.argsort(scores[indices])[::-1]]

    return [(float(scores[i]), feature_lookup[candidate_ids[i]]) for i in indices]