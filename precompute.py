import pickle
import numpy as np
from tqdm import tqdm
from loader import load_candidates
from features import extract_features
from semantic import model
import os

os.makedirs("artifacts", exist_ok=True)


print("Loading candidates...")

candidates = load_candidates(r"C:\Users\KIIT0001\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl")[:10]

print(f"Loaded {len(candidates)} candidates")



print("Extracting features...")

feature_objects = []

semantic_texts = []

candidate_ids = []

for candidate in tqdm(candidates, desc="Extracting features"):

    feature = extract_features(candidate)

    feature_objects.append(feature)

    semantic_texts.append(feature.semantic_text)

    candidate_ids.append(feature.candidate_id)



print("Generating embeddings...")

embeddings = model.encode(
    semantic_texts,
    batch_size=128,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)


print("Saving artifacts...")

np.save("artifacts/resume_embeddings.npy", embeddings)

np.save("artifacts/candidate_ids.npy", np.array(candidate_ids))

with open("artifacts/features.pkl", "wb") as f:
    pickle.dump(feature_objects, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Done!")