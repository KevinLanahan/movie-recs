import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import pickle

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

class CFModel:
    def __init__(self, ratings, k=30, metric="cosine"):
        self.user_ids = np.sort(ratings["userId"].unique())
        self.movie_ids = np.sort(ratings["movieId"].unique())
        uindex = {u:i for i,u in enumerate(self.user_ids)}
        mindex = {m:i for i,m in enumerate(self.movie_ids)}
        rows = ratings["userId"].map(uindex).to_numpy()
        cols = ratings["movieId"].map(mindex).to_numpy()
        vals = ratings["rating"].to_numpy(dtype=np.float32)
        self.R = csr_matrix((vals, (rows, cols)), shape=(len(self.user_ids), len(self.movie_ids)))
        self.knn = NearestNeighbors(n_neighbors=min(k, self.R.shape[0]-1), metric=metric, algorithm="brute")
        self.knn.fit(self.R)

    def recommend_for_user(self, user_id, top_k=10, exclude_seen=True):
        if user_id not in set(self.user_ids): return []
        uidx = np.where(self.user_ids == user_id)[0][0]
        distances, indices = self.knn.kneighbors(self.R[uidx], return_distance=True)
        neighbor_idxs = indices.flatten()[1:] 
        neighbor_d = distances.flatten()[1:]
        weights = 1 - neighbor_d
        neighbor_R = self.R[neighbor_idxs]
        scores = neighbor_R.multiply(weights[:,None]).sum(axis=0) / (weights.sum() + 1e-8)
        scores = np.asarray(scores).ravel()
        if exclude_seen:
            seen = self.R[uidx].indices
            scores[seen] = -np.inf
        top_cols = np.argpartition(-scores, kth=min(top_k, len(scores)-1))[:top_k]
        top_scores = scores[top_cols]
        order = np.argsort(-top_scores)
        return [(int(self.movie_ids[top_cols[i]]), float(top_scores[i])) for i in order]

def save(model, path=MODEL_DIR/"knn_cf.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load(path=MODEL_DIR/"knn_cf.pkl"):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
