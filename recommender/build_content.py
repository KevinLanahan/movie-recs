import numpy as np, pandas as pd, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

class ContentModel:
    def __init__(self, movies):
        corpus = movies["genres"].fillna("").replace("(no genres listed)", "", regex=False)
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^|]+")
        X = self.vectorizer.fit_transform(corpus)
        self.sim = cosine_similarity(X)  
        self.movie_ids = movies["movieId"].to_numpy()
        self.index = {m:i for i,m in enumerate(self.movie_ids)}

    def similar_movies(self, movie_id, top_k=10):
        if movie_id not in self.index: return []
        i = self.index[movie_id]
        row = self.sim[i]
        row[i] = -np.inf
        cols = np.argpartition(-row, top_k)[:top_k]
        order = np.argsort(-row[cols])
        return [(int(self.movie_ids[cols[j]]), float(row[cols[j]])) for j in order]

def save(model, path=MODEL_DIR/"content.pkl"):
    with open(path, "wb") as f: pickle.dump(model, f)

def load(path=MODEL_DIR/"content.pkl"):
    with open(path, "rb") as f: return pickle.load(f)
