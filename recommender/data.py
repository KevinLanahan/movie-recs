import io, zipfile, requests, pandas as pd, numpy as np
from pathlib import Path

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

def ensure_data():
    ratings_csv = DATA_DIR / "ratings.csv"
    movies_csv  = DATA_DIR / "movies.csv"
    if ratings_csv.exists() and movies_csv.exists():
        return ratings_csv, movies_csv
    r = requests.get(ML_URL, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in ["ml-latest-small/ratings.csv", "ml-latest-small/movies.csv"]:
        with z.open(name) as f, open(DATA_DIR / Path(name).name, "wb") as out:
            out.write(f.read())
    return ratings_csv, movies_csv

def load_data():
    ratings_csv, movies_csv = ensure_data()
    ratings = pd.read_csv(ratings_csv)  
    movies  = pd.read_csv(movies_csv)   
    return ratings, movies

def train_test_split_time(ratings, test_frac=0.2):
    ratings = ratings.sort_values("timestamp")
    def split_user(df):
        n_test = max(1, int(len(df)*test_frac))
        return df.iloc[:-n_test], df.iloc[-n_test:]
    train_list, test_list = [], []
    for _, g in ratings.groupby("userId"):
        tr, te = split_user(g)
        train_list.append(tr); test_list.append(te)
    return pd.concat(train_list), pd.concat(test_list)
