from flask import Flask, request, jsonify, render_template, abort
from pathlib import Path
import re, difflib

print("Importing app.py...")

try:
    from recommender.data import load_data
    from recommender.build_cf import CFModel, save as save_cf, load as load_cf
    from recommender.build_content import ContentModel, save as save_ct, load as load_ct
    from recommender.hybrid import blend
except Exception as e:
    print("Import failure:", e)
    raise

app = Flask(__name__)

# Globals
movies_cache = {}
cf_model = None
ct_model = None
movies_df = None
titles_norm = {}   # movieId -> normalized title
titles_index = {}  # normalized title -> movieId (first seen)

def _normalize_title(t: str) -> str:
    # "Toy Story (1995)" -> "toy story"
    return re.sub(r"\s*\(\d{4}\)$", "", t or "").strip().lower()

def boot():
    """Load data, train/load models, and build title lookup tables."""
    global movies_cache, cf_model, ct_model, movies_df, titles_norm, titles_index
    print("Boot: loading data...")
    ratings, movies = load_data()
    movies_df = movies.copy()

    # titles cache for display
    movies_cache = movies_df.set_index("movieId")["title"].to_dict()

    # model paths
    model_path = Path("models/knn_cf.pkl")
    content_path = Path("models/content.pkl")

    # CF model
    if model_path.exists():
        print("Boot: loading CF model...")
        cf_model = load_cf(model_path)
    else:
        print("Boot: training CF model...")
        cf_model = CFModel(ratings)
        save_cf(cf_model, model_path)
        print("Boot: CF saved.")

    # Content model
    if content_path.exists():
        print("Boot: loading content model...")
        ct_model = load_ct(content_path)
    else:
        print("Boot: training content model...")
        ct_model = ContentModel(movies_df)
        save_ct(ct_model, content_path)
        print("Boot: content saved.")

    # ---- build title lookup tables for search ----
    print("Boot: building title lookup...")
    titles_norm = {int(mid): _normalize_title(t) for mid, t in zip(movies_df.movieId, movies_df.title)}
    titles_index = {}
    for mid, norm in titles_norm.items():
        if norm and norm not in titles_index:
            titles_index[norm] = int(mid)

boot()
print("Boot complete.")

# ------------------- routes -------------------

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/recs")
def api_recs():
    user_id = int(request.args.get("user_id", 1))
    top_k = int(request.args.get("k", 10))
    out = [{"movieId": m, "title": movies_cache.get(m, str(m)), "score": s}
           for m, s in cf_model.recommend_for_user(user_id, top_k=top_k)]
    return jsonify(out)

@app.get("/api/similar")
def api_similar():
    movie_id = int(request.args.get("movie_id", 1))
    top_k = int(request.args.get("k", 10))
    out = [{"movieId": m, "title": movies_cache.get(m, str(m)), "score": s}
           for m, s in ct_model.similar_movies(movie_id, top_k=top_k)]
    return jsonify(out)

@app.get("/api/search_title")
def api_search_title():
    if not titles_norm:
        return jsonify([])  # safety
    q = (request.args.get("q") or "").strip().lower()
    if not q:
        return jsonify([])
    population = list(set(titles_norm.values()))  # normalized titles
    matches = difflib.get_close_matches(q, population, n=8, cutoff=0.6)
    results = []
    for norm in matches:
        mid = titles_index.get(norm)
        if mid is not None:
            results.append({"movieId": mid, "title": movies_cache.get(mid, str(mid))})
    return jsonify(results)

@app.get("/api/similar_by_title")
def api_similar_by_title():
    if not titles_norm:
        return jsonify({"match": None, "results": []})
    q = (request.args.get("q") or "").strip().lower()
    k = int(request.args.get("k", 10))
    if not q:
        abort(400, "Missing q")
    population = list(set(titles_norm.values()))
    best = difflib.get_close_matches(q, population, n=1, cutoff=0.5)
    if not best:
        return jsonify({"match": None, "results": []})
    norm = best[0]
    mid = titles_index.get(norm)
    similar = ct_model.similar_movies(mid, top_k=k)
    out = [{"movieId": m, "title": movies_cache.get(m, str(m)), "score": s} for m, s in similar]
    return jsonify({"match": {"movieId": mid, "title": movies_cache.get(mid, str(mid))}, "results": out})

if __name__ == "__main__":
    app.run(debug=True)
