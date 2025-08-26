import argparse, pandas as pd
from recommender.data import load_data
from recommender.build_cf import load as load_cf
from recommender.build_content import load as load_ct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, help="User ID for CF recs")
    ap.add_argument("--movie", type=int, help="Movie ID for similar (content)")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    _, movies = load_data(); titles = dict(zip(movies.movieId, movies.title))
    if args.user:
        cf = load_cf()
        out = cf.recommend_for_user(args.user, top_k=args.k)
        print("\nTop CF Recs:")
        for m,s in out: print(f"{titles.get(m,m)}\t{round(s,4)}")
    if args.movie:
        ct = load_ct()
        out = ct.similar_movies(args.movie, top_k=args.k)
        print("\nTop Similar Movies (Content):")
        for m,s in out: print(f"{titles.get(m,m)}\t{round(s,4)}")

if __name__ == "__main__":
    main()
