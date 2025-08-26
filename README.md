# Movie Recommender (CF + optional content)

## Quick start
# 1) Clone + env
git clone https://github.com/<you>/movie-recs.git
cd movie-recs
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Run (auto-downloads MovieLens small on first run)
python app.py
# visit http://127.0.0.1:5000 and try user_id=1,2,3,...

# CLI:
python cli.py --user 1 -k 10
python cli.py --movie 1 -k 10
