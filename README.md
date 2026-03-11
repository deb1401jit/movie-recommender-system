# Movie Recommendation System

Live movie recommender built with Python and the TMDB API. Search a movie title and get recommendations ranked by our own similarity model.

**Features**
- Live TMDB search for movie titles
- Candidate pool expanded via genres, cast/crew overlap, and keywords
- TF-IDF similarity using genres, keywords, cast, crew, and overview
- Franchise-first ranking to surface sequels early
- Streamlit web UI for interactive recommendations

**Quick Start**
1. Create a virtual environment and install dependencies.
2. Set your TMDB API key in the environment.
3. Run the Streamlit app. It will build recommendations live from TMDB data.

```bash
python -m venv MovieRecom
.\MovieRecom\Scripts\activate
pip install -r requirements.txt
```

```bash
set TMDB_API_KEY=your_key_here
streamlit run app.py
```

**Environment**
- Create a `.env` file if you prefer, and add `TMDB_API_KEY=your_key_here`.

**Deployment (Streamlit Community Cloud)**
1. Create a new app from this GitHub repo.
2. Set the main file to `app.py`.
3. Add a secret in TOML format:
   ```
   TMDB_API_KEY = "your_key_here"
   ```
4. Deploy.

**Project Layout**
- `app.py` Streamlit application
- `config.py` App constants
- `tmdb_client.py` TMDB API helpers
- `recommender.py` Candidate discovery + similarity ranking
- `utils.py` Utility helpers
