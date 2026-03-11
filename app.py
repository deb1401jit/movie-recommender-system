from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime

import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
LIVE_CACHE_TTL = 900
BLOCKED_TITLE_TOKENS = [
    "parody",
    "spoof",
    "bts",
    "behind the scenes",
    "behind-the-scenes",
    "making of",
    "featurette",
    "outtakes",
    "bloopers",
    "special",
]

load_dotenv()


def tmdb_get(path: str, api_key: str, params: dict | None = None) -> dict:
    if not path.startswith("/"):
        path = "/" + path
    query = {"api_key": api_key, "language": "en-US"}
    if params:
        query.update(params)
    response = requests.get(f"{TMDB_BASE_URL}{path}", params=query, timeout=30)
    response.raise_for_status()
    return response.json()


@dataclass
class MovieOption:
    label: str
    movie_id: int
    release_date: str | None


def format_year(release_date: str | None) -> str:
    if not release_date:
        return "Unknown year"
    try:
        return str(datetime.strptime(release_date, "%Y-%m-%d").year)
    except ValueError:
        return release_date


def is_released(release_date: str | None) -> bool:
    if not release_date:
        return False
    try:
        return datetime.strptime(release_date, "%Y-%m-%d").date() <= date.today()
    except ValueError:
        return False


def normalize(values: list[str]) -> list[str]:
    return [value.replace(" ", "").lower() for value in values if value]


def extract_names(items: list[dict], key: str = "name", limit: int | None = None) -> list[str]:
    names = [item.get(key) for item in items if item.get(key)]
    if limit is not None:
        names = names[:limit]
    return names


def extract_ids(items: list[dict], key: str = "id", limit: int | None = None) -> list[int]:
    ids = [item.get(key) for item in items if item.get(key)]
    if limit is not None:
        ids = ids[:limit]
    return ids


def build_soup(
    genres: list[str],
    keywords: list[str],
    cast: list[str],
    crew: list[str],
    overview: str,
) -> str:
    tokens = normalize(genres + keywords + cast + crew)
    return " ".join(tokens) + " " + overview.lower()


@st.cache_data(ttl=LIVE_CACHE_TTL, show_spinner=False)
def search_movies(query: str, api_key: str) -> list[dict]:
    payload = tmdb_get("/search/movie", api_key, params={"query": query, "page": 1})
    return payload.get("results", [])


@st.cache_data(ttl=LIVE_CACHE_TTL, show_spinner=False)
def fetch_movie_details(movie_id: int, api_key: str) -> dict:
    return tmdb_get(f"/movie/{movie_id}", api_key)


@st.cache_data(ttl=LIVE_CACHE_TTL, show_spinner=False)
def fetch_movie_credits(movie_id: int, api_key: str) -> dict:
    return tmdb_get(f"/movie/{movie_id}/credits", api_key)


@st.cache_data(ttl=LIVE_CACHE_TTL, show_spinner=False)
def fetch_movie_keywords(movie_id: int, api_key: str) -> dict:
    return tmdb_get(f"/movie/{movie_id}/keywords", api_key)


@st.cache_data(ttl=LIVE_CACHE_TTL, show_spinner=False)
def fetch_movie_bundle(movie_id: int, api_key: str) -> dict:
    return tmdb_get(
        f"/movie/{movie_id}",
        api_key,
        params={"append_to_response": "credits,keywords"},
    )


@st.cache_data(ttl=LIVE_CACHE_TTL, show_spinner=False)
def fetch_collection(collection_id: int, api_key: str) -> list[dict]:
    payload = tmdb_get(f"/collection/{collection_id}", api_key)
    return payload.get("parts", [])


@st.cache_data(ttl=LIVE_CACHE_TTL, show_spinner=False)
def build_movie_record(movie_id: int, api_key: str, bundle: dict | None = None) -> dict:
    details = bundle or fetch_movie_bundle(movie_id, api_key)
    credits = details.get("credits", {})
    keywords_payload = details.get("keywords", {})
    keyword_items = keywords_payload.get("keywords") or keywords_payload.get("results") or []

    genres = extract_names(details.get("genres", []))
    keywords_list = extract_names(keyword_items)
    cast = extract_names(credits.get("cast", []), limit=6)
    crew = [
        member.get("name")
        for member in credits.get("crew", [])
        if member.get("job") == "Director" and member.get("name")
    ]
    countries = [
        country.get("iso_3166_1")
        for country in details.get("production_countries", [])
        if country.get("iso_3166_1")
    ]

    overview = details.get("overview") or ""
    soup = build_soup(genres, keywords_list, cast, crew, overview)

    return {
        "id": details.get("id"),
        "title": details.get("title"),
        "release_date": details.get("release_date"),
        "poster_path": details.get("poster_path"),
        "vote_average": details.get("vote_average"),
        "overview": overview,
        "soup": soup,
        "original_language": details.get("original_language"),
        "countries": countries,
        "keywords": keywords_list,
        "video": details.get("video"),
    }


@st.cache_data(ttl=LIVE_CACHE_TTL, show_spinner=False)
def discover_candidates(
    seed_details: dict,
    seed_credits: dict,
    seed_keywords: dict,
    api_key: str,
    pages: int,
    seed_language: str | None,
    seed_countries: list[str],
    language_pref: str,
    region_pref: str,
) -> list[int]:
    genre_ids = extract_ids(seed_details.get("genres", []), limit=2)
    cast_ids = extract_ids(seed_credits.get("cast", []), limit=3)
    director_ids = extract_ids(
        [member for member in seed_credits.get("crew", []) if member.get("job") == "Director"],
        limit=2,
    )
    keyword_items = seed_keywords.get("keywords") or seed_keywords.get("results") or []
    keyword_ids = extract_ids(keyword_items, limit=4)

    base_params = {
        "sort_by": "popularity.desc",
        "include_adult": "false",
    }
    if seed_language and language_pref != "Any language":
        base_params["with_original_language"] = seed_language
    if seed_countries and region_pref != "Any region":
        base_params["with_origin_country"] = seed_countries[0]

    query_sets: list[dict] = []
    if genre_ids:
        query_sets.append({"with_genres": ",".join(str(g) for g in genre_ids)})
    if keyword_ids:
        query_sets.append({"with_keywords": ",".join(str(k) for k in keyword_ids)})
    if cast_ids:
        query_sets.append({"with_cast": ",".join(str(c) for c in cast_ids)})
    if director_ids:
        query_sets.append({"with_crew": ",".join(str(d) for d in director_ids)})
    if genre_ids and keyword_ids:
        query_sets.append(
            {
                "with_genres": ",".join(str(g) for g in genre_ids),
                "with_keywords": ",".join(str(k) for k in keyword_ids[:3]),
            }
        )
    if not query_sets:
        query_sets.append({})

    movie_ids: list[int] = []
    seen_ids: set[int] = set()

    for params in query_sets:
        for page in range(1, pages + 1):
            merged_params = {"page": page, **base_params, **params}
            payload = tmdb_get("/discover/movie", api_key, params=merged_params)
            for item in payload.get("results", []):
                movie_id = item.get("id")
                if not movie_id or movie_id in seen_ids:
                    continue
                movie_ids.append(movie_id)
                seen_ids.add(movie_id)

    return movie_ids


def rank_by_similarity(seed_record: dict, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    corpus = [seed_record["soup"]] + [movie["soup"] for movie in candidates]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(corpus)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    for movie, score in zip(candidates, scores):
        movie["score"] = float(score)
    return sorted(candidates, key=lambda item: item.get("score", 0), reverse=True)


def matches_language(record: dict, seed_language: str | None) -> bool:
    if not seed_language:
        return False
    return record.get("original_language") == seed_language


def matches_region(record: dict, seed_countries: list[str]) -> bool:
    if not seed_countries:
        return False
    record_countries = record.get("countries") or []
    return any(code in seed_countries for code in record_countries)


def preference_match_score(
    record: dict,
    seed_language: str | None,
    seed_countries: list[str],
    language_pref: str,
    region_pref: str,
) -> int:
    score = 0
    if language_pref == "Prefer same language" and matches_language(record, seed_language):
        score += 1
    if region_pref == "Prefer same region" and matches_region(record, seed_countries):
        score += 1
    return score


def is_unwanted_content(record: dict) -> bool:
    title = (record.get("title") or "").lower()
    overview = (record.get("overview") or "").lower()
    keywords = [kw.lower() for kw in record.get("keywords") or []]
    if record.get("video") is True:
        return True
    for token in BLOCKED_TITLE_TOKENS:
        if token in title or token in overview:
            return True
    for token in BLOCKED_TITLE_TOKENS:
        if any(token in kw for kw in keywords):
            return True
    return False


st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommender System")
st.write("Search a movie title and get personalized recommendations.")

speed_option = st.sidebar.selectbox(
    "Scan depth (more movies = better recommendations, slower)",
    ["Fast (20 movies)", "Balanced (40 movies)", "Thorough (60 movies)", "Deep (100 movies)"],
    index=1,
)
pages_by_option = {
    "Fast (20 movies)": 1,
    "Balanced (40 movies)": 2,
    "Thorough (60 movies)": 3,
    "Deep (100 movies)": 5,
}
pages = pages_by_option[speed_option]
result_count = st.sidebar.slider("Results to show", min_value=5, max_value=20, value=10)
prioritize_franchise = st.sidebar.checkbox("Prioritize franchise sequels", value=True)
language_pref = st.sidebar.selectbox(
    "Language filter",
    ["Same language only", "Prefer same language", "Any language"],
    index=0,
)
region_pref = st.sidebar.selectbox(
    "Region preference",
    ["Prefer same region", "Same region only", "Any region"],
    index=0,
)
exclude_bonus = True

api_key = os.getenv("TMDB_API_KEY", "").strip()
if not api_key:
    api_key = st.sidebar.text_input("TMDB API key", type="password").strip()
if not api_key:
    st.error("TMDB_API_KEY is missing. Add it to .env or the sidebar.")
    st.stop()

query = st.text_input("Search for a movie", placeholder="e.g., Avatar")
if not query:
    st.info("Start typing a movie name to see results.")
    st.stop()

with st.spinner("Searching TMDB..."):
    search_results = search_movies(query, api_key)

if not search_results:
    st.warning("No movies found. Try a different title.")
    st.stop()

options: list[MovieOption] = []
for item in search_results:
    title = item.get("title")
    movie_id = item.get("id")
    release_date = item.get("release_date")
    if not title or not movie_id:
        continue
    year = format_year(release_date)
    options.append(
        MovieOption(label=f"{title} ({year})", movie_id=movie_id, release_date=release_date)
    )

selected_label = st.selectbox("Choose the exact movie", [opt.label for opt in options])
selected_movie = next(opt for opt in options if opt.label == selected_label)

with st.spinner("Building recommendations..."):
    seed_bundle = fetch_movie_bundle(selected_movie.movie_id, api_key)
    seed_details = seed_bundle
    seed_credits = seed_bundle.get("credits", {})
    seed_keywords = seed_bundle.get("keywords", {})
    seed_language = seed_details.get("original_language") if seed_details else None
    seed_countries = [
        country.get("iso_3166_1")
        for country in seed_details.get("production_countries", [])
        if country.get("iso_3166_1")
    ]
    seed_record = build_movie_record(selected_movie.movie_id, api_key, bundle=seed_bundle)

    candidate_ids = discover_candidates(
        seed_details,
        seed_credits,
        seed_keywords,
        api_key,
        pages,
        seed_language,
        seed_countries,
        language_pref,
        region_pref,
    )
    candidate_ids = [mid for mid in candidate_ids if mid != selected_movie.movie_id]

    candidate_records: list[dict] = []
    for movie_id in candidate_ids:
        record = build_movie_record(movie_id, api_key)
        if not record.get("id"):
            continue
        if not is_released(record.get("release_date")):
            continue
        if exclude_bonus and is_unwanted_content(record):
            continue
        if language_pref == "Same language only" and seed_language:
            if not matches_language(record, seed_language):
                continue
        if region_pref == "Same region only" and seed_countries:
            if not matches_region(record, seed_countries):
                continue
        candidate_records.append(record)

    ranked = rank_by_similarity(seed_record, candidate_records)
    if language_pref == "Prefer same language" or region_pref == "Prefer same region":
        ranked = sorted(
            ranked,
            key=lambda item: (
                preference_match_score(
                    item, seed_language, seed_countries, language_pref, region_pref
                ),
                item.get("score", 0),
            ),
            reverse=True,
        )

    prioritized: list[dict] = []
    if prioritize_franchise:
        collection = seed_details.get("belongs_to_collection") if seed_details else None
        if collection and collection.get("id"):
            parts = fetch_collection(collection["id"], api_key)
            for item in parts:
                movie_id = item.get("id")
                if not movie_id or movie_id == selected_movie.movie_id:
                    continue
                record = build_movie_record(movie_id, api_key)
                if not is_released(record.get("release_date")):
                    continue
                if exclude_bonus and is_unwanted_content(record):
                    continue
                if language_pref == "Same language only" and seed_language:
                    if not matches_language(record, seed_language):
                        continue
                if region_pref == "Same region only" and seed_countries:
                    if not matches_region(record, seed_countries):
                        continue
                prioritized.append(record)

    combined: list[dict] = []
    seen_ids: set[int] = set()
    for movie in prioritized + ranked:
        movie_id = movie.get("id")
        if not movie_id or movie_id in seen_ids:
            continue
        combined.append(movie)
        seen_ids.add(movie_id)

    recs = combined[:result_count]

if not recs:
    st.info("No recommendations found. Try another title.")
    st.stop()

st.subheader("Recommended movies")
for rec in recs:
    cols = st.columns([1, 3])
    with cols[0]:
        poster_path = rec.get("poster_path")
        if poster_path:
            st.image(f"{POSTER_BASE_URL}{poster_path}", width=140)
        else:
            st.write("No poster")
    with cols[1]:
        st.markdown(f"**{rec.get('title', 'Untitled')}**")
        st.write(f"Release date: {rec.get('release_date', 'Unknown')}")
        st.write(f"TMDB rating: {rec.get('vote_average', 'N/A')}")
        overview = rec.get("overview") or "No overview available."
        st.write(overview)
