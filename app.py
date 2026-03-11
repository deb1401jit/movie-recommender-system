from __future__ import annotations

import html
import os
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv

from config import POSTER_BASE_URL
from recommender import (
    build_movie_record,
    discover_candidates,
    is_unwanted_content,
    matches_language,
    matches_region,
    preference_match_score,
    rank_by_similarity,
)
from tmdb_client import fetch_collection, fetch_movie_bundle, search_movies
from utils import format_year, is_released

load_dotenv()


@dataclass
class MovieOption:
    label: str
    movie_id: int
    release_date: str | None


st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommender System")
st.write("Search a movie title and get personalized recommendations.")

pages = 2
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
        st.markdown(f"<div style=\"font-size:1.2em;font-weight:600;\">{rec.get('title', 'Untitled')}</div>", unsafe_allow_html=True)
        st.write(f"Release date: {rec.get('release_date', 'Unknown')}")
        st.write(f"TMDB rating: {rec.get('vote_average', 'N/A')}")
        overview = rec.get("overview") or "No overview available."
        safe_overview = html.escape(overview).replace("\n", "<br>")
        st.markdown(f"<div>{safe_overview}</div>", unsafe_allow_html=True)
