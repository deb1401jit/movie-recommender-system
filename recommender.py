from __future__ import annotations

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import BLOCKED_TITLE_TOKENS, LIVE_CACHE_TTL
from tmdb_client import fetch_movie_bundle, tmdb_get
from utils import build_soup, extract_ids, extract_names


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
