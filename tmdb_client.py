from __future__ import annotations

import requests
import streamlit as st

from config import LIVE_CACHE_TTL, TMDB_BASE_URL


def tmdb_get(path: str, api_key: str, params: dict | None = None) -> dict:
    if not path.startswith("/"):
        path = "/" + path
    query = {"api_key": api_key, "language": "en-US"}
    if params:
        query.update(params)
    response = requests.get(f"{TMDB_BASE_URL}{path}", params=query, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=LIVE_CACHE_TTL, show_spinner=False)
def search_movies(query: str, api_key: str) -> list[dict]:
    payload = tmdb_get("/search/movie", api_key, params={"query": query, "page": 1})
    return payload.get("results", [])


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
