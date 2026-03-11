from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://api.themoviedb.org/3"


def tmdb_get(path: str, api_key: str, params: dict | None = None) -> dict:
    if not path.startswith("/"):
        path = "/" + path
    query = {"api_key": api_key, "language": "en-US"}
    if params:
        query.update(params)
    response = requests.get(f"{BASE_URL}{path}", params=query, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_names(items: list[dict], key: str = "name", limit: int | None = None) -> list[str]:
    names = [item.get(key) for item in items if item.get(key)]
    if limit is not None:
        names = names[:limit]
    return names


def build_movie_record(movie_id: int, api_key: str) -> dict:
    details = tmdb_get(f"/movie/{movie_id}", api_key)
    credits = tmdb_get(f"/movie/{movie_id}/credits", api_key)
    keywords = tmdb_get(f"/movie/{movie_id}/keywords", api_key)

    genres = extract_names(details.get("genres", []))
    keywords_list = extract_names(keywords.get("keywords", []))
    cast = extract_names(credits.get("cast", []), limit=5)
    crew = [
        member.get("name")
        for member in credits.get("crew", [])
        if member.get("job") == "Director" and member.get("name")
    ]

    return {
        "id": details.get("id"),
        "title": details.get("title"),
        "overview": details.get("overview"),
        "genres": "|".join(genres),
        "keywords": "|".join(keywords_list),
        "cast": "|".join(cast),
        "crew": "|".join(crew),
        "release_date": details.get("release_date"),
        "vote_average": details.get("vote_average"),
        "vote_count": details.get("vote_count"),
        "popularity": details.get("popularity"),
        "poster_path": details.get("poster_path"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TMDB movie dataset.")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="TMDB API key. If omitted, uses TMDB_API_KEY env var.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="Number of pages to fetch from /movie/popular (20 movies per page).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between movie detail requests to be gentle on the API.",
    )
    parser.add_argument(
        "--output",
        default="data/movies.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    api_key = args.api_key or os.getenv("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("TMDB_API_KEY is missing. Set it or pass --api-key.")

    records: list[dict] = []
    seen_ids: set[int] = set()

    for page in range(1, args.pages + 1):
        payload = tmdb_get("/movie/popular", api_key, params={"page": page})
        for movie in payload.get("results", []):
            movie_id = movie.get("id")
            if not movie_id or movie_id in seen_ids:
                continue
            try:
                record = build_movie_record(movie_id, api_key)
            except requests.HTTPError as exc:
                print(f"Skipping {movie_id} due to HTTP error: {exc}")
                continue
            records.append(record)
            seen_ids.add(movie_id)
            if args.sleep:
                time.sleep(args.sleep)

    df = pd.DataFrame(records)
    df = df.dropna(subset=["title"]).drop_duplicates(subset=["id"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} movies to {output_path}")


if __name__ == "__main__":
    main()
