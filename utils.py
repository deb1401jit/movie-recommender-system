from __future__ import annotations

from datetime import date, datetime


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
