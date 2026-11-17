
import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

TOKEN = os.getenv("TMDB_BEARER_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "accept": "application/json"
}


def _get_json(endpoint, params=None):
    """Send a request to TMDB and return JSON data."""

    if not TOKEN:
        return None

    try:
        response = requests.get(
            f"{BASE_URL}{endpoint}",
            headers=HEADERS,
            params=params,
            timeout=15
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return None


def search_movie(movie_name, year=None):
    """Search for a movie, optionally using its release year."""

    params = {
        "query": movie_name,
        "include_adult": False
    }

    if year:
        params["year"] = year

    data = _get_json("/search/movie", params)

    if not data:
        return None

    results = data.get("results", [])

    if not results:
        return None

    # Prefer an exact release-year match.
    if year:
        for movie in results:
            release_date = movie.get("release_date", "")
            if release_date.startswith(str(year)):
                return movie

    return results[0]


def get_movie_details(movie_id):
    """Retrieve movie details from TMDB."""

    return _get_json(f"/movie/{movie_id}")


def get_movie_credits(movie_id):
    """Retrieve cast and crew information from TMDB."""

    return _get_json(f"/movie/{movie_id}/credits")


def get_poster_url(poster_path):
    """Generate a full movie poster URL."""

    if not poster_path:
        return None

    return f"{IMAGE_BASE_URL}{poster_path}"
