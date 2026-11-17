
import re
import pandas as pd
import streamlit as st

from pathlib import Path
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

from tmdb_api import (
    search_movie,
    get_movie_details,
    get_movie_credits,
    get_poster_url
)


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CineMatch | Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide"
)

DATA_DIR = Path(__file__).resolve().parent / "data"


# =====================================================
# 2. LOAD DATA AND BUILD FEATURES
# =====================================================

@st.cache_resource
def load_data_and_model():

    movies_path = DATA_DIR / "movies.csv"
    ratings_path = DATA_DIR / "ratings.csv"
    tags_path = DATA_DIR / "tags.csv"

    for path in [movies_path, ratings_path, tags_path]:

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing dataset file: {path}"
            )

        if path.stat().st_size == 0:
            raise ValueError(
                f"Dataset file is empty: {path}"
            )

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path)

    if not {"movieId", "title", "genres"}.issubset(movies.columns):
        raise ValueError("movies.csv has invalid columns.")

    if not {"movieId", "rating"}.issubset(ratings.columns):
        raise ValueError("ratings.csv has invalid columns.")

    if not {"movieId", "tag"}.issubset(tags.columns):
        raise ValueError("tags.csv has invalid columns.")

    movies = movies.dropna(
        subset=["movieId", "title"]
    ).copy()

    movies["genres"] = movies["genres"].fillna("").astype(str)

    # Combine all tags associated with each movie

    valid_tags = tags.dropna(
        subset=["movieId", "tag"]
    ).copy()

    tag_groups = (
        valid_tags.groupby("movieId")["tag"]
        .apply(lambda values: " ".join(values.astype(str)))
    )

    movies["tags"] = (
        movies["movieId"]
        .map(tag_groups)
        .fillna("")
    )

    # Normalize genre and tag text

    movies["genre_text"] = (
        movies["genres"]
        .str.replace("|", " ", regex=False)
        .str.lower()
        .str.strip()
    )

    movies["tags"] = (
        movies["tags"]
        .str.lower()
        .str.strip()
    )

    # Retain movies with usable content

    movies = movies[
        (movies["genre_text"] != "") |
        (movies["tags"] != "")
    ].copy()

    movies.reset_index(drop=True, inplace=True)

    if movies.empty:
        raise ValueError("No usable movie information was found.")

    # Genre features

    genre_vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        binary=True,
        use_idf=False,
        norm="l2"
    )

    genre_matrix = genre_vectorizer.fit_transform(
        movies["genre_text"]
    )

    # Tag and phrase features

    tag_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    try:

        tag_matrix = tag_vectorizer.fit_transform(
            movies["tags"]
        )

        combined_matrix = hstack([
            genre_matrix * 1.5,
            tag_matrix
        ]).tocsr()

    except ValueError:

        combined_matrix = genre_matrix.tocsr()

    combined_matrix = normalize(
        combined_matrix,
        norm="l2"
    )

    # Ratings are used for display only

    ratings["rating"] = pd.to_numeric(
        ratings["rating"],
        errors="coerce"
    )

    rating_stats = (
        ratings.dropna(
            subset=["movieId", "rating"]
        )
        .groupby("movieId")["rating"]
        .agg(
            average_rating="mean",
            rating_count="count"
        )
    )

    movies = movies.join(
        rating_stats,
        on="movieId"
    )

    movies["average_rating"] = (
        movies["average_rating"].round(2)
    )

    movies["rating_count"] = (
        movies["rating_count"]
        .fillna(0)
        .astype(int)
    )

    return movies, combined_matrix


# =====================================================
# 3. TITLE AND FRANCHISE HEURISTICS
# =====================================================

def normalize_title(title):
    """Normalize a title and remove its release year."""

    title = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        str(title)
    )

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        title.lower()
    ).strip()


def extract_year(title):
    """Extract release year from MovieLens title."""

    match = re.search(
        r"\((\d{4})\)\s*$",
        str(title)
    )

    return int(match.group(1)) if match else None


def title_similarity(title_a, title_b):
    """Estimate title resemblance, not verified franchise data."""

    a = normalize_title(title_a)
    b = normalize_title(title_b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    shorter, longer = sorted(
        [a, b],
        key=len
    )

    if len(shorter) >= 6 and longer.startswith(shorter + " "):
        return 1.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# =====================================================
# 4. RECOMMENDATION ENGINE
# =====================================================

def recommend_movies(
    selected_title,
    movies,
    feature_matrix
):

    matches = movies.index[
        movies["title"] == selected_title
    ].tolist()

    if not matches:
        return pd.DataFrame()

    selected_index = matches[0]

    selected_movie_id = (
        movies.iloc[selected_index]["movieId"]
    )

    # Content similarity using genres and tags

    content_scores = cosine_similarity(
        feature_matrix[selected_index],
        feature_matrix
    ).ravel()

    candidates = []

    for idx in range(len(movies)):

        candidate = movies.iloc[idx]

        # Exclude the input movie

        if candidate["movieId"] == selected_movie_id:
            continue

        if candidate["title"] == selected_title:
            continue

        content_score = float(content_scores[idx])

        series_score = title_similarity(
            selected_title,
            candidate["title"]
        )

        # Combined recommendation score

        final_score = (
            0.80 * content_score
            + 0.20 * series_score
        )

        candidates.append({

            "title": candidate["title"],

            "genres": candidate["genres"].replace(
                "|",
                ", "
            ),

            "similarity": content_score,

            "final_score": final_score,

            "average_rating": candidate["average_rating"],

            "rating_count": candidate["rating_count"]

        })

    # Rank by combined score

    candidates.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    # Select three distinct titles

    results = []
    seen_titles = set()

    for candidate in candidates:

        if candidate["title"] in seen_titles:
            continue

        seen_titles.add(candidate["title"])

        results.append(candidate)

        if len(results) == 3:
            break

    return pd.DataFrame(results)


# =====================================================
# 5. TMDB METADATA INTEGRATION
# =====================================================

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tmdb_metadata(movie_title):

    year = extract_year(movie_title)

    # Remove year from the search query

    clean_title = normalize_title(movie_title)

    search_result = search_movie(
        clean_title,
        year
    )

    if not search_result:
        return None

    movie_id = search_result.get("id")

    if not movie_id:
        return None

    details = get_movie_details(movie_id)

    credits = get_movie_credits(movie_id)

    if not details:
        return None

    # Extract cast

    cast_names = []

    if credits:

        cast_names = [
            person.get("name")
            for person in credits.get("cast", [])[:5]
            if person.get("name")
        ]

    # Extract director

    director = "Not available"

    if credits:

        for person in credits.get("crew", []):

            if person.get("job") == "Director":

                director = person.get(
                    "name",
                    "Not available"
                )

                break

    # Extract collection/franchise

    collection = details.get(
        "belongs_to_collection"
    )

    collection_name = (
        collection.get("name")
        if collection
        else None
    )

    # Extract genres

    tmdb_genres = [
        genre.get("name")
        for genre in details.get("genres", [])
        if genre.get("name")
    ]

    return {

        "poster": get_poster_url(
            details.get("poster_path")
        ),

        "overview": (
            details.get("overview")
            or "Overview not available."
        ),

        "cast": cast_names,

        "director": director,

        "collection": collection_name,

        "tmdb_rating": details.get("vote_average"),

        "release_date": details.get("release_date"),

        "genres": tmdb_genres

    }


# =====================================================
# 6. DISPLAY MOVIE INFORMATION
# =====================================================

def display_movie_card(rank, row):

    with st.container(border=True):

        st.markdown(
            f"### {rank}. {row['title']}"
        )

        with st.spinner("Loading movie information from TMDB..."):

            metadata = fetch_tmdb_metadata(
                row["title"]
            )

        if metadata:

            col1, col2 = st.columns(
                [1, 3]
            )

            with col1:

                if metadata["poster"]:

                    st.image(
                        metadata["poster"],
                        width=170
                    )

                else:

                    st.caption("Poster unavailable")

            with col2:

                st.write(
                    f"**Release date:** "
                    f"{metadata['release_date'] or 'Not available'}"
                )

                st.write(
                    f"**Overview:** {metadata['overview']}"
                )

                st.write(
                    f"**Director:** {metadata['director']}"
                )

                st.write(
                    f"**Cast:** "
                    f"{', '.join(metadata['cast']) or 'Not available'}"
                )

                if metadata["collection"]:

                    st.write(
                        f"**Movie collection:** "
                        f"{metadata['collection']}"
                    )

                if metadata["tmdb_rating"] is not None:

                    st.write(
                        f"**TMDB rating:** "
                        f"{metadata['tmdb_rating']:.1f}/10"
                    )

                if metadata["genres"]:

                    st.write(
                        f"**TMDB genres:** "
                        f"{', '.join(metadata['genres'])}"
                    )

        else:

            st.info(
                "TMDB information is currently unavailable."
            )

        # MovieLens information

        st.write(
            f"**MovieLens genres:** "
            f"{row['genres'] or 'Not available'}"
        )

        st.write(
            f"**Content similarity:** "
            f"{row['similarity']:.1%}"
        )

        if pd.notna(row["average_rating"]):

            st.write(
                f"**MovieLens average rating:** "
                f"{row['average_rating']}/5 "
                f"({row['rating_count']:,} ratings)"
            )

        else:

            st.write(
                "**MovieLens average rating:** Not available"
            )


# =====================================================
# 7. STREAMLIT INTERFACE
# =====================================================

st.title("🎬 CineMatch")

st.subheader(
    "Discover your next favourite movie"
)

st.write(
    "Select a movie to discover three recommendations "
    "based on genres, tags and title resemblance."
)

st.caption(
    "Python · TF-IDF · Cosine Similarity · "
    "Content-Based Filtering · TMDB API"
)


try:

    movies, feature_matrix = load_data_and_model()

    st.success(
        f"Model ready: {len(movies):,} movies loaded."
    )

    movie_titles = sorted(
        movies["title"].unique().tolist()
    )

    selected_title = st.selectbox(
        "Search and select a movie",
        options=movie_titles,
        index=None,
        placeholder="Start typing a movie title..."
    )

    if st.button(
        "Recommend 3 Movies",
        type="primary"
    ):

        if selected_title is None:

            st.warning(
                "Please select a movie first."
            )

        else:

            st.markdown(
                "## Your Recommendations"
            )

            st.write(
                f"Movies similar to **{selected_title}**"
            )

            results = recommend_movies(
                selected_title,
                movies,
                feature_matrix
            )

            if results.empty:

                st.warning(
                    "No recommendations could be generated."
                )

            else:

                for rank, row in enumerate(
                    results.to_dict("records"),
                    start=1
                ):

                    display_movie_card(
                        rank,
                        row
                    )


except FileNotFoundError as error:

    st.error(
        f"Dataset file missing: {error}"
    )

except pd.errors.EmptyDataError:

    st.error(
        "One of the CSV files is empty or unreadable."
    )

except (ValueError, KeyError) as error:

    st.error(
        f"Dataset error: {error}"
    )

except Exception as error:

    st.error(
        f"Unexpected error: {error}"
    )

    st.caption(
        "Check the VS Code terminal for the full traceback."
    )


# =====================================================
# 8. TMDB ATTRIBUTION
# =====================================================

st.divider()

st.caption(
    "This product uses the TMDB API but is not endorsed "
    "or certified by TMDB."
)
