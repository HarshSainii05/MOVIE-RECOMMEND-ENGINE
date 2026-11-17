# CineMatch 🎬
### Intelligent Movie Recommendation Engine

**CineMatch** is a content-based movie recommendation engine built with Python and Machine Learning. It analyzes movie genres, tags, and title similarities to discover and recommend movies relevant to a user's selection.

The application combines machine learning techniques with The Movie Database (TMDB) API to deliver movie recommendations alongside rich movie information through an interactive web interface.

---

## Overview

Finding movies that match individual interests can be challenging when browsing large movie collections. CineMatch addresses this by analyzing movie characteristics and calculating similarity scores to identify relevant recommendations.

The recommendation engine uses TF-IDF vectorization and cosine similarity to compare movie features, with an additional title similarity component to refine its ranking.

## Key Features

- **Content-Based Recommendations:** Finds movies with similar characteristics to a selected movie.
- **TF-IDF Feature Extraction:** Converts movie genres and tags into numerical representations.
- **Cosine Similarity:** Measures similarity between movie feature vectors.
- **Title Similarity:** Uses string matching to incorporate title-based similarity.
- **Weighted Ranking:** Combines content and title similarity into a final recommendation score.
- **Top 3 Recommendations:** Displays the three highest-ranked results.
- **TMDB Integration:** Retrieves movie metadata through the TMDB API.
- **Movie Details:** Displays posters, overviews, release dates, cast, directors, ratings, and collection information.
- **Interactive Interface:** Provides a user-friendly application built with Streamlit.

## Technology Stack

| Technology | Application |
|---|---|
| Python | Core development |
| Pandas | Data processing |
| Scikit-learn | Feature extraction and similarity computation |
| TF-IDF | Text vectorization |
| Cosine Similarity | Movie similarity measurement |
| SciPy | Sparse matrix operations |
| Streamlit | Interactive web interface |
| TMDB API | Movie metadata and poster retrieval |
| Requests | HTTP communication |
| Python-dotenv | Environment variable management |

## Recommendation Methodology

CineMatch follows a content-based recommendation approach.

### 1. Data Processing

The application processes the MovieLens dataset, including:

- `movies.csv` – Movie titles and genres.
- `ratings.csv` – User ratings.
- `tags.csv` – Movie-related tags.

### 2. Feature Extraction

TF-IDF vectorization converts textual movie genres and tags into numerical feature vectors.

### 3. Feature Engineering

Genre and tag feature matrices are combined using SciPy's horizontal stacking (`hstack`) operation. Genre features receive an additional weighting factor.

### 4. Similarity Computation

Cosine similarity measures the similarity between movie feature vectors to identify movies with related characteristics.

### 5. Title-Based Matching

Python's `SequenceMatcher` calculates title similarity to provide an additional ranking signal.

### 6. Weighted Recommendation Ranking

The final recommendation score is calculated using:

\[
S = 0.80C + 0.20T
\]

Where:

- \(S\) = Final recommendation score
- \(C\) = Content similarity score
- \(T\) = Title similarity score

The engine ranks candidate movies and returns the top three recommendations.

### 7. Movie Metadata Enrichment

The TMDB API retrieves additional movie information for presentation. TMDB metadata does not currently influence the recommendation ranking.

---

## Application Workflow

```text
User Selects a Movie
        |
        v
MovieLens Dataset
        |
        v
Data Preprocessing
        |
        v
TF-IDF Feature Extraction
        |
        v
Genre + Tag Feature Combination
        |
        v
Cosine Similarity Calculation
        |
        v
Title Similarity Calculation
        |
        v
Weighted Recommendation Ranking
        |
        v
Top 3 Movie Recommendations
        |
        v
TMDB Metadata Integration
        |
        v
Streamlit Interface
```

---

## Project Structure

```text
CineMatch/
│
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
│
├── app.py
├── tmdb_api.py
├── test_tmdb.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- pip
- TMDB API Read Access Token

### Installation

**1. Clone the repository**

```bash
git clone YOUR_REPOSITORY_URL
cd CineMatch
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate the environment**

Windows:

```powershell
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Configure TMDB API**

Create a `.env` file in the project root:

```env
TMDB_BEARER_TOKEN=your_tmdb_read_access_token
```

Generate your API Read Access Token through [TMDB](https://www.themoviedb.org/).

**6. Configure the dataset**

Place the MovieLens CSV files inside the `data/` directory:

- movies.csv
- ratings.csv
- tags.csv

**7. Launch CineMatch**

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, typically:

`http://localhost:8501`

---

## Configuration & Security

API credentials are managed through environment variables using `python-dotenv`.

Never commit your `.env` file or expose API credentials publicly.

Recommended `.gitignore`:

```gitignore
.venv/
.env
__pycache__/
*.pyc
```

## Future Development

Potential improvements include:

- Plot-based semantic similarity.
- More accurate movie franchise detection.
- Hybrid recommendation algorithms.
- Personalized recommendations based on user preferences.
- Improved ranking and recommendation evaluation.
- Enhanced application interface and user experience.

## Author

**Harsh Saini**

Python Developer | Machine Learning | AI/ML


---

## Acknowledgements

- [MovieLens](https://grouplens.org/datasets/movielens/) – Movie recommendation research datasets.
- [The Movie Database (TMDB)](https://www.themoviedb.org/) – Movie metadata and image services.
- [Scikit-learn](https://scikit-learn.org/) – Machine learning tools.
- [Streamlit](https://streamlit.io/) – Application framework.

---

*Built with Python and Machine Learning.*
