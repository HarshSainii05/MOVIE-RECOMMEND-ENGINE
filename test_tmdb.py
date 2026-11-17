
from tmdb_api import search_movie, get_movie_details

print("Testing TMDB connection...")

movie = search_movie("Spider-Man 2")

if movie:
    print("\nTMDB connection successful!")
    print("---------------------------")

    print("Movie:", movie.get("title"))
    print("TMDB ID:", movie.get("id"))
    print("Release Date:", movie.get("release_date"))

    movie_id = movie.get("id")
    details = get_movie_details(movie_id)

    if details:
        print("\nMovie Overview:")
        print(details.get("overview"))

        print("\nGenres:")
        for genre in details.get("genres", []):
            print("-", genre["name"])

        print("\nTMDB Rating:", details.get("vote_average"))

else:
    print("\nMovie not found or TMDB connection failed.")
