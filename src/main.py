"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    # Starter example profile (add tempo/danceability if desired)
    user_prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "tempo_bpm": 120,
        "danceability": 0.75,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for idx, (song, score, explanation) in enumerate(recommendations, start=1):
        title = song.get("title", "Unknown")
        print(f"{idx}. {title}")
        print(f"   Score: {score:.3f}")
        if explanation:
            print("   Reasons:")
            for reason in explanation.split("; "):
                print(f"     - {reason}")
        print()


if __name__ == "__main__":
    main()
