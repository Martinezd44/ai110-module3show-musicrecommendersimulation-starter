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

    # Define multiple taste profiles
    profiles = {
        "default": {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "tempo_bpm": 120,
            "danceability": 0.75,
        },
        "high_energy_pop": {
            "genre": "pop",
            "mood": "confident",
            "energy": 0.95,
            "tempo_bpm": 130,
            "danceability": 0.90,
        },
        "chill_lofi": {
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.35,
            "tempo_bpm": 75,
            "danceability": 0.50,
        },
        "relaxed_jazz": {
            "genre": "jazz",
            "mood": "relaxed",
            "energy": 0.40,
            "tempo_bpm": 90,
            "danceability": 0.55,
        },
        # Exercises soft genre/mood matching: blues fan who likes melancholy songs
        # will now get partial credit for jazz and dark-adjacent moods
        "blues_adjacent": {
            "genre": "blues",
            "mood": "melancholy",
            "energy": 0.45,
            "tempo_bpm": 85,
        },
        # Exercises extreme-low-energy path and the new valence/acousticness dimensions
        "acoustic_introvert": {
            "genre": "folk",
            "mood": "intimate",
            "energy": 0.15,
            "acousticness": 0.90,
            "valence": 0.65,
        },
        # Exercises extreme-high-energy path
        "max_intensity": {
            "genre": "metal",
            "mood": "aggressive",
            "energy": 0.98,
            "tempo_bpm": 165,
            "danceability": 0.50,
        },
    }

    # Run recommendations for each profile and print formatted output
    for name, user_prefs in profiles.items():
        print(f"\n=== Profile: {name} ===")
        recommendations = recommend_songs(user_prefs, songs, k=5)
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
