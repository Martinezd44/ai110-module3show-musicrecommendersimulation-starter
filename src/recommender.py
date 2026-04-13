from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv
import math
import heapq

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Score each song against the provided `UserProfile` and return the
        top-k `Song` objects sorted by descending score.
        """
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
        }

        scored: List[Tuple[Song, float, List[str]]] = []
        for s in self.songs:
            score, reasons = score_song(user_prefs, s)
            scored.append((s, score, reasons))

        scored.sort(key=lambda t: t[1], reverse=True)
        return [t[0] for t in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a short, human-readable explanation for why a song was
        recommended to the user.
        """
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
        }
        score, reasons = score_song(user_prefs, song)
        return f"Score {score:.3f} — {'; '.join(reasons)}"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    rows: List[Dict] = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                rows.append({
                    "id": int(r.get("id", 0)),
                    "title": r.get("title", ""),
                    "artist": r.get("artist", ""),
                    "genre": r.get("genre", ""),
                    "mood": r.get("mood", ""),
                    "energy": float(r.get("energy", 0.0)),
                    "tempo_bpm": float(r.get("tempo_bpm", 0.0)),
                    "valence": float(r.get("valence", 0.0)) if r.get("valence") is not None else 0.0,
                    "danceability": float(r.get("danceability", 0.0)),
                    "acousticness": float(r.get("acousticness", 0.0)),
                })
            except Exception:
                continue
    return rows

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
    reasons: List[str] = []

    def _get(field, default=None):
        # support both dict-like and dataclass-like song inputs
        if isinstance(song, dict):
            return song.get(field, default)
        return getattr(song, field, default)

    s_genre = _get("genre", "")
    s_mood = _get("mood", "")
    s_energy = float(_get("energy", 0.0))
    s_tempo = float(_get("tempo_bpm", 0.0))
    s_dance = float(_get("danceability", 0.0))

    # point budgets
    P_genre = 2.0
    P_mood = 1.0
    P_energy = 2.0
    P_tempo = 1.0
    P_dance = 1.0

    sigma_energy = user_prefs.get("sigma_energy", 0.10)
    sigma_tempo = user_prefs.get("sigma_tempo", 10.0)
    sigma_dance = user_prefs.get("sigma_dance", 0.12)

    # categorical
    genre_pts = P_genre if (user_prefs.get("genre") and user_prefs.get("genre") == s_genre) else 0.0
    if genre_pts > 0:
        reasons.append(f"genre match (+{P_genre:.1f})")
    mood_pts = P_mood if (user_prefs.get("mood") and user_prefs.get("mood") == s_mood) else 0.0
    if mood_pts > 0:
        reasons.append(f"mood match (+{P_mood:.1f})")

    def gaussian(x, u, sigma):
        if sigma <= 0:
            return 1.0 if x == u else 0.0
        return math.exp(-((x - u) ** 2) / (2 * sigma * sigma))

    energy_pref = float(user_prefs.get("energy", user_prefs.get("target_energy", 0.5)))
    s_energy_sim = gaussian(s_energy, energy_pref, sigma_energy)
    energy_pts = P_energy * s_energy_sim
    reasons.append(f"energy match ({s_energy_sim:.2f} → +{energy_pts:.2f})")

    tempo_pref = float(user_prefs.get("tempo_bpm", user_prefs.get("target_tempo", 0.0)))
    tempo_pts = 0.0
    if tempo_pref > 0:
        s_tempo_sim = gaussian(s_tempo, tempo_pref, sigma_tempo)
        tempo_pts = P_tempo * s_tempo_sim
        reasons.append(f"tempo match ({s_tempo_sim:.2f} → +{tempo_pts:.2f})")

    dance_pref = float(user_prefs.get("danceability", 0.0))
    dance_pts = 0.0
    if dance_pref > 0:
        s_dance_sim = gaussian(s_dance, dance_pref, sigma_dance)
        dance_pts = P_dance * s_dance_sim
        reasons.append(f"dance match ({s_dance_sim:.2f} → +{dance_pts:.2f})")

    raw_score = genre_pts + mood_pts + energy_pts + tempo_pts + dance_pts
    max_possible = P_genre + P_mood + P_energy + P_tempo + P_dance
    final_score = raw_score / (max_possible if max_possible > 0 else 1.0)

    return final_score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    # Build scored list and return top-k efficiently using a heap for large lists
    scored = []
    for s in songs:
        score, reasons = score_song(user_prefs, s)
        explanation = "; ".join(reasons)
        scored.append((s, score, explanation))

    # Use heapq.nlargest to avoid sorting entire list when k << n
    topk = heapq.nlargest(k, scored, key=lambda t: t[1])
    return topk
