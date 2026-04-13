from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv
import math
import heapq

# Genres in the same family earn partial credit instead of zero.
# This prevents a blues fan from scoring identically to someone who wants EDM.
_GENRE_FAMILY: Dict[str, str] = {
    "pop": "pop", "indie pop": "pop",
    "lofi": "electronic", "ambient": "electronic", "synthwave": "electronic", "electronic": "electronic",
    "rock": "rock", "metal": "rock",
    "jazz": "jazz", "blues": "jazz",
    "folk": "folk", "country": "folk", "classical": "folk",
    "hip hop": "urban", "reggae": "urban",
}

# Moods in the same cluster earn partial credit.
_MOOD_FAMILY: Dict[str, str] = {
    "happy": "upbeat", "confident": "upbeat", "euphoric": "upbeat",
    "chill": "calm", "relaxed": "calm", "focused": "calm", "laid-back": "calm",
    "intense": "energetic", "aggressive": "energetic",
    "moody": "dark", "melancholy": "dark",
    "nostalgic": "reflective", "intimate": "reflective", "heartfelt": "reflective",
}

_PARTIAL_CREDIT = 0.4  # fraction of full points awarded for a same-family (non-exact) match

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
    P_valence = 0.75
    P_acoustic = 0.75

    # Wider default sigma so extreme-energy users (≤0.2 or ≥0.9) aren't
    # unfairly penalised when the dataset has few songs near their preference.
    sigma_energy = user_prefs.get("sigma_energy", 0.15)
    sigma_tempo = user_prefs.get("sigma_tempo", 10.0)
    sigma_dance = user_prefs.get("sigma_dance", 0.12)
    sigma_valence = user_prefs.get("sigma_valence", 0.15)
    sigma_acoustic = user_prefs.get("sigma_acoustic", 0.15)

    def gaussian(x, u, sigma):
        if sigma <= 0:
            return 1.0 if math.isclose(x, u, rel_tol=1e-9, abs_tol=1e-9) else 0.0
        return math.exp(-((x - u) ** 2) / (2 * sigma * sigma))

    def _soft_categorical(pref: str, song_val: str, family_map: Dict[str, str]) -> Tuple[float, str]:
        """Return (score_fraction, label) using exact → family → no-match tiers."""
        if not pref:
            return 0.0, ""
        if pref == song_val:
            return 1.0, "exact"
        if family_map.get(pref) and family_map.get(pref) == family_map.get(song_val):
            return _PARTIAL_CREDIT, "related"
        return 0.0, ""

    # --- categorical: genre & mood (now soft) ---
    g_frac, g_label = _soft_categorical(user_prefs.get("genre", ""), s_genre, _GENRE_FAMILY)
    genre_pts = P_genre * g_frac
    if g_label == "exact":
        reasons.append(f"genre match (+{genre_pts:.2f})")
    elif g_label == "related":
        reasons.append(f"related genre (+{genre_pts:.2f})")

    m_frac, m_label = _soft_categorical(user_prefs.get("mood", ""), s_mood, _MOOD_FAMILY)
    mood_pts = P_mood * m_frac
    if m_label == "exact":
        reasons.append(f"mood match (+{mood_pts:.2f})")
    elif m_label == "related":
        reasons.append(f"related mood (+{mood_pts:.2f})")

    # --- numeric: energy (always scored) ---
    energy_pref = max(0.0, min(1.0, float(user_prefs.get("energy", user_prefs.get("target_energy", 0.5)))))
    s_energy_sim = gaussian(s_energy, energy_pref, sigma_energy)
    energy_pts = P_energy * s_energy_sim
    reasons.append(f"energy match ({s_energy_sim:.2f} -> +{energy_pts:.2f})")

    # --- numeric: tempo (optional) ---
    tempo_pref = float(user_prefs.get("tempo_bpm", user_prefs.get("target_tempo", 0.0)))
    tempo_pts = 0.0
    if tempo_pref > 0:
        s_tempo_sim = gaussian(s_tempo, tempo_pref, sigma_tempo)
        tempo_pts = P_tempo * s_tempo_sim
        reasons.append(f"tempo match ({s_tempo_sim:.2f} -> +{tempo_pts:.2f})")

    # --- numeric: danceability (optional) ---
    dance_pref = float(user_prefs.get("danceability", 0.0))
    dance_pts = 0.0
    if dance_pref > 0:
        s_dance_sim = gaussian(s_dance, dance_pref, sigma_dance)
        dance_pts = P_dance * s_dance_sim
        reasons.append(f"dance match ({s_dance_sim:.2f} -> +{dance_pts:.2f})")

    # --- numeric: valence (optional) ---
    s_valence = float(_get("valence", 0.0))
    valence_pref = float(user_prefs.get("valence", 0.0))
    valence_pts = 0.0
    if valence_pref > 0:
        s_valence_sim = gaussian(s_valence, valence_pref, sigma_valence)
        valence_pts = P_valence * s_valence_sim
        reasons.append(f"valence match ({s_valence_sim:.2f} -> +{valence_pts:.2f})")

    # --- numeric: acousticness (optional) ---
    s_acoustic = float(_get("acousticness", 0.0))
    acoustic_pref = float(user_prefs.get("acousticness", 0.0))
    acoustic_pts = 0.0
    if acoustic_pref > 0:
        s_acoustic_sim = gaussian(s_acoustic, acoustic_pref, sigma_acoustic)
        acoustic_pts = P_acoustic * s_acoustic_sim
        reasons.append(f"acousticness match ({s_acoustic_sim:.2f} -> +{acoustic_pts:.2f})")

    raw_score = genre_pts + mood_pts + energy_pts + tempo_pts + dance_pts + valence_pts + acoustic_pts

    # Only count a dimension in max_possible when it was actually scored,
    # so omitting optional fields never caps the score below 1.0.
    max_possible = P_genre + P_mood + P_energy
    if tempo_pref > 0:
        max_possible += P_tempo
    if dance_pref > 0:
        max_possible += P_dance
    if valence_pref > 0:
        max_possible += P_valence
    if acoustic_pref > 0:
        max_possible += P_acoustic

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
