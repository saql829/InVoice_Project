# utils/topics.py
import re

def guess_topic(text: str) -> str:
    text = (text or "").lower()
    if not text:
        return "other"
    if any(w in text for w in ["weather", "rain", "sunny", "forecast"]):
        return "weather"
    if any(w in text for w in ["food", "cook", "eat", "restaurant"]):
        return "food"
    if any(w in text for w in ["school", "study", "teacher", "class"]):
        return "education"
    if any(w in text for w in ["music", "song", "guitar", "band"]):
        return "music"
    if any(w in text for w in ["travel", "trip", "flight", "journey"]):
        return "travel"
    return "general"
