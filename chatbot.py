"""
chatbot.py
----------
Simple rule-based chatbot that answers questions about air quality.
(Stub — will be filled in a later task)
"""

import logging

log = logging.getLogger(__name__)


def answer(question: str) -> str:
    """
    Takes a user question and returns a text answer.
    Right now it only handles a few basic questions.
    """
    q = question.lower().strip()

    if "aqi" in q and "mean" in q:
        return (
            "AQI stands for Air Quality Index. "
            "It is a number from 0 to 500 that tells you how clean or polluted the air is. "
            "Higher = worse. Below 50 is Good; above 400 is Severe."
        )

    if "pm2.5" in q:
        return (
            "PM2.5 are tiny particles smaller than 2.5 micrometres. "
            "They can enter your lungs and cause health problems. "
            "India's safe limit is 60 µg/m³ (24-hour average)."
        )

    if "good" in q or "safe" in q:
        return "An AQI between 0 and 50 is considered Good — safe for everyone."

    # Default reply when the question is not recognised
    return (
        "Sorry, I don't know the answer to that yet. "
        "Try asking about AQI, PM2.5, or what is a safe AQI level."
    )
