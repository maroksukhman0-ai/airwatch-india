"""
data_answers.py
---------------
Answers chatbot questions using the AirWatch database.
Every answer shows how it was calculated.
Returns None when it cannot answer, so the old chatbot can try.
"""
import os
import sqlite3

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE, "data", "airwatch.db")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

WORST_WORDS = ["most polluted", "worst city", "worst air", "dirtiest",
               "highest aqi", "highest pollution", "most pollution"]
BEST_WORDS = ["cleanest", "least polluted", "best city", "best air",
              "lowest aqi", "lowest pollution", "least pollution"]


def _query(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _find_cities(q):
    """Return the city names found in the question, in the order written."""
    found = []
    for (city,) in _query("SELECT DISTINCT City FROM air_quality"):
        pos = q.find(city.lower())
        if pos >= 0:
            found.append((pos, city))
    found.sort()
    return [c for _, c in found]


def _rank(worst=True):
    order = "DESC" if worst else "ASC"
    rows = _query(
        f"""SELECT City, ROUND(AVG(AQI), 1), COUNT(AQI)
            FROM air_quality WHERE AQI IS NOT NULL
            GROUP BY City ORDER BY AVG(AQI) {order} LIMIT 3"""
    )
    title = "Most polluted cities" if worst else "Cleanest cities"
    lines = [f"{title} (by average AQI):"]
    for i, (city, avg, n) in enumerate(rows, 1):
        lines.append(f"{i}. {city}: {avg} (from {n} daily readings)")
    lines.append("How I calculated: average of all daily AQI readings "
                 "for each city in the dataset.")
    return "\n".join(lines)


def _average(city):
    avg, mn, mx, n = _query(
        """SELECT ROUND(AVG(AQI), 1), ROUND(MIN(AQI), 1),
                  ROUND(MAX(AQI), 1), COUNT(AQI)
           FROM air_quality WHERE LOWER(City)=LOWER(?) AND AQI IS NOT NULL""",
        (city,),
    )[0]
    if not n:
        return f"I have no AQI readings for {city}."
    return (f"The average AQI of {city} is {avg}.\n"
            f"Lowest day: {mn}. Highest day: {mx}.\n"
            f"How I calculated: average of {n} daily AQI readings for {city}.")


def _month(city, best):
    rows = _query(
        """SELECT strftime('%m', Date), ROUND(AVG(AQI), 1), COUNT(AQI)
           FROM air_quality WHERE LOWER(City)=LOWER(?) AND AQI IS NOT NULL
           GROUP BY 1""",
        (city,),
    )
    if not rows:
        return f"I have no AQI readings for {city}."
    pick = min(rows, key=lambda r: r[1]) if best else max(rows, key=lambda r: r[1])
    name = MONTHS[int(pick[0]) - 1]
    kind = "best (lowest AQI)" if best else "worst (highest AQI)"
    return (f"The {kind} month in {city} is {name}, with an average AQI of {pick[1]}.\n"
            f"How I calculated: average AQI of every {name} day in the data "
            f"({pick[2]} readings), compared across all 12 months.")


def _compare(c1, c2):
    out = ["Comparison (average AQI):"]
    avgs = []
    for c in (c1, c2):
        avg, n = _query(
            """SELECT ROUND(AVG(AQI), 1), COUNT(AQI) FROM air_quality
               WHERE LOWER(City)=LOWER(?) AND AQI IS NOT NULL""", (c,))[0]
        avgs.append((c, avg, n))
        out.append(f"- {c}: {avg} (from {n} daily readings)")
    (a, av, _), (b, bv, _) = avgs
    if av is None or bv is None:
        return "I could not compare these cities because data is missing."
    if av == bv:
        out.append("Both cities have the same average AQI.")
    else:
        hi, lo = (a, b) if av > bv else (b, a)
        diff = round(abs(av - bv), 1)
        out.append(f"{hi} has a higher average AQI than {lo}, by {diff} points.")
    out.append("How I calculated: average of all daily AQI readings per city.")
    return "\n".join(out)


def answer(question):
    """Return an answer from the data, or None if this is not a data question."""
    q = (question or "").lower().strip()
    try:
        # Health questions stay with the old chatbot
        if any(w in q for w in ["safe", "health", "child", "asthma", "mask"]):
            return None

        cities = _find_cities(q)

        if not cities:
            if any(k in q for k in WORST_WORDS):
                return _rank(worst=True)
            if any(k in q for k in BEST_WORDS):
                return _rank(worst=False)
            return None

        if "month" in q:
            best = any(w in q for w in ["best", "cleanest", "lowest", "least"])
            return _month(cities[0], best)

        if len(cities) >= 2 or "compare" in q or " vs " in q:
            if len(cities) >= 2:
                return _compare(cities[0], cities[1])
            return "Please name two cities to compare, for example: Compare Delhi and Mumbai."

        if any(w in q for w in ["average", "avg", "mean", "aqi", "pollution"]):
            return _average(cities[0])

        return None
    except Exception:
        return None