"""
analysis.py
-----------
Five analyses on the cleaned AirWatch India dataset stored in data/airwatch.db.

Analyses:
  1. Average AQI by city                    ← SQL query
  2. Most polluted cities (top 10)          ← SQL query
  3. Yearly AQI trend                       ← SQL query
  4. Monthly and seasonal AQI pattern       ← pandas (date maths needed)
  5. Pollutant correlation with AQI         ← pandas (numeric correlation)

After computing everything the script:
  - Prints results to the terminal
  - Writes 5 plain-English insights to data/insights.md

Run with:  python src/analysis.py
"""

import os
import sys
import logging
import sqlite3
import json

import pandas as pd

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
# Force UTF-8 on the stdout stream handler so the arrow character
# does not crash on Windows terminals that default to cp1252.
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.stream = open(sys.stdout.fileno(), mode="w",
                               encoding="utf-8", buffering=1, closefd=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        _stdout_handler,
    ],
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DB_PATH       = os.path.join("data", "airwatch.db")
INSIGHTS_PATH = os.path.join("data", "insights.md")

# Pollutant columns we want to check against AQI
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx",
              "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

# Season mapping: month number → season name
# India has two clear extremes: Winter (Oct–Feb) and Summer (Mar–Jun);
# Monsoon (Jul–Sep) is kept separate because it affects air quality differently.
SEASON_MAP = {
    1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Winter", 11: "Winter", 12: "Winter",
}

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# ── Helper: open DB connection ─────────────────────────────────────────────────
def _connect() -> sqlite3.Connection:
    """Open the SQLite database and return a connection."""
    if not os.path.exists(DB_PATH):
        log.error(
            f"Database not found at {DB_PATH}. "
            "Run 'python src/cleaning.py' first to create it."
        )
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    return conn


# ── Helper: print a section header ────────────────────────────────────────────
def _header(title: str) -> None:
    # Use sys.stdout.buffer.write to safely print UTF-8 on Windows
    line = "=" * 60
    out = f"\n{line}\n  {title}\n{line}\n"
    sys.stdout.buffer.write(out.encode("utf-8"))
    sys.stdout.buffer.flush()


# ══════════════════════════════════════════════════════════════════
# Analysis 1 — Average AQI by city  (SQL)
# ══════════════════════════════════════════════════════════════════
def avg_aqi_by_city() -> pd.DataFrame:
    """
    Use a SQL GROUP BY to calculate the average AQI for every city.
    SQL is perfect here because the grouping is simple and the DB does it fast.
    Returns a DataFrame sorted from worst (highest) to best (lowest) AQI.
    """
    sql = """
        SELECT
            City,
            ROUND(AVG(AQI), 1)  AS Avg_AQI,
            ROUND(MIN(AQI), 1)  AS Min_AQI,
            ROUND(MAX(AQI), 1)  AS Max_AQI,
            COUNT(*)            AS Days
        FROM air_quality
        GROUP BY City
        ORDER BY Avg_AQI DESC
    """
    conn = _connect()
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()

    _header("1. Average AQI by City (worst -> best)")
    sys.stdout.buffer.write((df.to_string(index=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()
    log.info(f"avg_aqi_by_city: computed for {len(df)} cities")
    return df


# ══════════════════════════════════════════════════════════════════
# Analysis 2 — Most polluted cities, top 10  (SQL)
# ══════════════════════════════════════════════════════════════════
def most_polluted_cities(top_n: int = 10) -> pd.DataFrame:
    """
    Find the top N most polluted cities by average AQI.
    Also counts how many days each city spent in 'Poor', 'Very Poor',
    or 'Severe' — a practical measure of how often residents breathe bad air.
    Uses SQL because filtering and counting by category is clean in SQL.
    """
    sql = f"""
        SELECT
            City,
            ROUND(AVG(AQI), 1) AS Avg_AQI,
            SUM(CASE WHEN AQI_Category IN ('Poor','Very Poor','Severe','Severe+')
                     THEN 1 ELSE 0 END) AS Bad_Air_Days,
            COUNT(*) AS Total_Days,
            ROUND(
                100.0 * SUM(CASE WHEN AQI_Category IN ('Poor','Very Poor','Severe','Severe+')
                                 THEN 1 ELSE 0 END) / COUNT(*), 1
            ) AS Bad_Air_Pct
        FROM air_quality
        GROUP BY City
        ORDER BY Avg_AQI DESC
        LIMIT {top_n}
    """
    conn = _connect()
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()

    _header(f"2. Top {top_n} Most Polluted Cities")
    sys.stdout.buffer.write((df.to_string(index=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()
    log.info(f"most_polluted_cities: top {top_n} computed")
    return df


# ══════════════════════════════════════════════════════════════════
# Analysis 3 — Yearly AQI trend  (SQL)
# ══════════════════════════════════════════════════════════════════
def yearly_trend() -> pd.DataFrame:
    """
    Calculate the average AQI for each year across all cities.
    SQLite's strftime('%Y', Date) extracts the year from a date string.
    This tells us whether air quality is improving over time.
    """
    sql = """
        SELECT
            strftime('%Y', Date)   AS Year,
            ROUND(AVG(AQI), 1)     AS Avg_AQI,
            COUNT(*)               AS Days
        FROM air_quality
        GROUP BY Year
        ORDER BY Year
    """
    conn = _connect()
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()

    _header("3. Yearly AQI Trend (all cities combined)")
    sys.stdout.buffer.write((df.to_string(index=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()

    # Simple direction: compare first and last year
    if len(df) >= 2:
        first_aqi = df.iloc[0]["Avg_AQI"]
        last_aqi  = df.iloc[-1]["Avg_AQI"]
        change    = last_aqi - first_aqi
        direction = "improved" if change < 0 else "worsened"
        msg = (f"\n  -> AQI has {direction} by {abs(change):.1f} points "
               f"from {df.iloc[0]['Year']} to {df.iloc[-1]['Year']}\n")
        sys.stdout.buffer.write(msg.encode("utf-8"))
        sys.stdout.buffer.flush()

    log.info(f"yearly_trend: {len(df)} years of data")
    return df


# ══════════════════════════════════════════════════════════════════
# Analysis 4 — Monthly & seasonal pattern  (pandas)
# ══════════════════════════════════════════════════════════════════
def monthly_seasonal_pattern() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all data into pandas and use date maths to find:
      a) Average AQI for each month (1–12)
      b) Average AQI for each season (Winter / Summer / Monsoon)

    We use pandas here because we need to parse dates and map months
    to season names — something easier to do in Python than in SQL.
    """
    conn = _connect()
    try:
        # Load only the columns we need to keep memory small
        df = pd.read_sql("SELECT Date, AQI FROM air_quality", conn)
    finally:
        conn.close()

    # Parse dates and extract month number
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].map(SEASON_MAP)

    # ── Monthly averages ──────────────────────────────────────────
    monthly = (
        df.groupby("Month")["AQI"]
          .agg(Avg_AQI="mean", Days="count")
          .reset_index()
    )
    monthly["Avg_AQI"]    = monthly["Avg_AQI"].round(1)
    monthly["Month_Name"] = monthly["Month"].map(MONTH_NAMES)
    monthly = monthly[["Month", "Month_Name", "Avg_AQI", "Days"]]

    _header("4a. Monthly AQI Pattern (avg across all cities & years)")
    sys.stdout.buffer.write((monthly.to_string(index=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()

    worst_month  = monthly.loc[monthly["Avg_AQI"].idxmax(), "Month_Name"]
    best_month   = monthly.loc[monthly["Avg_AQI"].idxmin(), "Month_Name"]
    msg = f"\n  -> Worst month: {worst_month}  |  Best month: {best_month}\n"
    sys.stdout.buffer.write(msg.encode("utf-8"))
    sys.stdout.buffer.flush()

    # ── Seasonal averages ─────────────────────────────────────────
    seasonal = (
        df.groupby("Season")["AQI"]
          .agg(Avg_AQI="mean", Days="count")
          .reset_index()
          .sort_values("Avg_AQI", ascending=False)
    )
    seasonal["Avg_AQI"] = seasonal["Avg_AQI"].round(1)

    _header("4b. Seasonal AQI Pattern")
    sys.stdout.buffer.write((seasonal.to_string(index=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()

    log.info("monthly_seasonal_pattern: complete")
    return monthly, seasonal


# ══════════════════════════════════════════════════════════════════
# Analysis 5 — Pollutant correlation with AQI  (pandas)
# ══════════════════════════════════════════════════════════════════
def pollutant_correlation() -> pd.Series:
    """
    Calculate how strongly each pollutant is correlated with AQI.

    Correlation (Pearson r) is a number from -1 to +1:
      +1  = the pollutant rises and falls perfectly with AQI
       0  = no relationship
      -1  = pollutant rises as AQI falls (rare for pollutants)

    We use pandas .corr() because it handles all columns in one call
    and is cleaner than writing 12 separate SQL queries.
    """
    conn = _connect()
    try:
        # Wrap each column name in double-quotes so SQLite handles
        # names with dots (PM2.5) and other special characters correctly.
        cols = ", ".join(f'"{c}"' for c in POLLUTANTS + ["AQI"])
        df = pd.read_sql(f"SELECT {cols} FROM air_quality", conn)
    finally:
        conn.close()

    # corr_with_aqi: correlation of each pollutant column vs the AQI column
    corr_with_aqi = (
        df[POLLUTANTS].corrwith(df["AQI"])
          .sort_values(ascending=False)
          .round(3)
    )
    corr_with_aqi.name = "Correlation_with_AQI"

    _header("5. Pollutant Correlation with AQI (Pearson r, highest first)")
    sys.stdout.buffer.write((corr_with_aqi.to_frame().to_string() + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()

    top_pollutant = corr_with_aqi.idxmax()
    top_r         = corr_with_aqi.max()
    msg = f"\n  -> Most linked pollutant: {top_pollutant}  (r = {top_r})\n"
    sys.stdout.buffer.write(msg.encode("utf-8"))
    sys.stdout.buffer.flush()

    log.info(f"pollutant_correlation: top = {top_pollutant} (r={top_r})")
    return corr_with_aqi


# ══════════════════════════════════════════════════════════════════
# Write insights.md
# ══════════════════════════════════════════════════════════════════
def write_insights(
    city_df:    pd.DataFrame,
    top10_df:   pd.DataFrame,
    yearly_df:  pd.DataFrame,
    monthly_df: pd.DataFrame,
    seasonal_df:pd.DataFrame,
    corr:       pd.Series,
) -> None:
    """
    Use the real numbers from all five analyses to write 5 plain-English
    insights into data/insights.md.  Numbers are pulled directly from the
    DataFrames so the file always reflects the actual data.
    """

    # ── Pull numbers we need ──────────────────────────────────────
    worst_city      = city_df.iloc[0]["City"]
    worst_city_aqi  = city_df.iloc[0]["Avg_AQI"]
    best_city       = city_df.iloc[-1]["City"]
    best_city_aqi   = city_df.iloc[-1]["Avg_AQI"]

    bad_pct         = top10_df.iloc[0]["Bad_Air_Pct"]
    bad_city        = top10_df.iloc[0]["City"]

    first_year      = yearly_df.iloc[0]["Year"]
    first_aqi       = yearly_df.iloc[0]["Avg_AQI"]
    last_year       = yearly_df.iloc[-1]["Year"]
    last_aqi        = yearly_df.iloc[-1]["Avg_AQI"]
    trend_dir       = "improved" if last_aqi < first_aqi else "worsened"
    trend_delta     = abs(last_aqi - first_aqi)

    worst_month_row = monthly_df.loc[monthly_df["Avg_AQI"].idxmax()]
    best_month_row  = monthly_df.loc[monthly_df["Avg_AQI"].idxmin()]
    worst_season    = seasonal_df.iloc[0]["Season"]
    worst_s_aqi     = seasonal_df.iloc[0]["Avg_AQI"]
    best_season     = seasonal_df.iloc[-1]["Season"]
    best_s_aqi      = seasonal_df.iloc[-1]["Avg_AQI"]

    top_pollutant   = corr.idxmax()
    top_r           = corr.max()
    second_poll     = corr.index[1]
    second_r        = corr.iloc[1]

    # ── Build markdown ─────────────────────────────────────────────
    md = f"""# AirWatch India — 5 Key Insights
*Generated automatically from data/airwatch.db*

---

## Insight 1: City Inequality — Some Cities Breathe Twice as Bad

**{worst_city}** has the highest average AQI of **{worst_city_aqi}**,
which falls in the *{"Poor" if worst_city_aqi > 200 else "Moderate"}* category.
Compare that with **{best_city}** (avg AQI {best_city_aqi}), which consistently
has clean air. This shows that where you live in India makes an enormous
difference to the air you breathe every single day.

---

## Insight 2: The Most Polluted City Spends {bad_pct}% of Days in Unhealthy Air

**{bad_city}** spent **{bad_pct}%** of all days in the dataset at Poor, Very Poor,
Severe, or Severe+ AQI levels. That means residents of {bad_city} experience
health-damaging air for more than {int(bad_pct)}% of the year on average — 
not just occasional spikes, but a persistent and serious problem.

---

## Insight 3: Air Quality Has {trend_dir.title()} Over {int(last_year) - int(first_year)} Years

The national average AQI was **{first_aqi}** in {first_year} and
**{last_aqi}** in {last_year} — a change of {trend_delta:.1f} points.
{"This improvement is encouraging, though AQI levels remain high in many cities."
 if trend_dir == "improved"
 else "This worsening trend suggests pollution control measures need to be strengthened."}

---

## Insight 4: Winter Is the Most Polluted Season — Nearly {round(worst_s_aqi / best_s_aqi, 1)}× Worse Than {best_season}

Average AQI in **{worst_season}** is **{worst_s_aqi}** versus just **{best_s_aqi}**
in **{best_season}**. The worst individual month is **{worst_month_row["Month_Name"]}**
(avg AQI {worst_month_row["Avg_AQI"]}) and the best is **{best_month_row["Month_Name"]}**
(avg AQI {best_month_row["Avg_AQI"]}).
Cold, still winter air traps pollutants close to the ground; monsoon rains wash
them away, which is why {best_season} is cleaner.

---

## Insight 5: {top_pollutant} Is the Strongest Predictor of Bad AQI (r = {top_r})

Of all 12 pollutants measured, **{top_pollutant}** has the highest correlation
with AQI (Pearson r = **{top_r}**), followed closely by **{second_poll}**
(r = {second_r}). A correlation near 1.0 means these pollutants rise and fall
almost in lockstep with the overall AQI score. Reducing {top_pollutant} 
emissions — from vehicles, industry, and crop burning — would have the
biggest measurable impact on air quality.

---
*To regenerate this file run: `python src/analysis.py`*
"""

    os.makedirs(os.path.dirname(INSIGHTS_PATH), exist_ok=True)
    with open(INSIGHTS_PATH, "w", encoding="utf-8") as f:
        f.write(md)

    log.info(f"Insights written to {INSIGHTS_PATH}")
    sys.stdout.buffer.write(f"\n  [OK] Insights saved to {INSIGHTS_PATH}\n".encode("utf-8"))
    sys.stdout.buffer.flush()


# ══════════════════════════════════════════════════════════════════
# Public helper used by app.py / API routes
# ══════════════════════════════════════════════════════════════════
def load_clean_data() -> pd.DataFrame:
    """Load the full air_quality table into a DataFrame (used by Flask routes)."""
    conn = _connect()
    try:
        df = pd.read_sql("SELECT * FROM air_quality", conn)
        log.info(f"load_clean_data: {len(df):,} rows loaded")
        return df
    finally:
        conn.close()


def city_avg_aqi(city: str) -> float:
    """Return the average AQI for one city. Returns -1 if city not found."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT AVG(AQI) FROM air_quality WHERE LOWER(City) = LOWER(?)",
            (city,),
        ).fetchone()
        val = row[0]
        if val is None:
            log.warning(f"city_avg_aqi: no data for '{city}'")
            return -1
        return round(val, 2)
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════
# Entry point: run all analyses
# ══════════════════════════════════════════════════════════════════
def run_all() -> None:
    """Run every analysis in order, print results, and write insights.md."""
    log.info("Analysis pipeline START")

    try:
        city_df             = avg_aqi_by_city()
        top10_df            = most_polluted_cities(top_n=10)
        yearly_df           = yearly_trend()
        monthly_df, seas_df = monthly_seasonal_pattern()
        corr                = pollutant_correlation()

        write_insights(city_df, top10_df, yearly_df, monthly_df, seas_df, corr)

        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
        log.info("Analysis pipeline COMPLETE")

    except Exception as e:
        log.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_all()
