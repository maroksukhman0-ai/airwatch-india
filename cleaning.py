"""
cleaning.py
-----------
Loads city_day.csv, cleans the data, and saves it to SQLite (airwatch.db).

Steps:
  1. Load the CSV file
  2. Fix date column
  3. Handle missing values in each column
  4. Remove impossible (out-of-range) values
  5. Add AQI category column
  6. Save clean data to SQLite database
"""

import os
import sys
import logging
import sqlite3
import warnings
import pandas as pd
import numpy as np

# Suppress numpy "Mean of empty slice" warning that appears when a city has
# zero readings for a rare pollutant (e.g. Xylene in some cities).
# This is expected — the code falls back to the global median for those cases.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

# ── Logging setup ──────────────────────────────────────────────────────────────
# All messages go to logs/app.log AND to the terminal so you can watch progress.
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),   # write to file
        logging.StreamHandler(sys.stdout),      # also print to terminal
    ],
)
log = logging.getLogger(__name__)

# ── File paths ─────────────────────────────────────────────────────────────────
# Change these if your files are in different locations.
CSV_PATH = os.path.join("data", "city_day.csv")
DB_PATH  = os.path.join("data", "airwatch.db")

# ── AQI category thresholds (India CPCB standard) ─────────────────────────────
# Source: Central Pollution Control Board, India
AQI_BINS   = [0, 50, 100, 200, 300, 400, 500, float("inf")]
AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe", "Severe+"]


def load_csv(path: str) -> pd.DataFrame:
    """
    Read the CSV file into a pandas DataFrame.
    If the file does not exist, stop immediately with a clear message.
    """
    if not os.path.exists(path):
        log.error(
            f"File not found: {path}\n"
            "Please download city_day.csv from Kaggle and place it in the data/ folder.\n"
            "  Dataset: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india"
        )
        sys.exit(1)   # stop the program cleanly

    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns from {path}")
    return df


def fix_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'Date' column from a string like '2015-01-01' to a real date object.
    Rows where the date cannot be parsed are dropped so we never have NaT dates
    mixed into time-series plots later.
    """
    before = len(df)
    # errors='coerce' turns unparseable dates into NaT (Not a Time) instead of crashing.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])   # drop rows where date parsing failed
    after = len(df)
    log.info(f"Date fix: kept {after:,} rows (dropped {before - after} bad dates)")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deal with missing (NaN) values column by column.
    The strategy is chosen based on the nature of each column.
    """

    # ── City & Date: these are identifiers — we must have them ────────────────
    # If either is missing we cannot know what the row represents, so drop it.
    before = len(df)
    df = df.dropna(subset=["City", "Date"])
    log.info(f"Dropped {before - len(df)} rows with missing City or Date")

    # ── Pollutant columns: fill missing with the city-level median ─────────────
    # Why median (not mean)?  Pollutant data often has extreme spikes (outliers).
    # The median is the "middle value" and is not pulled up by those spikes,
    # so it gives a more realistic typical value for a city.
    pollutant_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx",
                      "NH3", "CO", "SO2", "O3", "Benzene",
                      "Toluene", "Xylene"]

    for col in pollutant_cols:
        if col not in df.columns:
            log.warning(f"Column '{col}' not found in dataset — skipping")
            continue

        missing_count = df[col].isna().sum()
        if missing_count == 0:
            continue

        # Fill each city's missing values with that city's median for this pollutant.
        # If a city has NO data at all for this pollutant, fall back to the
        # overall median of all cities.
        df[col] = df.groupby("City")[col].transform(
            lambda x: x.fillna(x.median())
        )
        # After city-level fill, some cities may still have NaN if the whole
        # city group was NaN for this column. Fill those with the global median.
        global_median = df[col].median()
        df[col] = df[col].fillna(global_median)

        filled = missing_count - df[col].isna().sum()
        log.info(f"  {col}: filled {filled:,} missing values with city/global median")

    # ── AQI column: this is the target value ──────────────────────────────────
    # Strategy: drop rows where AQI is missing.
    # Why drop instead of fill?  AQI is the main thing we want to study/predict.
    # Making up a value for it would introduce fake data and skew our analysis.
    if "AQI" in df.columns:
        before = len(df)
        df = df.dropna(subset=["AQI"])
        log.info(f"Dropped {before - len(df)} rows with missing AQI")

    # ── AQI_Bucket: already a label column — we will recreate it ourselves ────
    # Drop the original if it exists; we add our own version below.
    if "AQI_Bucket" in df.columns:
        df = df.drop(columns=["AQI_Bucket"])

    log.info(f"After missing-value handling: {len(df):,} rows remain")
    return df


def remove_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where values are physically impossible.
    For example, a concentration of -5 µg/m³ cannot exist in the real world.
    We also cap AQI at a reasonable maximum (1000) to remove data-entry errors.
    """

    # Pollutant concentrations must be >= 0  (cannot be negative)
    pollutant_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx",
                      "NH3", "CO", "SO2", "O3", "Benzene",
                      "Toluene", "Xylene"]

    before = len(df)
    for col in pollutant_cols:
        if col in df.columns:
            # Replace negative values with NaN, then fill with 0
            # (a reading of 0 is at least physically possible)
            df[col] = df[col].where(df[col] >= 0, other=np.nan)
            df[col] = df[col].fillna(0)

    # AQI must be between 0 and 999 (CPCB scale stops at 500, but some
    # datasets use higher numbers for "off the charts" values; 999 is generous)
    if "AQI" in df.columns:
        df = df[(df["AQI"] >= 0) & (df["AQI"] <= 999)]

    after = len(df)
    log.info(f"Removed {before - after} rows with impossible values; {after:,} rows remain")
    return df


def add_aqi_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new column 'AQI_Category' based on India's CPCB AQI scale:
      0–50     → Good
      51–100   → Satisfactory
      101–200  → Moderate
      201–300  → Poor
      301–400  → Very Poor
      401–500  → Severe
      >500     → Severe+   (extreme events; uncommon but present in the data)
    """
    if "AQI" not in df.columns:
        log.warning("AQI column not found — skipping AQI_Category")
        return df

    # pd.cut assigns each AQI value to the correct bin label.
    # right=True means the right boundary is included: e.g., 100 → "Satisfactory"
    df["AQI_Category"] = pd.cut(
        df["AQI"],
        bins=AQI_BINS,
        labels=AQI_LABELS,
        right=True,
    )

    # Show how many rows fall in each category so you can sanity-check the result
    counts = df["AQI_Category"].value_counts().sort_index()
    log.info("AQI category distribution:\n" + counts.to_string())
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str) -> None:
    """
    Save the clean DataFrame to a SQLite database.
    SQLite is a simple file-based database — no server needed.
    The table is called 'air_quality'.
    If the table already exists, it is replaced with the new data.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Connect to (or create) the database file
    conn = sqlite3.connect(db_path)
    try:
        # Convert the Date column back to string so SQLite stores it cleanly
        df_to_save = df.copy()
        df_to_save["Date"] = df_to_save["Date"].dt.strftime("%Y-%m-%d")

        # Convert AQI_Category from pandas Categorical to plain string
        if "AQI_Category" in df_to_save.columns:
            df_to_save["AQI_Category"] = df_to_save["AQI_Category"].astype(str)

        df_to_save.to_sql("air_quality", conn, if_exists="replace", index=False)
        log.info(f"Saved {len(df_to_save):,} rows to {db_path}  (table: air_quality)")
    finally:
        conn.close()


def run_cleaning_pipeline() -> pd.DataFrame:
    """
    Master function: runs all cleaning steps in order and returns the clean DataFrame.
    Call this from app.py or from the command line.
    """
    log.info("=" * 60)
    log.info("AirWatch India — Data Cleaning Pipeline START")
    log.info("=" * 60)

    try:
        df = load_csv(CSV_PATH)
        df = fix_dates(df)
        df = handle_missing_values(df)
        df = remove_impossible_values(df)
        df = add_aqi_category(df)
        save_to_sqlite(df, DB_PATH)

        log.info("=" * 60)
        log.info(f"Pipeline COMPLETE — {len(df):,} clean rows ready")
        log.info("=" * 60)
        return df

    except Exception as e:
        # Catch any unexpected error, log it clearly, and stop
        log.error(f"Pipeline FAILED: {e}", exc_info=True)
        sys.exit(1)


# ── Run directly: python src/cleaning.py ──────────────────────────────────────
if __name__ == "__main__":
    run_cleaning_pipeline()
