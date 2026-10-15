"""
models.py
---------
Forecasts the next 7 days of AQI for any city in the database.

What this file does, step by step:
  1. Load a city's daily AQI time-series from SQLite.
  2. Build features: lag-1, lag-2, lag-3, lag-7, 7-day rolling mean,
     month number, and day-of-week number.
  3. Split data by TIME (older rows = train, newer rows = test).
     We never split randomly for time-series — that would leak future info
     into training and give falsely optimistic results.
  4. Run TimeSeriesSplit cross-validation on the training set only.
  5. Compare three models:
       a) Baseline   — "tomorrow's AQI = today's AQI" (no maths needed)
       b) Linear Regression — fits a straight-line relationship
       c) Random Forest     — builds many decision trees and averages them
  6. Report MAE and RMSE for every model; show how much the best beats baseline.
  7. Build a 7-day forecast with a prediction range (lower / upper bound).
  8. Print Random Forest feature importances as one plain-English sentence.
  9. Save the best model to data/models/<city>_model.pkl.

Run with:  python src/models.py
  (Trains and evaluates on Delhi by default — change DEFAULT_CITY to try others)
"""

import os
import sys
import logging
import pickle
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")   # suppress sklearn minor warnings

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DB_PATH       = os.path.join("data", "airwatch.db")
MODELS_DIR    = os.path.join("data", "models")
DEFAULT_CITY  = "Delhi"        # used when running this file directly

# How many days ahead to forecast
FORECAST_DAYS = 7

# Fraction of data used for testing (the most recent rows)
TEST_FRAC = 0.20

# Minimum rows needed to train a model (need enough for lags + splits)
MIN_ROWS = 60

# Feature column names (must match what build_features() creates)
FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_7",
    "rolling_7", "month", "dayofweek",
]

# ── Helper: safe stdout write (avoids cp1252 issues on Windows) ────────────────
def _print(text: str) -> None:
    """Write text to stdout using UTF-8 so special chars work on Windows."""
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


# ── Step 1: Load city data ─────────────────────────────────────────────────────
def load_city_series(city: str) -> pd.DataFrame:
    """
    Pull daily AQI for one city from SQLite.
    Returns a DataFrame with columns [Date, AQI], sorted by date.
    Gaps in the date sequence are filled by forward-filling the previous
    day's AQI — a reasonable assumption for air quality data.
    """
    if not os.path.exists(DB_PATH):
        log.error(f"Database not found: {DB_PATH}. Run cleaning.py first.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT Date, AQI FROM air_quality "
            "WHERE LOWER(City) = LOWER(?) ORDER BY Date",
            conn,
            params=(city,),
        )
    finally:
        conn.close()

    if df.empty:
        log.error(f"No data found for city '{city}'. "
                  "Check the name with: python src/analysis.py")
        sys.exit(1)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Fill date gaps — if a day is missing, carry the previous value forward
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_range).ffill()
    df.index.name = "Date"

    log.info(f"Loaded {len(df)} days of AQI data for {city}")
    return df


# ── Step 2: Build features ─────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-series features from the AQI column.

    lag_N       = AQI N days ago      (captures recent trend)
    rolling_7   = average of last 7 days (smooths out daily noise)
    month       = month number 1-12    (captures seasonal pattern)
    dayofweek   = 0=Mon … 6=Sun        (captures weekly pattern)

    Rows with NaN (the first 7 rows that don't have enough history) are dropped.
    """
    d = df.copy()
    d["lag_1"]     = d["AQI"].shift(1)   # yesterday
    d["lag_2"]     = d["AQI"].shift(2)   # 2 days ago
    d["lag_3"]     = d["AQI"].shift(3)   # 3 days ago
    d["lag_7"]     = d["AQI"].shift(7)   # same day last week
    # rolling_7: mean of the 7 days *before* today (shift(1) avoids using today)
    d["rolling_7"] = d["AQI"].shift(1).rolling(7).mean()
    d["month"]     = d.index.month
    d["dayofweek"] = d.index.dayofweek

    d = d.dropna()   # drop the first 7 rows that lack enough history
    return d


# ── Step 3: Train / test split by time ────────────────────────────────────────
def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset so the OLDEST rows are training and the NEWEST are test.
    Never shuffle time-series data — that leaks future information into training.
    """
    split_idx = int(len(df) * (1 - TEST_FRAC))
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    log.info(f"Train: {len(train)} rows  |  Test: {len(test)} rows")
    return train, test


# ── Step 4: Metrics helpers ────────────────────────────────────────────────────
def _mae_rmse(y_true, y_pred) -> tuple[float, float]:
    """Calculate MAE and RMSE, both rounded to 2 decimal places."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return round(mae, 2), round(rmse, 2)


# ── Step 5: Cross-validation on training set ──────────────────────────────────
def cross_validate_model(model, X_train: np.ndarray, y_train: np.ndarray,
                          n_splits: int = 5) -> dict:
    """
    TimeSeriesSplit creates N train/validation folds where each fold's
    validation set is strictly AFTER its training set.
    We use this only on training data — the test set is never touched here.

    Returns the mean and std of MAE and RMSE across all folds.
    """
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []

    for train_idx, val_idx in tscv.split(X_train):
        Xtr, Xval = X_train[train_idx], X_train[val_idx]
        ytr, yval = y_train[train_idx], y_train[val_idx]
        model.fit(Xtr, ytr)
        preds = model.predict(Xval)
        mae, rmse = _mae_rmse(yval, preds)
        maes.append(mae)
        rmses.append(rmse)

    return {
        "cv_mae_mean":  round(float(np.mean(maes)),  2),
        "cv_mae_std":   round(float(np.std(maes)),   2),
        "cv_rmse_mean": round(float(np.mean(rmses)), 2),
        "cv_rmse_std":  round(float(np.std(rmses)),  2),
    }


# ── Step 6: Evaluate all three models ─────────────────────────────────────────
def evaluate_models(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Train and evaluate all three models.  Returns a dict with scores and
    the fitted LinearRegression and RandomForest objects.
    """
    X_train = train[FEATURE_COLS].values
    y_train = train["AQI"].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test["AQI"].values

    results = {}

    # ── (a) Baseline: predict tomorrow = today (use lag_1 feature) ─────────────
    # The simplest possible forecast: whatever AQI was yesterday, predict that.
    baseline_preds = test["lag_1"].values
    b_mae, b_rmse  = _mae_rmse(y_test, baseline_preds)
    results["baseline"] = {
        "name":   "Baseline (tomorrow = today)",
        "mae":    b_mae,
        "rmse":   b_rmse,
        "preds":  baseline_preds,
    }

    # ── (b) Linear Regression ───────────────────────────────────────────────────
    lr = LinearRegression()
    lr_cv = cross_validate_model(LinearRegression(), X_train, y_train)
    lr.fit(X_train, y_train)
    lr_preds     = lr.predict(X_test)
    lr_mae, lr_rmse = _mae_rmse(y_test, lr_preds)
    results["linear"] = {
        "name":   "Linear Regression",
        "mae":    lr_mae,
        "rmse":   lr_rmse,
        "preds":  lr_preds,
        "model":  lr,
        "cv":     lr_cv,
    }

    # ── (c) Random Forest ───────────────────────────────────────────────────────
    # n_estimators=200 trees; random_state=42 makes results reproducible.
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_cv = cross_validate_model(
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        X_train, y_train
    )
    rf.fit(X_train, y_train)
    rf_preds     = rf.predict(X_test)
    rf_mae, rf_rmse = _mae_rmse(y_test, rf_preds)
    results["rf"] = {
        "name":   "Random Forest",
        "mae":    rf_mae,
        "rmse":   rf_rmse,
        "preds":  rf_preds,
        "model":  rf,
        "cv":     rf_cv,
    }

    return results, y_test


# ── Step 7: Print comparison table ────────────────────────────────────────────
def print_results(results: dict, y_test: np.ndarray) -> str:
    """
    Print a clean comparison table and return the key ('linear' or 'rf')
    of the best model (lowest MAE on the test set).
    """
    _print("\n" + "=" * 64)
    _print("  MODEL COMPARISON  (test-set performance)")
    _print("=" * 64)
    _print(f"  {'Model':<35} {'MAE':>7} {'RMSE':>7}")
    _print("  " + "-" * 50)

    b_mae = results["baseline"]["mae"]
    best_key  = "baseline"
    best_mae  = b_mae

    for key, r in results.items():
        cv_line = ""
        if "cv" in r:
            cv_line = (f"  (CV MAE {r['cv']['cv_mae_mean']} "
                       f"+/- {r['cv']['cv_mae_std']})")
        _print(f"  {r['name']:<35} {r['mae']:>7.2f} {r['rmse']:>7.2f}{cv_line}")
        if r["mae"] < best_mae:
            best_mae = r["mae"]
            best_key = key

    _print("=" * 64)

    # How much does the best model beat the baseline?
    if best_key != "baseline":
        pct_better = round(100 * (b_mae - best_mae) / b_mae, 1)
        _print(f"\n  Best model : {results[best_key]['name']}")
        _print(f"  Beats baseline by : {b_mae - best_mae:.2f} MAE points "
               f"({pct_better}% improvement)")
    else:
        _print("\n  No model beat the baseline — city AQI may be very stable.")

    return best_key


# ── Step 8: Feature importance explanation ─────────────────────────────────────
def explain_feature_importance(rf_model: RandomForestRegressor) -> str:
    """
    Extract feature importances from the trained Random Forest and return
    one plain-English sentence describing which feature matters most.

    Feature importance = how much each feature reduces prediction error
    across all trees, expressed as a percentage.
    """
    importances = rf_model.feature_importances_
    pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)

    # Build a readable percentage list
    top3 = [f"{name} ({100*imp:.1f}%)" for name, imp in pairs[:3]]

    # One-line plain-English explanation
    explanation = (
        f"The most useful clues for predicting tomorrow's AQI are: "
        f"{top3[0]}, then {top3[1]}, then {top3[2]}. "
        f"This means recent days' air quality is the strongest signal."
    )

    _print("\n  Random Forest — Feature Importance")
    _print("  " + "-" * 50)
    for name, imp in pairs:
        bar = "#" * int(imp * 40)   # simple ASCII bar chart
        _print(f"  {name:<12}  {100*imp:5.1f}%  {bar}")
    _print(f"\n  Plain English: {explanation}")

    return explanation


# ── Step 9: 7-day forecast with prediction interval ───────────────────────────
def make_forecast(model, last_known: pd.DataFrame,
                  test_rmse: float, days: int = FORECAST_DAYS) -> pd.DataFrame:
    """
    Produce a 'days'-step ahead forecast starting from the day after
    the last date in last_known.

    Prediction interval (95% range):
      lower = forecast - 1.96 * test_rmse
      upper = forecast + 1.96 * test_rmse
    1.96 is the z-score for a 95% confidence interval.
    The interval widens for further-out days because errors compound.

    The function rolls the forecast forward one day at a time:
    each new predicted value is fed back as the next row's lag features.
    """
    # Work on a copy so we don't mutate the original DataFrame
    buf = last_known[["AQI"]].copy()

    forecast_rows = []
    z95 = 1.96   # 95% confidence z-score

    for step in range(1, days + 1):
        # The date we are predicting
        next_date = buf.index[-1] + pd.Timedelta(days=1)

        # Build one row of features from the current buffer
        row = {
            "lag_1":     buf["AQI"].iloc[-1],
            "lag_2":     buf["AQI"].iloc[-2] if len(buf) >= 2 else buf["AQI"].iloc[-1],
            "lag_3":     buf["AQI"].iloc[-3] if len(buf) >= 3 else buf["AQI"].iloc[-1],
            "lag_7":     buf["AQI"].iloc[-7] if len(buf) >= 7 else buf["AQI"].iloc[-1],
            "rolling_7": buf["AQI"].iloc[-7:].mean(),
            "month":     next_date.month,
            "dayofweek": next_date.dayofweek,
        }
        X = np.array([[row[c] for c in FEATURE_COLS]])
        pred = float(model.predict(X)[0])
        pred = max(0.0, round(pred, 1))   # AQI cannot be negative

        # Widen the interval slightly for each step ahead
        # (uncertainty grows the further into the future we go)
        half_width = z95 * test_rmse * np.sqrt(step)
        lower = max(0.0, round(pred - half_width, 1))
        upper = round(pred + half_width, 1)

        forecast_rows.append({
            "Date":    next_date.strftime("%Y-%m-%d"),
            "Day":     step,
            "Forecast_AQI": pred,
            "Lower_95": lower,
            "Upper_95": upper,
        })

        # Append the predicted value to the buffer so the next step
        # can use it as a lag feature
        new_row = pd.DataFrame({"AQI": [pred]}, index=[next_date])
        buf = pd.concat([buf, new_row])

    return pd.DataFrame(forecast_rows)


# ── Step 10: Save best model ───────────────────────────────────────────────────
def save_model(city: str, model, model_name: str,
               test_mae: float, test_rmse: float,
               feature_importance_note: str) -> str:
    """
    Save the trained model plus metadata to a .pkl file.
    pickle serialises the Python object so it can be loaded later
    without re-training.

    Returns the path to the saved file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Sanitise city name for use as a filename (replace spaces with underscores)
    safe_name = city.lower().replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{safe_name}_model.pkl")

    payload = {
        "city":          city,
        "model_name":    model_name,
        "model":         model,
        "feature_cols":  FEATURE_COLS,
        "test_mae":      test_mae,
        "test_rmse":     test_rmse,
        "importance_note": feature_importance_note,
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)

    log.info(f"Model saved: {path}")
    return path


# ── Step 11: Load a saved model ────────────────────────────────────────────────
def load_model(city: str) -> dict | None:
    """
    Load a previously saved model for a city.
    Returns the payload dict, or None if no model has been saved yet.
    """
    safe_name = city.lower().replace(" ", "_")
    path = os.path.join(MODELS_DIR, f"{safe_name}_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════
# Public API — used by app.py Flask routes
# ══════════════════════════════════════════════════════════════════
def forecast(city: str, days: int = FORECAST_DAYS) -> dict:
    """
    Main entry point for the Flask API.
    Trains (or loads) a model for the city and returns a 7-day forecast.

    Returns a dict:
      {
        "city": "Delhi",
        "model_name": "Random Forest",
        "test_mae": 18.4,
        "test_rmse": 25.1,
        "forecast": [
          {"Date": "2021-01-01", "Day": 1, "Forecast_AQI": 210.3,
           "Lower_95": 161.1, "Upper_95": 259.5},
          ...
        ],
        "feature_importance_note": "..."
      }
    """
    try:
        saved = load_model(city)
        if saved:
            # Use the already-trained model — no need to retrain
            model     = saved["model"]
            test_rmse = saved["test_rmse"]
            test_mae  = saved["test_mae"]
            note      = saved["importance_note"]
            log.info(f"Loaded saved model for {city}")
        else:
            # Train from scratch
            result = train_and_evaluate(city, verbose=False)
            model     = result["model"]
            test_rmse = result["test_rmse"]
            test_mae  = result["test_mae"]
            note      = result["importance_note"]

        # Load the last 30 days of known data to seed the forecast
        raw = load_city_series(city)
        df  = build_features(raw)
        last_30 = raw.iloc[-30:]

        fc_df = make_forecast(model, last_30, test_rmse, days=days)

        return {
            "city":                   city,
            "model_name":             saved["model_name"] if saved else result["model_name"],
            "test_mae":               test_mae,
            "test_rmse":              test_rmse,
            "forecast":               fc_df.to_dict(orient="records"),
            "feature_importance_note": note,
        }

    except SystemExit:
        raise
    except Exception as e:
        log.error(f"forecast() failed for {city}: {e}", exc_info=True)
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════
# Full training pipeline (also called by forecast() when no saved model)
# ══════════════════════════════════════════════════════════════════
def train_and_evaluate(city: str, verbose: bool = True) -> dict:
    """
    Run the complete training and evaluation pipeline for one city.
    Prints results if verbose=True.
    Returns a dict with the best model, scores, and forecast.
    """
    if verbose:
        _print(f"\n{'='*64}")
        _print(f"  AirWatch India — Forecasting for: {city}")
        _print(f"{'='*64}")

    # 1. Load data
    raw = load_city_series(city)

    if len(raw) < MIN_ROWS:
        msg = (f"{city} has only {len(raw)} days of data "
               f"(minimum required: {MIN_ROWS}). Try a different city.")
        log.error(msg)
        raise ValueError(msg)

    # 2. Build features
    df = build_features(raw)

    # 3. Time split
    train, test = time_split(df)

    # 4. Evaluate all models
    results, y_test = evaluate_models(train, test)

    # 5. Print comparison table
    if verbose:
        best_key = print_results(results, y_test)
    else:
        # Find best key silently
        best_key = min(
            results, key=lambda k: results[k]["mae"]
        )

    best_result   = results[best_key]
    best_model    = best_result.get("model")   # None for baseline
    best_mae      = best_result["mae"]
    best_rmse     = best_result["rmse"]

    # If baseline won, fall back to RF for actual forecasting
    # (baseline can't roll forward — it needs a real model)
    if best_model is None:
        if verbose:
            _print("\n  Note: Baseline won, using Random Forest for forecasting.")
        best_model  = results["rf"]["model"]
        best_key    = "rf"
        best_mae    = results["rf"]["mae"]
        best_rmse   = results["rf"]["rmse"]

    # 6. Feature importance
    rf_model = results["rf"]["model"]
    note = explain_feature_importance(rf_model) if verbose else (
        "Feature importance: " + ", ".join(
            f"{n}={100*v:.1f}%"
            for n, v in sorted(
                zip(FEATURE_COLS, rf_model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )[:3]
        )
    )

    # 7. Make 7-day forecast
    last_30 = raw.iloc[-30:]
    fc_df   = make_forecast(best_model, last_30, best_rmse)

    if verbose:
        _print("\n" + "=" * 64)
        _print(f"  7-Day AQI Forecast for {city}  "
               f"(model: {results[best_key]['name']})")
        _print("=" * 64)
        _print(f"  {'Date':<12} {'Day':>4} {'Forecast':>10} "
               f"{'Lower 95%':>11} {'Upper 95%':>11}")
        _print("  " + "-" * 52)
        for _, row in fc_df.iterrows():
            _print(f"  {row['Date']:<12} {int(row['Day']):>4} "
                   f"{row['Forecast_AQI']:>10.1f} "
                   f"{row['Lower_95']:>11.1f} {row['Upper_95']:>11.1f}")

    # 8. Save best model (use the current best_key which may have been updated
    #    to 'rf' when baseline won but we still need a real model for rolling forecast)
    saved_path = save_model(
        city, best_model, results[best_key]["name"],
        best_mae, best_rmse, note,
    )
    if verbose:
        _print(f"\n  Model saved to: {saved_path}")

    return {
        "city":                   city,
        "model_name":             results[best_key]["name"],
        "model":                  best_model,
        "test_mae":               best_mae,
        "test_rmse":              best_rmse,
        "importance_note":        note,
        "forecast_df":            fc_df,
        "all_results":            results,
    }


# ── Entry point: run on DEFAULT_CITY ──────────────────────────────────────────
if __name__ == "__main__":
    # Allow passing a city name as a command-line argument:
    #   python src/models.py Mumbai
    city = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CITY
    try:
        train_and_evaluate(city, verbose=True)
    except Exception as e:
        log.error(f"models.py failed: {e}", exc_info=True)
        sys.exit(1)
