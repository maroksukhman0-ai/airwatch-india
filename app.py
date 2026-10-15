"""
app.py
------
AirWatch India — Flask web application.

Routes
------
  GET  /                              → dashboard page
  GET  /api/cities                   → list all cities
  GET  /api/summary?city=            → avg/min/max AQI + category breakdown
  GET  /api/trend?city=              → yearly AQI averages
  GET  /api/seasonal?city=           → monthly + seasonal AQI averages
  GET  /api/pollutants?city=         → avg of every pollutant + correlation with AQI
  GET  /api/daily?city=              → every day's AQI (for heatmap)
  GET  /api/forecast?city=           → 7-day forecast with lower/upper range
  GET  /api/feature-importance?city= → RF feature importances + plain-English note
  GET  /api/compare?cities=          → side-by-side avg AQI for comma-separated cities
  POST /api/chat                     → chatbot   {"question": "..."}

Run with:
  python app.py
Then open http://127.0.0.1:5000
"""

import os
import sys
import logging
import sqlite3

from flask import Flask, render_template, jsonify, request

# ── Make src/ importable ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from cleaning import run_cleaning_pipeline        # noqa: E402
from chatbot  import answer as chatbot_answer     # noqa: E402
import models as ml
from data_answers import answer as data_answer    # noqa: E402                               # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
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

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
import math
import numpy as np
from flask.json.provider import DefaultJSONProvider


def clean_nan(obj):
    """Turn NaN and infinity into None, so the browser can read the JSON."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_nan(v) for v in obj]
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


class SafeJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        return super().dumps(clean_nan(obj), **kwargs)


app.json = SafeJSONProvider(app)

DB_PATH = os.path.join("data", "airwatch.db")

# Pollutant column names exactly as stored in the database
POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]


# ══════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════

def _db() -> sqlite3.Connection:
    """Open a SQLite connection whose rows behave like dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _require_city(city: str):
    """
    Validate the city parameter.
    Returns (city_str, None) on success or (None, error_response) on failure.
    """
    if not city:
        return None, (
            jsonify({"error": "Missing required parameter: ?city=CityName  "
                              "Example: /api/summary?city=Delhi"}),
            400,
        )
    # Check the city actually exists in the DB
    try:
        conn = _db()
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM air_quality WHERE LOWER(City)=LOWER(?)",
            (city,),
        ).fetchone()
        conn.close()
        if row["n"] == 0:
            # Give the caller a helpful list of valid names
            conn2 = _db()
            valid = [
                r["City"]
                for r in conn2.execute(
                    "SELECT DISTINCT City FROM air_quality ORDER BY City"
                ).fetchall()
            ]
            conn2.close()
            return None, (
                jsonify({
                    "error": f"City '{city}' not found in the database.",
                    "valid_cities": valid,
                }),
                404,
            )
    except Exception as e:
        log.error(f"_require_city DB error: {e}", exc_info=True)
        return None, (jsonify({"error": "Database error checking city name."}), 500)
    return city, None


# ══════════════════════════════════════════════════════════════════
# Page
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main dashboard HTML page."""
    return render_template("index.html")


# ══════════════════════════════════════════════════════════════════
# GET /api/cities
# ══════════════════════════════════════════════════════════════════

@app.route("/api/cities")
def api_cities():
    """
    Return a sorted list of every city in the database.

    Response:
      {"cities": ["Ahmedabad", "Aizawl", ...]}
    """
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT DISTINCT City FROM air_quality ORDER BY City"
        ).fetchall()
        conn.close()
        return jsonify({"cities": [r["City"] for r in rows]})
    except Exception as e:
        log.error(f"/api/cities: {e}", exc_info=True)
        return jsonify({"error": "Could not load cities. Is the database ready?"}), 500


# ══════════════════════════════════════════════════════════════════
# GET /api/summary?city=
# ══════════════════════════════════════════════════════════════════

@app.route("/api/summary")
def api_summary():
    """
    City-level summary: avg/min/max AQI, total days, and a breakdown
    of how many days fell in each AQI category.

    Response:
      {
        "city": "Delhi",
        "avg_aqi": 259.5, "min_aqi": 29.0, "max_aqi": 716.0,
        "total_days": 1999,
        "category_counts": {"Good": 12, "Moderate": 310, ...}
      }
    """
    city, err = _require_city(request.args.get("city", "").strip())
    if err:
        return err

    try:
        conn = _db()

        # Overall stats — one SQL call
        stats = conn.execute(
            """SELECT
                 ROUND(AVG(AQI),1) AS avg_aqi,
                 ROUND(MIN(AQI),1) AS min_aqi,
                 ROUND(MAX(AQI),1) AS max_aqi,
                 COUNT(*)          AS total_days
               FROM air_quality
               WHERE LOWER(City)=LOWER(?)""",
            (city,),
        ).fetchone()

        # Category breakdown — one SQL call
        cats = conn.execute(
            """SELECT AQI_Category, COUNT(*) AS cnt
               FROM air_quality
               WHERE LOWER(City)=LOWER(?)
               GROUP BY AQI_Category
               ORDER BY cnt DESC""",
            (city,),
        ).fetchall()
        conn.close()

        category_counts = {r["AQI_Category"]: r["cnt"] for r in cats}

        return jsonify({
            "city":             city,
            "avg_aqi":          stats["avg_aqi"],
            "min_aqi":          stats["min_aqi"],
            "max_aqi":          stats["max_aqi"],
            "total_days":       stats["total_days"],
            "category_counts":  category_counts,
        })

    except Exception as e:
        log.error(f"/api/summary city={city}: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch summary data."}), 500


# ══════════════════════════════════════════════════════════════════
# GET /api/trend?city=
# ══════════════════════════════════════════════════════════════════

@app.route("/api/trend")
def api_trend():
    """
    Yearly average AQI for a city — useful for a line chart showing
    whether air quality is getting better or worse over time.

    Response:
      {
        "city": "Delhi",
        "trend": [{"year": "2015", "avg_aqi": 320.1, "days": 365}, ...]
      }
    """
    city, err = _require_city(request.args.get("city", "").strip())
    if err:
        return err

    try:
        conn = _db()
        rows = conn.execute(
            """SELECT
                 strftime('%Y', Date)   AS year,
                 ROUND(AVG(AQI), 1)    AS avg_aqi,
                 COUNT(*)              AS days
               FROM air_quality
               WHERE LOWER(City)=LOWER(?)
               GROUP BY year
               ORDER BY year""",
            (city,),
        ).fetchall()
        conn.close()

        return jsonify({"city": city, "trend": [dict(r) for r in rows]})

    except Exception as e:
        log.error(f"/api/trend city={city}: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch trend data."}), 500


# ══════════════════════════════════════════════════════════════════
# GET /api/seasonal?city=
# ══════════════════════════════════════════════════════════════════

@app.route("/api/seasonal")
def api_seasonal():
    """
    Monthly and seasonal AQI averages for a city.

    Seasons:
      Winter  = Oct–Feb
      Summer  = Mar–Jun
      Monsoon = Jul–Sep

    Response:
      {
        "city": "Delhi",
        "monthly":  [{"month": 1, "month_name": "Jan", "avg_aqi": 310.2}, ...],
        "seasonal": [{"season": "Winter", "avg_aqi": 320.1, "days": 950}, ...]
      }
    """
    city, err = _require_city(request.args.get("city", "").strip())
    if err:
        return err

    # Month-number → name mapping
    month_names = {
        "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
        "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
        "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
    }

    # Month → season mapping
    season_map = {
        "01": "Winter", "02": "Winter",
        "03": "Summer", "04": "Summer", "05": "Summer", "06": "Summer",
        "07": "Monsoon", "08": "Monsoon", "09": "Monsoon",
        "10": "Winter", "11": "Winter", "12": "Winter",
    }

    try:
        conn = _db()
        rows = conn.execute(
            """SELECT
                 strftime('%m', Date)  AS month_num,
                 ROUND(AVG(AQI), 1)   AS avg_aqi,
                 COUNT(*)             AS days
               FROM air_quality
               WHERE LOWER(City)=LOWER(?)
               GROUP BY month_num
               ORDER BY month_num""",
            (city,),
        ).fetchall()
        conn.close()

        monthly = [
            {
                "month":      int(r["month_num"]),
                "month_name": month_names[r["month_num"]],
                "avg_aqi":    r["avg_aqi"],
                "days":       r["days"],
            }
            for r in rows
        ]

        # Aggregate monthly results into seasonal buckets in Python
        seasonal_acc: dict[str, list] = {"Winter": [], "Summer": [], "Monsoon": []}
        for r in rows:
            season = season_map[r["month_num"]]
            seasonal_acc[season].append((r["avg_aqi"], r["days"]))

        seasonal = []
        for season, pairs in seasonal_acc.items():
            if not pairs:
                continue
            total_days = sum(d for _, d in pairs)
            # Weighted average: each month's avg_aqi weighted by its day count
            w_avg = round(
                sum(a * d for a, d in pairs) / total_days, 1
            )
            seasonal.append({"season": season, "avg_aqi": w_avg, "days": total_days})
        seasonal.sort(key=lambda x: x["avg_aqi"], reverse=True)

        return jsonify({"city": city, "monthly": monthly, "seasonal": seasonal})

    except Exception as e:
        log.error(f"/api/seasonal city={city}: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch seasonal data."}), 500


# ══════════════════════════════════════════════════════════════════
# GET /api/pollutants?city=
# ══════════════════════════════════════════════════════════════════

@app.route("/api/pollutants")
def api_pollutants():
    """
    Average concentration of every pollutant for a city,
    plus each pollutant's Pearson correlation with AQI.

    Response:
      {
        "city": "Delhi",
        "pollutants": [
          {"name": "PM2.5", "avg": 95.3, "correlation_with_aqi": 0.72},
          ...
        ]
      }
    """
    city, err = _require_city(request.args.get("city", "").strip())
    if err:
        return err

    try:
        import pandas as pd
        import numpy as np

        conn = sqlite3.connect(DB_PATH)
        # Load pollutant + AQI columns; quote names to handle "PM2.5" dot
        quoted_cols = ", ".join(f'"{c}"' for c in POLLUTANT_COLS + ["AQI"])
        df = pd.read_sql(
            f'SELECT {quoted_cols} FROM air_quality WHERE LOWER(City)=LOWER(?)',
            conn,
            params=(city,),
        )
        conn.close()

        result = []
        for col in POLLUTANT_COLS:
            avg  = round(float(df[col].mean()), 3)
            corr = round(float(df[col].corr(df["AQI"])), 3)
            result.append({"name": col, "avg": avg, "correlation_with_aqi": corr})

        # Sort by correlation descending so the most important pollutant is first
        result.sort(key=lambda x: x["correlation_with_aqi"], reverse=True)

        return jsonify({"city": city, "pollutants": result})

    except Exception as e:
        log.error(f"/api/pollutants city={city}: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch pollutant data."}), 500


# ══════════════════════════════════════════════════════════════════
# GET /api/daily?city=
# ══════════════════════════════════════════════════════════════════

@app.route("/api/daily")
def api_daily():
    """
    Every day's AQI for a city — intended for a calendar heatmap.

    Optional filters:
      year=2019        → only return that year
      limit=365        → cap number of rows returned (default: all)

    Response:
      {
        "city": "Delhi",
        "data": [{"date": "2015-01-01", "aqi": 472.0, "category": "Severe"}, ...]
      }
    """
    city, err = _require_city(request.args.get("city", "").strip())
    if err:
        return err

    year  = request.args.get("year",  "").strip()
    limit = request.args.get("limit", "").strip()

    # Validate optional year parameter
    if year:
        if not year.isdigit() or not (2000 <= int(year) <= 2100):
            return jsonify({
                "error": f"Invalid year '{year}'. Use a 4-digit year like 2019."
            }), 400

    # Validate optional limit parameter
    if limit:
        if not limit.isdigit() or int(limit) < 1:
            return jsonify({
                "error": f"Invalid limit '{limit}'. Use a positive integer."
            }), 400
        limit_int = int(limit)
    else:
        limit_int = None

    try:
        conn = _db()
        sql = (
            "SELECT Date AS date, AQI AS aqi, AQI_Category AS category "
            "FROM air_quality WHERE LOWER(City)=LOWER(?)"
        )
        params: list = [city]

        if year:
            sql += " AND strftime('%Y', Date)=?"
            params.append(year)

        sql += " ORDER BY Date"

        if limit_int:
            sql += f" LIMIT {limit_int}"

        rows = conn.execute(sql, params).fetchall()
        conn.close()

        return jsonify({"city": city, "data": [dict(r) for r in rows]})

    except Exception as e:
        log.error(f"/api/daily city={city}: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch daily data."}), 500


# ══════════════════════════════════════════════════════════════════
# GET /api/forecast?city=
# ══════════════════════════════════════════════════════════════════

@app.route("/api/forecast")
def api_forecast():
    """
    7-day AQI forecast for a city using the best trained model.
    Each day includes a lower and upper 95% prediction bound.

    Optional: ?days=7  (1–30 days ahead, default 7)

    Response:
      {
        "city": "Delhi",
        "model_name": "Random Forest",
        "test_mae": 31.5, "test_rmse": 43.5,
        "forecast": [
          {"Date": "2020-07-02", "Day": 1,
           "Forecast_AQI": 140.2, "Lower_95": 54.9, "Upper_95": 225.5},
          ...
        ],
        "feature_importance_note": "..."
      }
    """
    city, err = _require_city(request.args.get("city", "").strip())
    if err:
        return err

    # Validate optional days parameter
    days_str = request.args.get("days", "7").strip()
    if not days_str.isdigit() or not (1 <= int(days_str) <= 30):
        return jsonify({
            "error": f"Invalid days value '{days_str}'. Use a number between 1 and 30."
        }), 400
    days = int(days_str)

    try:
        result = ml.forecast(city, days=days)

        # ml.forecast() returns {"error": "..."} on failure
        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify(result)

    except Exception as e:
        log.error(f"/api/forecast city={city}: {e}", exc_info=True)
        return jsonify({"error": "Forecast failed. Check logs/app.log for details."}), 500


# ══════════════════════════════════════════════════════════════════
# GET /api/feature-importance?city=
# ══════════════════════════════════════════════════════════════════

@app.route("/api/feature-importance")
def api_feature_importance():
    """
    Random Forest feature importances for a city's AQI forecast model.
    Trains the model if it hasn't been saved yet.

    Response:
      {
        "city": "Delhi",
        "features": [
          {"feature": "lag_1",     "importance": 0.831, "importance_pct": 83.1},
          {"feature": "rolling_7", "importance": 0.039, "importance_pct": 3.9},
          ...
        ],
        "plain_english": "The most useful clues for predicting tomorrow's AQI are: ..."
      }
    """
    city, err = _require_city(request.args.get("city", "").strip())
    if err:
        return err

    try:
        # Load saved model or train fresh
        saved = ml.load_model(city)
        if saved is None:
            log.info(f"/api/feature-importance: training model for {city}")
            result = ml.train_and_evaluate(city, verbose=False)
            model = result["model"]
            note  = result["importance_note"]
        else:
            model = saved["model"]
            note  = saved["importance_note"]

        # Extract importances from the model
        # If the best model was Linear Regression, fall back to RF
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        if isinstance(model, LinearRegression):
            # Re-run to get RF importances (RF always gets trained in pipeline)
            raw_series = ml.load_city_series(city)
            df = ml.build_features(raw_series)
            train, _ = ml.time_split(df)
            import numpy as np
            rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            rf.fit(train[ml.FEATURE_COLS].values, train["AQI"].values)
            importances = rf.feature_importances_
        else:
            importances = model.feature_importances_

        features = [
            {
                "feature":        name,
                "importance":     round(float(imp), 4),
                "importance_pct": round(float(imp) * 100, 1),
            }
            for name, imp in sorted(
                zip(ml.FEATURE_COLS, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

        return jsonify({
            "city":         city,
            "features":     features,
            "plain_english": note,
        })

    except Exception as e:
        log.error(f"/api/feature-importance city={city}: {e}", exc_info=True)
        return jsonify({"error": "Could not compute feature importance."}), 500


# ══════════════════════════════════════════════════════════════════
# GET /api/compare?cities=Delhi,Mumbai,Chennai
# ══════════════════════════════════════════════════════════════════

@app.route("/api/compare")
def api_compare():
    """
    Side-by-side comparison of multiple cities.

    Parameter: ?cities=Delhi,Mumbai,Chennai  (comma-separated, 2–10 cities)

    Response:
      {
        "cities": [
          {
            "city": "Delhi",
            "avg_aqi": 259.5, "min_aqi": 29.0, "max_aqi": 716.0,
            "total_days": 1999,
            "best_category": "Good", "worst_category": "Severe+",
            "bad_air_pct": 65.1
          },
          ...
        ]
      }
    """
    raw = request.args.get("cities", "").strip()
    if not raw:
        return jsonify({
            "error": "Missing required parameter: ?cities=City1,City2  "
                     "Example: /api/compare?cities=Delhi,Mumbai"
        }), 400

    # Split and clean the city list
    city_list = [c.strip() for c in raw.split(",") if c.strip()]

    if len(city_list) < 2:
        return jsonify({
            "error": "Please provide at least 2 city names separated by commas."
        }), 400

    if len(city_list) > 10:
        return jsonify({
            "error": "Too many cities. Maximum is 10 at once."
        }), 400

    results = []
    not_found = []

    try:
        conn = _db()

        for city in city_list:
            row = conn.execute(
                """SELECT
                     ROUND(AVG(AQI),1)  AS avg_aqi,
                     ROUND(MIN(AQI),1)  AS min_aqi,
                     ROUND(MAX(AQI),1)  AS max_aqi,
                     COUNT(*)           AS total_days,
                     ROUND(
                       100.0 * SUM(CASE WHEN AQI_Category IN
                         ('Poor','Very Poor','Severe','Severe+')
                         THEN 1 ELSE 0 END) / COUNT(*), 1
                     ) AS bad_air_pct
                   FROM air_quality
                   WHERE LOWER(City)=LOWER(?)""",
                (city,),
            ).fetchone()

            if row["total_days"] == 0:
                not_found.append(city)
                continue

            # Find most common best and worst category
            cats = conn.execute(
                """SELECT AQI_Category, COUNT(*) AS cnt
                   FROM air_quality
                   WHERE LOWER(City)=LOWER(?)
                   GROUP BY AQI_Category
                   ORDER BY MIN(AQI)""",
                (city,),
            ).fetchall()

            results.append({
                "city":         city,
                "avg_aqi":      row["avg_aqi"],
                "min_aqi":      row["min_aqi"],
                "max_aqi":      row["max_aqi"],
                "total_days":   row["total_days"],
                "bad_air_pct":  row["bad_air_pct"],
            })

        conn.close()

        response: dict = {"cities": results}
        if not_found:
            response["not_found"] = not_found
            response["warning"]   = (
                f"These cities were not found in the database: "
                f"{', '.join(not_found)}"
            )

        if not results:
            return jsonify({
                "error": "None of the requested cities were found.",
                "not_found": not_found,
            }), 404

        return jsonify(response)

    except Exception as e:
        log.error(f"/api/compare cities={raw}: {e}", exc_info=True)
        return jsonify({"error": "Could not fetch comparison data."}), 500


# ══════════════════════════════════════════════════════════════════
# POST /api/chat
# ══════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Simple chatbot endpoint.

    Request body (JSON):  {"question": "What does AQI mean?"}
    Response:             {"answer": "AQI stands for ..."}
    """
    try:
        body     = request.get_json(force=True, silent=True) or {}
        question = body.get("question", "").strip()
        if not question:
            return jsonify({
                "error": "Please send a JSON body with a 'question' field. "
                         'Example: {"question": "What is PM2.5?"}'
            }), 400

        reply = data_answer(question) or chatbot_answer(question)
        return jsonify({"answer": reply})

    except Exception as e:
        log.error(f"/api/chat: {e}", exc_info=True)
        return jsonify({"error": "Chatbot error. Please try again."}), 500


# ══════════════════════════════════════════════════════════════════
# Legacy route kept for backward compatibility with the HTML dashboard
# ══════════════════════════════════════════════════════════════════

@app.route("/api/aqi")
def api_aqi():
    """
    Alias for /api/daily — kept so the existing index.html chart still works.
    Returns {city, data: [{Date, AQI, AQI_Category}]}
    """
    city, err = _require_city(request.args.get("city", "").strip())
    if err:
        return err

    try:
        conn = _db()
        rows = conn.execute(
            "SELECT Date, AQI, AQI_Category FROM air_quality "
            "WHERE LOWER(City)=LOWER(?) ORDER BY Date",
            (city,),
        ).fetchall()
        conn.close()
        return jsonify({"city": city, "data": [dict(r) for r in rows]})

    except Exception as e:
        log.error(f"/api/aqi city={city}: {e}", exc_info=True)
        return jsonify({"error": "Database error."}), 500


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Auto-build the database if it doesn't exist yet
    if not os.path.exists(DB_PATH):
        log.info("Database not found — running cleaning pipeline first …")
        run_cleaning_pipeline()

    log.info("Starting AirWatch India on http://127.0.0.1:5000")
    # use_reloader=False prevents the server from starting twice in debug mode
    app.run(debug=False, use_reloader=False)
