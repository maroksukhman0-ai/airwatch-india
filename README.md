# 🌿 AirWatch India — Air Quality Analytics Dashboard

> **Real-world problem:** Air pollution is one of India's most pressing public health issues. Millions of people live in cities where the Air Quality Index (AQI) regularly exceeds safe limits — yet most citizens have no easy way to explore historical trends, understand seasonal patterns, or see a short-term forecast. AirWatch India turns raw government sensor data into an interactive, beginner-friendly analytics platform.

---

## 📊 Dataset

| Item | Details |
|------|---------|
| **Source** | [Air Quality Data in India — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) |
| **Author** | Rohan Rao (vopani) |
| **Licence** | CC0 1.0 Public Domain — free to use, share, and build on |
| **Coverage** | 26 Indian cities, January 2015 – July 2020 |
| **Raw rows** | 29,531 daily measurements |
| **Clean rows** | 24,757 (after dropping missing AQI, impossible values) |
| **Pollutants** | PM2.5, PM10, NO, NO₂, NOₓ, NH₃, CO, SO₂, O₃, Benzene, Toluene, Xylene |

**Licence note:** The dataset is released under CC0 (no rights reserved). You may use it freely for research, education, or commercial projects without attribution. This project is educational only and is not affiliated with or endorsed by CPCB or any government body.

---

## 🛠️ Tools & Libraries

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Language | Python | 3.13 | Everything |
| Web framework | Flask | 3.1.3 | REST API + serve HTML |
| Data | pandas | 2.3.3 | Loading, cleaning, analysis |
| ML | scikit-learn | 1.7.2 | Linear Regression, Random Forest, TimeSeriesSplit |
| Numerics | NumPy | 2.3.3 | Array maths, lag features |
| Database | SQLite (stdlib) | — | File-based DB, no server needed |
| Tests | pytest | 8.3.2 | 106 automated tests |
| Charts | Chart.js | 4.4.3 | Browser-side visualisations (CDN) |
| Font | Inter (Google Fonts) | — | Dashboard typography |

---

## ✨ Features

### Data Pipeline
- Loads `city_day.csv`, fixes dates, fills missing pollutant values with **city-level median** (falls back to global median), removes physically impossible readings, adds a 7-tier **AQI Category** column following India's CPCB standard.

### Analysis (5 insights)
1. Average AQI by city — ranked worst to best
2. Most polluted cities — days spent in unhealthy air
3. Yearly AQI trend — has air quality improved?
4. Monthly & seasonal pattern — winter vs monsoon
5. Pollutant correlation — which chemical most drives high AQI?

### ML Forecasting
- 7-day AQI forecast per city using lag features, rolling average, month, and day-of-week
- Three models compared: Baseline, Linear Regression, Random Forest
- Time-based 80/20 train/test split (no data leakage)
- TimeSeriesSplit cross-validation (5 folds)
- 95% prediction interval: forecast ± 1.96 × RMSE × √step
- Feature importance extracted from Random Forest, explained in plain English

### Dashboard (4 sections)
| Section | What you see |
|---------|-------------|
| Overview | Animated AQI gauge, 4 stat cards, AQI category doughnut chart |
| Trends | Daily history line chart with date-range filter, seasonal bar chart, pollutant correlation chart, calendar heatmap |
| Forecast | 7-day forecast line with shaded confidence band, day-by-day cards, model performance, feature importance bars |
| Compare | Pick 2–3 cities, bar chart + ranked table |

---

## 🚀 How to Run

### Prerequisites
- Python 3.10 or newer
- The Kaggle dataset file `city_day.csv` placed in the `data/` folder

### Step 1 — Get the dataset
1. Go to: <https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india>
2. Download `city_day.csv`
3. Place it at `data/city_day.csv`

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Build the database (first time only)
```bash
python src/cleaning.py
```
This creates `data/airwatch.db`. You only need to do this once.

### Step 4 — (Optional) Run analysis and train models
```bash
python src/analysis.py          # prints insights, writes data/insights.md
python src/models.py            # trains Delhi model, saves to data/models/
python src/models.py Mumbai     # train for any other city
```

### Step 5 — Start the web server
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

> **Shortcut:** `app.py` runs the cleaning pipeline automatically if `data/airwatch.db` does not exist yet, so you can skip Step 3.

### Step 6 — Run the tests
```bash
python -m pytest tests/ -v
```
Expected: **106 passed** in ~20 seconds.

---

## 📈 Model Results

All numbers are test-set performance on **Delhi** (1,999 days, 80/20 time split).

| Model | Test MAE | Test RMSE | CV MAE (mean ± std) |
|-------|----------|-----------|---------------------|
| **Baseline** (tomorrow = today) | **31.47** | 45.00 | — |
| Linear Regression | 31.86 | 43.66 | 40.02 ± 3.60 |
| **Random Forest** | **31.51** | **43.54** | 49.71 ± 10.22 |

**Best model:** Random Forest  
**Improvement over baseline:** 31.47 → 31.51 MAE (≈ tie on Delhi; Linear Regression wins on Mumbai with MAE 11.10 vs baseline 11.49, a 3.4% improvement)

**Why the baseline is so hard to beat:**  
Delhi's daily AQI is highly autocorrelated — today's value is the strongest single predictor of tomorrow's (lag_1 feature importance: **83.1%**). Adding weather data (wind speed, temperature, humidity) would likely give ML a larger edge.

**Feature importance (Random Forest — Delhi):**

| Feature | Importance |
|---------|-----------|
| lag_1 (yesterday's AQI) | 83.1% |
| rolling_7 (7-day mean) | 3.9% |
| lag_2 (2 days ago) | 3.5% |
| lag_3 (3 days ago) | 3.2% |
| lag_7 (same day last week) | 3.1% |
| month | 1.8% |
| day of week | 1.4% |

---

## ⚠️ Limitations

1. **Data ends July 2020.** The forecast shows dates after July 2020 but these are projections — the model was not trained on events after that date (COVID-19 lockdowns caused an unusual AQI dip in 2020 that may affect the forecast).

2. **No weather features.** Temperature, wind speed, and humidity strongly influence air quality but are not included in this dataset. Adding them would likely improve forecast accuracy significantly.

3. **City-average AQI.** Each row is already a city-day average. Neighbourhood-level variation is not captured.

4. **Simple lag features only.** The forecast model uses only AQI history, not pollutant readings, which limits its predictive power.

5. **Not real-time.** This is a historical dataset; the dashboard does not pull live sensor data.

6. **Educational project only.** Do not use this dashboard to make health or policy decisions. Always refer to official CPCB sources.

---

## 🖼️ Screenshots

_Add screenshots here after running the app. Suggested captures:_

| Screenshot | What to capture |
|-----------|----------------|
| `screenshots/overview_delhi.png` | Overview page — gauge showing Delhi AQI 259, stat cards, doughnut chart |
| `screenshots/trends_history.png` | Trends page — full daily AQI history line chart for Delhi |
| `screenshots/trends_heatmap.png` | Trends page — calendar heatmap showing seasonal colour pattern |
| `screenshots/forecast_delhi.png` | Forecast page — 7-day prediction with shaded confidence band |
| `screenshots/compare_cities.png` | Compare page — Delhi vs Mumbai vs Chennai bar chart and table |
| `screenshots/light_theme.png` | Any page in light mode |

---

## 📂 Project Structure

```
airwatch/
├── app.py                  ← Flask server — all API routes
├── requirements.txt        ← Python packages with exact versions
├── README.md               ← This file
├── HOW_IT_WORKS.md         ← Plain-English explanation of every file + interview Q&A
├── data/
│   ├── city_day.csv        ← ⚠️  You must add this (from Kaggle)
│   ├── airwatch.db         ← SQLite database (auto-created)
│   ├── models/             ← Saved ML model pickle files
│   └── insights.md         ← Auto-generated key insights
├── src/
│   ├── cleaning.py         ← Data cleaning pipeline
│   ├── analysis.py         ← 5 statistical analyses
│   ├── models.py           ← ML forecasting
│   └── chatbot.py          ← Simple rule-based Q&A
├── templates/
│   └── index.html          ← Full analytics dashboard (HTML/CSS/JS)
├── tests/
│   ├── test_cleaning.py    ← 21 tests for cleaning.py
│   ├── test_analysis.py    ← 28 tests for analysis.py
│   ├── test_models.py      ← 31 tests for models.py
│   └── test_chatbot.py     ← 14 tests for chatbot.py  (+ 4 parametrized)
└── logs/
    └── app.log             ← All server and pipeline log messages
```

---

## 🤝 Contributing

This is an educational project. Feel free to fork, experiment, and improve it. Suggested extensions:
- Add weather data (OpenWeatherMap API)
- Use LSTM or Prophet for time-series forecasting
- Add real-time AQI data from CPCB API
- Deploy on a cloud platform (Render, Railway, Fly.io)
