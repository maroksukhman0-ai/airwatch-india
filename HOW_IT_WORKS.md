# How AirWatch India Works — A Beginner's Guide

This document explains every file in the project in very simple English, how the machine learning models work, and answers 10 common interview questions about the project.

---

## Part 1: Every File Explained

### `app.py` — The Web Server

Think of `app.py` as a waiter in a restaurant. When your browser asks for information (e.g. "give me Delhi's AQI"), `app.py` receives that request, goes and fetches the data from the database, and sends back a reply (a JSON response).

It uses a Python library called **Flask**, which makes it very easy to create a web server. Every `@app.route(...)` line says "when the browser visits this address, run this function and return the result."

For example:
- `/api/cities` → returns a list of all 26 cities
- `/api/summary?city=Delhi` → returns average, min, max AQI for Delhi
- `/api/forecast?city=Delhi` → trains (or loads) an ML model and returns a 7-day forecast

---

### `src/cleaning.py` — The Data Janitor

Raw data from Kaggle is messy. This file cleans it up before we can use it. It does 6 things in order:

1. **Load the CSV** — reads `city_day.csv` into a pandas DataFrame (a table in Python memory)
2. **Fix dates** — converts text like `"2019-01-01"` into real date objects Python understands
3. **Handle missing values** — some sensor readings are blank. We fill pollutant blanks with the *median* value for that city. We drop rows where the AQI itself is missing (we can't make up a target value)
4. **Remove impossible values** — negative pollutant concentrations are physically impossible, so we replace them with 0. AQI above 999 is a data entry error, so we remove those rows
5. **Add AQI category** — we add a new column that says whether each AQI reading is "Good", "Moderate", "Severe", etc., based on India's official CPCB scale
6. **Save to SQLite** — writes the clean table to a database file (`data/airwatch.db`) so we can query it quickly with SQL

---

### `src/analysis.py` — The Statistician

This file answers five questions about the data using a mix of SQL queries and pandas calculations:

| Function | Question it answers |
|----------|-------------------|
| `avg_aqi_by_city()` | Which cities have the worst/best average air quality? |
| `most_polluted_cities()` | Which cities spend the most days in unhealthy air? |
| `yearly_trend()` | Is air quality getting better or worse year by year? |
| `monthly_seasonal_pattern()` | Which months/seasons are most polluted? |
| `pollutant_correlation()` | Which chemical pollutant most closely tracks the overall AQI? |

At the end it writes a `data/insights.md` file with 5 plain-English findings based on the real numbers.

---

### `src/models.py` — The Forecaster

This file predicts the next 7 days of AQI for any city. Here is what it does, step by step:

**Step 1 — Load and prepare data**
It loads the daily AQI for a city and fills any date gaps (if a day has no reading, it carries the previous day's value forward).

**Step 2 — Build features**
A "feature" is a number the model uses as a clue to make a prediction. We create:
- `lag_1` = yesterday's AQI (the single most useful clue)
- `lag_2`, `lag_3`, `lag_7` = AQI from 2, 3, and 7 days ago
- `rolling_7` = the average of the last 7 days (smooths out daily noise)
- `month` = what month it is (winter months have higher AQI)
- `dayofweek` = what day of the week it is

**Step 3 — Time-based train/test split**
We use the oldest 80% of days for training and the newest 20% for testing. We never shuffle the data — that would let the model "peek at the future", which would be cheating.

**Step 4 — Cross-validation**
On the training data only, we use `TimeSeriesSplit` which creates 5 folds. Each fold uses earlier data to train and later data to validate. This tells us how reliable our model is before we ever touch the test data.

**Step 5 — Compare 3 models**
- **Baseline**: "tomorrow's AQI = today's AQI" — the simplest possible approach, no maths at all
- **Linear Regression**: fits the best straight line through the data
- **Random Forest**: builds 200 decision trees and averages their predictions — more powerful but slower

**Step 6 — 7-day forecast with uncertainty**
The best model rolls forward day by day. For day 1 it uses real past data; for day 2 it uses the day-1 prediction as a lag feature, and so on. The further ahead we predict, the wider the error band grows (because uncertainty compounds).

**Step 7 — Save the model**
The trained model is saved to a `.pkl` file using Python's `pickle` module. Next time the dashboard loads, it just reads the file instead of re-training.

---

### `src/chatbot.py` — The Assistant

A very simple rule-based chatbot. It looks for keywords in the user's question:
- If the question contains "aqi" and "mean" → explains what AQI is
- If the question contains "pm2.5" → explains what PM2.5 particles are
- If the question contains "good" or "safe" → explains what a safe AQI level is
- Otherwise → returns a polite "I don't know" message

---

### `templates/index.html` — The Dashboard

A single HTML file (~63 KB) that contains everything the browser needs. It has three parts:

1. **CSS** (styles) — defines colours, layout, fonts, card sizes, skeleton loaders, and responsive breakpoints for mobile/tablet/desktop
2. **HTML** (structure) — the sidebar, topbar, and four page sections (Overview, Trends, Forecast, Compare)
3. **JavaScript** (behaviour) — fetches data from the Flask API routes and draws charts using Chart.js

The dashboard has a dark/light theme toggle that saves your preference to browser storage.

---

### `tests/` — The Quality Check

Four test files that automatically check the code works correctly:

| File | Tests | What it checks |
|------|-------|---------------|
| `test_cleaning.py` | 21 | Dates get fixed, missing values get filled, impossible values get removed, AQI categories are correct |
| `test_analysis.py` | 28 | SQL queries return the right shape, sorting, and value ranges |
| `test_models.py` | 31 | Feature engineering, train/test split (no leakage!), MAE/RMSE maths, forecast shape and bounds |
| `test_chatbot.py` | 14+4 | Known questions return sensible answers, empty input doesn't crash |

Run them all with: `python -m pytest tests/ -v`

---

## Part 2: The Models Explained Simply

### What is a Baseline model?
The baseline predicts "tomorrow will be the same as today." It sounds too simple to be useful — but it is surprisingly hard to beat for air quality data, because AQI changes slowly from day to day. Any ML model must beat this to prove it is actually learning something.

### What is Linear Regression?
Imagine drawing the best straight line through a scatter plot. Linear Regression finds the line (or flat surface, in multiple dimensions) that minimises the total prediction error. It is fast, easy to understand, and works well when the relationship between features and AQI is roughly linear.

### What is a Random Forest?
A Random Forest builds many decision trees (we use 200). Each tree is trained on a random subset of the data and a random subset of features. At prediction time, all 200 trees make a guess, and we take the average. This "wisdom of the crowd" approach usually beats a single tree. The Random Forest also gives us **feature importance** — how much each feature contributed to reducing prediction errors across all trees.

### Why time-based splitting?
If we randomly shuffled the data and split it, the model could see future data during training. That would make it look much more accurate than it really is in production. Time-based splitting simulates the real world: the model is only trained on what it would have known at the time.

### What is a prediction interval?
A single forecast number (e.g. "AQI tomorrow = 140") doesn't tell you how confident the model is. A prediction interval adds lower and upper bounds (e.g. "between 55 and 225 with 95% confidence"). We calculate ours as:

```
half_width = 1.96 × test_RMSE × √(step_number)
lower = forecast − half_width
upper = forecast + half_width
```

The `√(step_number)` makes the interval wider for days further in the future, because uncertainty compounds.

---

## Part 3: 10 Interview Questions and Answers

---

**Q1: Why do you use median instead of mean to fill missing values?**

**A:** Air quality data often has extreme spikes — one day of crop burning can push AQI to 900. The mean is pulled upward by those spikes and would give a misleadingly high estimate for a typical day. The median is the "middle value" and is not affected by outliers. For example, if a city's PM2.5 readings are [30, 35, 40, 42, 900], the mean is 209 but the median is 40. Filling with 40 is much more realistic.

---

**Q2: Why do you split train and test by time, not randomly?**

**A:** Time-series data has temporal order — the future depends on the past. If I randomly shuffle and split, a training row from March 2020 could sit next to a test row from January 2015. The model sees "future" patterns during training, which inflates its accuracy on the test set. This is called **data leakage**. Time-based splitting respects the order: the model only ever trains on data that came *before* the test period, exactly as it would work in production.

---

**Q3: What is TimeSeriesSplit and why use it?**

**A:** Regular k-fold cross-validation randomly shuffles folds, which causes data leakage for time-series. `TimeSeriesSplit` creates folds where the validation set always comes after the training set in time. With 5 splits, fold 1 trains on months 1–2 and validates on month 3; fold 2 trains on months 1–3 and validates on month 4; and so on. This gives a realistic estimate of how the model performs on unseen future data.

---

**Q4: Why does the baseline nearly tie with Random Forest on Delhi?**

**A:** Delhi's AQI is highly autocorrelated — it changes slowly and smoothly rather than jumping unpredictably. The lag_1 feature (yesterday's AQI) has 83.1% importance in the Random Forest, meaning the model mostly just says "it will be about what it was yesterday." The baseline already does that for free. To gain a significant edge, we would need to add weather features (wind speed, humidity, temperature) which are the main forces that cause sudden AQI changes.

---

**Q5: What is Pearson correlation and what does r = 0.73 mean for PM2.5?**

**A:** Pearson correlation (r) measures how closely two numbers move together. It ranges from −1 (perfect opposite movement) through 0 (no relationship) to +1 (perfect same movement). PM2.5 has r = 0.73 with AQI, which means when PM2.5 goes up, AQI goes up in a fairly consistent way — about 73% of the variation in AQI can be explained by PM2.5 alone. This makes PM2.5 the most useful single pollutant for predicting AQI.

---

**Q6: What is SQLite and why use it instead of a "real" database?**

**A:** SQLite is a database that lives entirely in a single file. There is no server process to install, configure, or manage. Python includes SQLite support in its standard library. For a project like this — a read-heavy dashboard with one user — SQLite is fast enough and much simpler than PostgreSQL or MySQL. The downside is that SQLite is not designed for many simultaneous writes, but that is not a concern here.

---

**Q7: How does the prediction interval get wider over time?**

**A:** We multiply by `√(step_number)`. On day 1, the multiplier is 1.0; on day 4, it is 2.0; on day 7, it is 2.65. This reflects real uncertainty — our forecast for tomorrow is based on real data, but our forecast for next week is based on our own predictions, which may already contain errors. Each step compounds the previous step's uncertainty. The √ function grows slowly, which keeps the interval from becoming unreasonably wide after just a few steps.

---

**Q8: What does "feature importance" mean in a Random Forest?**

**A:** When a decision tree makes a split (e.g. "if lag_1 > 200, go left"), it reduces the prediction error for the data in that node. We measure how much error each feature reduces, averaged across all 200 trees and all splits. The result is a percentage: lag_1 contributed 83.1% of all error reduction, rolling_7 contributed 3.9%, and so on. A feature with high importance is a strong predictor; one near 0% is barely used.

---

**Q9: Why do you add `AQI_Category` to the database instead of computing it in the query?**

**A:** Two reasons. First, it is faster — computing the category once during cleaning and storing it means every dashboard request can read it directly rather than running a CASE WHEN statement on every row. Second, it ensures consistency — the CPCB thresholds are defined in one place (`cleaning.py`) and applied once, so all routes see the same categories.

---

**Q10: What would you do to improve the model's accuracy?**

**A:** Several things:
1. **Add weather features** — wind speed, temperature, and humidity are the main physical drivers of AQI change. They are available from the OpenWeatherMap API.
2. **Add pollutant features** — PM2.5 and PM10 readings could be used as additional lag features since they strongly correlate with AQI.
3. **Use a sequence model** — LSTM (Long Short-Term Memory) neural networks are designed for time-series and can learn longer-range patterns that a Random Forest misses.
4. **City-specific hyperparameter tuning** — the number of trees and depth in Random Forest could be tuned separately for each city using grid search.
5. **Longer training history** — the dataset ends in July 2020. A real system would continuously add new daily readings to keep the model current.
