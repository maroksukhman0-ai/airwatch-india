"""
tests/test_models.py
--------------------
Unit tests for src/models.py

We test the pure-Python helper functions (feature engineering, metrics,
forecast math) with synthetic data so they run fast without needing
the real DB.  Tests that require the DB are marked with @db_required.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd
import tempfile

import models as ml

DB_PATH = os.path.join('data', 'airwatch.db')
db_required = pytest.mark.skipif(
    not os.path.exists(DB_PATH),
    reason="data/airwatch.db not found — run src/cleaning.py first"
)


# ── Helper: build a synthetic daily AQI series ────────────────────────────────
def synthetic_series(n=200, start='2018-01-01', seed=42):
    """
    Create a fake city time-series DataFrame with [AQI] column.
    AQI values are random integers between 50 and 400.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n, freq='D')
    aqi   = rng.integers(50, 400, size=n).astype(float)
    df    = pd.DataFrame({'AQI': aqi}, index=dates)
    df.index.name = 'Date'
    return df


# ══════════════════════════════════════════════════════════════════
# build_features
# ══════════════════════════════════════════════════════════════════

class TestBuildFeatures:

    def test_returns_dataframe(self):
        df = synthetic_series()
        result = ml.build_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_all_feature_columns_present(self):
        """Every column in FEATURE_COLS must appear in the output."""
        df = synthetic_series()
        result = ml.build_features(df)
        for col in ml.FEATURE_COLS:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_aqi_column_kept(self):
        """The original AQI column must still be present (it is the target)."""
        df = synthetic_series()
        result = ml.build_features(df)
        assert 'AQI' in result.columns

    def test_first_rows_dropped(self):
        """
        The first 7 rows lack enough history to compute lag_7 and rolling_7.
        They should be dropped, so output should be shorter than input.
        """
        df = synthetic_series(n=100)
        result = ml.build_features(df)
        assert len(result) < len(df)
        assert len(result) == len(df) - 7  # exactly 7 rows dropped

    def test_no_nan_in_features(self):
        """After feature engineering there should be no NaN values."""
        df = synthetic_series(n=100)
        result = ml.build_features(df)
        assert result[ml.FEATURE_COLS].isna().sum().sum() == 0

    def test_lag1_equals_previous_aqi(self):
        """lag_1 for a row should equal the AQI of the previous row."""
        df = synthetic_series(n=50)
        result = ml.build_features(df)
        # Check a few random rows
        for i in [10, 20, 30]:
            row_date = result.index[i]
            prev_date = result.index[i - 1]
            assert result.loc[row_date, 'lag_1'] == df.loc[prev_date, 'AQI']

    def test_month_values_in_range(self):
        """Month must be between 1 and 12."""
        df = synthetic_series(n=400)   # span > 1 year
        result = ml.build_features(df)
        assert result['month'].between(1, 12).all()

    def test_dayofweek_values_in_range(self):
        """Day of week must be between 0 (Mon) and 6 (Sun)."""
        df = synthetic_series(n=100)
        result = ml.build_features(df)
        assert result['dayofweek'].between(0, 6).all()


# ══════════════════════════════════════════════════════════════════
# time_split
# ══════════════════════════════════════════════════════════════════

class TestTimeSplit:

    def test_returns_two_dataframes(self):
        df = ml.build_features(synthetic_series())
        train, test = ml.time_split(df)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test,  pd.DataFrame)

    def test_total_rows_preserved(self):
        """train + test rows should equal the original row count."""
        df = ml.build_features(synthetic_series())
        train, test = ml.time_split(df)
        assert len(train) + len(test) == len(df)

    def test_train_comes_before_test_in_time(self):
        """The last train date must be before the first test date."""
        df = ml.build_features(synthetic_series())
        train, test = ml.time_split(df)
        assert train.index[-1] < test.index[0]

    def test_approx_80_20_split(self):
        """Train should be about 80% and test about 20% of the data."""
        df = ml.build_features(synthetic_series(n=200))
        train, test = ml.time_split(df)
        ratio = len(train) / len(df)
        assert 0.75 <= ratio <= 0.85, f"Train ratio {ratio:.2f} is not ~0.80"

    def test_no_data_leakage(self):
        """No index from test should appear in train."""
        df = ml.build_features(synthetic_series())
        train, test = ml.time_split(df)
        overlap = set(train.index) & set(test.index)
        assert len(overlap) == 0, "Data leakage: train and test share dates"


# ══════════════════════════════════════════════════════════════════
# _mae_rmse
# ══════════════════════════════════════════════════════════════════

class TestMaeRmse:

    def test_perfect_prediction_gives_zero(self):
        """When predictions equal truth, MAE and RMSE should both be 0."""
        y = np.array([100.0, 200.0, 300.0])
        mae, rmse = ml._mae_rmse(y, y)
        assert mae  == 0.0
        assert rmse == 0.0

    def test_returns_two_floats(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        result = ml._mae_rmse(y_true, y_pred)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_mae_less_than_or_equal_rmse(self):
        """MAE ≤ RMSE is always true (RMSE penalises large errors more)."""
        rng = np.random.default_rng(0)
        y_true = rng.uniform(50, 400, 50)
        y_pred = y_true + rng.uniform(-30, 30, 50)
        mae, rmse = ml._mae_rmse(y_true, y_pred)
        assert mae <= rmse

    def test_known_values(self):
        """Test with a known example: errors of [10, 20] → MAE=15, RMSE≈15.81."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])  # errors 10, 20
        mae, rmse = ml._mae_rmse(y_true, y_pred)
        assert mae == 15.0
        assert abs(rmse - 15.81) < 0.01


# ══════════════════════════════════════════════════════════════════
# evaluate_models
# ══════════════════════════════════════════════════════════════════

class TestEvaluateModels:

    def _get_train_test(self, n=200):
        df = ml.build_features(synthetic_series(n=n))
        return ml.time_split(df)

    def test_returns_dict_with_three_models(self):
        train, test = self._get_train_test()
        results, _ = ml.evaluate_models(train, test)
        assert set(results.keys()) == {'baseline', 'linear', 'rf'}

    def test_all_models_have_mae_and_rmse(self):
        train, test = self._get_train_test()
        results, _ = ml.evaluate_models(train, test)
        for key, r in results.items():
            assert 'mae'  in r, f"{key} missing 'mae'"
            assert 'rmse' in r, f"{key} missing 'rmse'"

    def test_mae_values_are_non_negative(self):
        train, test = self._get_train_test()
        results, _ = ml.evaluate_models(train, test)
        for key, r in results.items():
            assert r['mae']  >= 0, f"{key} MAE is negative"
            assert r['rmse'] >= 0, f"{key} RMSE is negative"

    def test_y_test_matches_test_set_length(self):
        train, test = self._get_train_test()
        _, y_test = ml.evaluate_models(train, test)
        assert len(y_test) == len(test)

    def test_rf_model_is_fitted(self):
        """Random Forest model object should be a fitted sklearn estimator."""
        from sklearn.ensemble import RandomForestRegressor
        train, test = self._get_train_test()
        results, _ = ml.evaluate_models(train, test)
        rf = results['rf']['model']
        assert isinstance(rf, RandomForestRegressor)
        # A fitted RF has estimators_
        assert hasattr(rf, 'estimators_')


# ══════════════════════════════════════════════════════════════════
# make_forecast
# ══════════════════════════════════════════════════════════════════

class TestMakeForecast:

    def _get_model_and_last30(self):
        """Train a quick RF on synthetic data, return (model, last_30, test_rmse)."""
        df     = ml.build_features(synthetic_series(n=200))
        train, test = ml.time_split(df)
        results, _  = ml.evaluate_models(train, test)
        model = results['rf']['model']
        rmse  = results['rf']['rmse']
        # last 30 rows of the raw series (no feature cols needed here)
        raw   = synthetic_series(n=200)
        return model, raw.iloc[-30:], rmse

    def test_returns_dataframe(self):
        model, last30, rmse = self._get_model_and_last30()
        result = ml.make_forecast(model, last30, rmse, days=7)
        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_forecast_days(self):
        model, last30, rmse = self._get_model_and_last30()
        for days in [1, 7, 14]:
            result = ml.make_forecast(model, last30, rmse, days=days)
            assert len(result) == days, f"Expected {days} rows, got {len(result)}"

    def test_required_columns_present(self):
        model, last30, rmse = self._get_model_and_last30()
        result = ml.make_forecast(model, last30, rmse)
        for col in ['Date', 'Day', 'Forecast_AQI', 'Lower_95', 'Upper_95']:
            assert col in result.columns, f"Missing column: {col}"

    def test_forecast_aqi_non_negative(self):
        """AQI cannot be negative."""
        model, last30, rmse = self._get_model_and_last30()
        result = ml.make_forecast(model, last30, rmse)
        assert (result['Forecast_AQI'] >= 0).all()

    def test_lower_leq_forecast_leq_upper(self):
        """Lower bound ≤ forecast ≤ upper bound for every day."""
        model, last30, rmse = self._get_model_and_last30()
        result = ml.make_forecast(model, last30, rmse)
        assert (result['Lower_95'] <= result['Forecast_AQI']).all()
        assert (result['Forecast_AQI'] <= result['Upper_95']).all()

    def test_interval_widens_over_time(self):
        """
        The width of the prediction interval (upper - lower) should grow
        with each step because uncertainty compounds.
        """
        model, last30, rmse = self._get_model_and_last30()
        result = ml.make_forecast(model, last30, rmse, days=7)
        widths = (result['Upper_95'] - result['Lower_95']).tolist()
        # At least the first and last widths should show growth
        assert widths[-1] > widths[0], "Interval should widen over forecast horizon"

    def test_days_are_sequential(self):
        """Day numbers should be 1, 2, 3, …, N."""
        model, last30, rmse = self._get_model_and_last30()
        result = ml.make_forecast(model, last30, rmse, days=7)
        assert result['Day'].tolist() == list(range(1, 8))

    def test_dates_are_sequential(self):
        """Each forecast date should be exactly one day after the previous."""
        model, last30, rmse = self._get_model_and_last30()
        result = ml.make_forecast(model, last30, rmse, days=7)
        dates = pd.to_datetime(result['Date'])
        diffs = dates.diff().dropna()
        assert (diffs == pd.Timedelta(days=1)).all()


# ══════════════════════════════════════════════════════════════════
# save_model / load_model
# ══════════════════════════════════════════════════════════════════

class TestSaveLoadModel:

    def test_save_and_load_roundtrip(self):
        """A saved model should load back with the same metadata."""
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=10, random_state=0)
        X  = np.array([[1,2,3,4,5,1,0],[2,3,4,5,6,2,1]])
        y  = np.array([100.0, 200.0])
        rf.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp:
            # Temporarily point MODELS_DIR to tmp
            original_dir = ml.MODELS_DIR
            ml.MODELS_DIR = tmp
            try:
                path = ml.save_model('TestCity', rf, 'Random Forest', 25.0, 35.0, 'test note')
                assert os.path.exists(path)
                payload = ml.load_model('TestCity')
            finally:
                ml.MODELS_DIR = original_dir

        assert payload is not None
        assert payload['city']       == 'TestCity'
        assert payload['model_name'] == 'Random Forest'
        assert payload['test_mae']   == 25.0
        assert payload['test_rmse']  == 35.0

    def test_load_nonexistent_model_returns_none(self):
        """Loading a model that was never saved should return None."""
        result = ml.load_model('CityThatDoesNotExist_XYZ')
        assert result is None


# ══════════════════════════════════════════════════════════════════
# load_city_series  (requires DB)
# ══════════════════════════════════════════════════════════════════

class TestLoadCitySeries:

    @db_required
    def test_returns_dataframe(self):
        result = ml.load_city_series('Delhi')
        assert isinstance(result, pd.DataFrame)

    @db_required
    def test_has_aqi_column(self):
        result = ml.load_city_series('Delhi')
        assert 'AQI' in result.columns

    @db_required
    def test_index_is_datetime(self):
        result = ml.load_city_series('Delhi')
        assert pd.api.types.is_datetime64_any_dtype(result.index)

    @db_required
    def test_no_nan_after_ffill(self):
        """Forward-fill should leave no NaN in the AQI column."""
        result = ml.load_city_series('Delhi')
        assert result['AQI'].isna().sum() == 0

    @db_required
    def test_sorted_by_date(self):
        result = ml.load_city_series('Delhi')
        assert result.index.is_monotonic_increasing
