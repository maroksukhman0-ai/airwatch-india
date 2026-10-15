"""
tests/test_analysis.py
----------------------
Unit tests for src/analysis.py

These tests call the real DB (data/airwatch.db) that was built by cleaning.py.
They skip gracefully if the DB is not present so CI can still run other tests.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd

DB_PATH = os.path.join('data', 'airwatch.db')
db_required = pytest.mark.skipif(
    not os.path.exists(DB_PATH),
    reason="data/airwatch.db not found — run src/cleaning.py first"
)

import analysis as an


# ══════════════════════════════════════════════════════════════════
# avg_aqi_by_city
# ══════════════════════════════════════════════════════════════════

class TestAvgAqiByCity:

    @db_required
    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        result = an.avg_aqi_by_city()
        assert isinstance(result, pd.DataFrame)

    @db_required
    def test_has_expected_columns(self):
        """DataFrame must have City, Avg_AQI, Min_AQI, Max_AQI, Days."""
        result = an.avg_aqi_by_city()
        for col in ['City', 'Avg_AQI', 'Min_AQI', 'Max_AQI', 'Days']:
            assert col in result.columns, f"Missing column: {col}"

    @db_required
    def test_sorted_worst_to_best(self):
        """First row should have a higher avg AQI than the last row."""
        result = an.avg_aqi_by_city()
        assert result.iloc[0]['Avg_AQI'] >= result.iloc[-1]['Avg_AQI']

    @db_required
    def test_all_avg_aqi_positive(self):
        """All average AQI values should be positive."""
        result = an.avg_aqi_by_city()
        assert (result['Avg_AQI'] > 0).all()

    @db_required
    def test_26_cities(self):
        """Should return 26 cities (matches the dataset)."""
        result = an.avg_aqi_by_city()
        assert len(result) == 26


# ══════════════════════════════════════════════════════════════════
# most_polluted_cities
# ══════════════════════════════════════════════════════════════════

class TestMostPollutedCities:

    @db_required
    def test_default_top_10(self):
        """Default call should return exactly 10 rows."""
        result = an.most_polluted_cities()
        assert len(result) == 10

    @db_required
    def test_custom_top_n(self):
        """top_n parameter should be respected."""
        result = an.most_polluted_cities(top_n=5)
        assert len(result) == 5

    @db_required
    def test_bad_air_pct_between_0_and_100(self):
        """Bad_Air_Pct must be in [0, 100]."""
        result = an.most_polluted_cities()
        assert result['Bad_Air_Pct'].between(0, 100).all()

    @db_required
    def test_has_required_columns(self):
        result = an.most_polluted_cities()
        for col in ['City', 'Avg_AQI', 'Bad_Air_Days', 'Total_Days', 'Bad_Air_Pct']:
            assert col in result.columns


# ══════════════════════════════════════════════════════════════════
# yearly_trend
# ══════════════════════════════════════════════════════════════════

class TestYearlyTrend:

    @db_required
    def test_returns_dataframe(self):
        result = an.yearly_trend()
        assert isinstance(result, pd.DataFrame)

    @db_required
    def test_has_year_and_aqi(self):
        result = an.yearly_trend()
        for col in ['Year', 'Avg_AQI', 'Days']:
            assert col in result.columns

    @db_required
    def test_years_are_sorted(self):
        """Years must be in ascending chronological order."""
        result = an.yearly_trend()
        years = result['Year'].tolist()
        assert years == sorted(years)

    @db_required
    def test_years_are_strings(self):
        """Year values should be strings like '2015'."""
        result = an.yearly_trend()
        assert all(isinstance(y, str) for y in result['Year'])

    @db_required
    def test_data_spans_at_least_3_years(self):
        """Dataset covers 2015–2020 so at least 3 years expected."""
        result = an.yearly_trend()
        assert len(result) >= 3


# ══════════════════════════════════════════════════════════════════
# monthly_seasonal_pattern
# ══════════════════════════════════════════════════════════════════

class TestMonthlySeasonalPattern:

    @db_required
    def test_returns_tuple_of_two_dataframes(self):
        monthly, seasonal = an.monthly_seasonal_pattern()
        assert isinstance(monthly, pd.DataFrame)
        assert isinstance(seasonal, pd.DataFrame)

    @db_required
    def test_12_months(self):
        """Should have one row per calendar month."""
        monthly, _ = an.monthly_seasonal_pattern()
        assert len(monthly) == 12

    @db_required
    def test_3_seasons(self):
        """Should have Winter, Summer, Monsoon."""
        _, seasonal = an.monthly_seasonal_pattern()
        assert set(seasonal['Season']) == {'Winter', 'Summer', 'Monsoon'}

    @db_required
    def test_monthly_has_month_name(self):
        monthly, _ = an.monthly_seasonal_pattern()
        assert 'Month_Name' in monthly.columns

    @db_required
    def test_seasonal_sorted_worst_first(self):
        """Seasonal DataFrame should be sorted highest-AQI first."""
        _, seasonal = an.monthly_seasonal_pattern()
        avqs = seasonal['Avg_AQI'].tolist()
        assert avqs == sorted(avqs, reverse=True)


# ══════════════════════════════════════════════════════════════════
# pollutant_correlation
# ══════════════════════════════════════════════════════════════════

class TestPollutantCorrelation:

    @db_required
    def test_returns_series(self):
        result = an.pollutant_correlation()
        import pandas as pd
        assert isinstance(result, pd.Series)

    @db_required
    def test_12_pollutants(self):
        """Should have one entry for each of the 12 pollutants."""
        result = an.pollutant_correlation()
        assert len(result) == 12

    @db_required
    def test_all_values_between_minus1_and_1(self):
        """Pearson r must be in [-1, 1]."""
        result = an.pollutant_correlation()
        assert result.between(-1, 1).all(), \
            f"Correlation out of range:\n{result[~result.between(-1,1)]}"

    @db_required
    def test_sorted_descending(self):
        """Series should be sorted from highest to lowest correlation."""
        result = an.pollutant_correlation()
        vals = result.tolist()
        assert vals == sorted(vals, reverse=True)

    @db_required
    def test_pm25_is_top_correlator(self):
        """PM2.5 should have the highest correlation with AQI for India data."""
        result = an.pollutant_correlation()
        assert result.index[0] == 'PM2.5', \
            f"Expected PM2.5 at top, got {result.index[0]}"


# ══════════════════════════════════════════════════════════════════
# city_avg_aqi  (the simple lookup used by Flask routes)
# ══════════════════════════════════════════════════════════════════

class TestCityAvgAqi:

    @db_required
    def test_known_city_returns_positive_float(self):
        result = an.city_avg_aqi('Delhi')
        assert isinstance(result, float)
        assert result > 0

    @db_required
    def test_unknown_city_returns_minus_one(self):
        result = an.city_avg_aqi('Atlantis')
        assert result == -1

    @db_required
    def test_case_insensitive(self):
        """'delhi' and 'Delhi' should give the same result."""
        lower = an.city_avg_aqi('delhi')
        upper = an.city_avg_aqi('Delhi')
        assert lower == upper
