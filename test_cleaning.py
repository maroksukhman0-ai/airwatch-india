"""
tests/test_cleaning.py
----------------------
Unit tests for src/cleaning.py

We build small, synthetic DataFrames so these tests run without needing
the real city_day.csv file.  Each test checks one specific behaviour.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np
import sqlite3
import tempfile

from cleaning import (
    fix_dates,
    handle_missing_values,
    remove_impossible_values,
    add_aqi_category,
    save_to_sqlite,
)


# ── Helper: build a minimal DataFrame that looks like city_day.csv ────────────
def make_df(**kwargs):
    """Return a small test DataFrame with sensible defaults."""
    base = {
        'City':  ['Delhi', 'Delhi', 'Mumbai', 'Mumbai'],
        'Date':  ['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04'],
        'PM2.5': [80.0, 90.0, 50.0, 60.0],
        'PM10':  [120.0, 130.0, 80.0, 90.0],
        'NO':    [10.0, 12.0, 8.0, 9.0],
        'NO2':   [30.0, 35.0, 20.0, 25.0],
        'NOx':   [40.0, 47.0, 28.0, 34.0],
        'NH3':   [15.0, 18.0, 10.0, 12.0],
        'CO':    [1.5, 2.0, 0.8, 1.0],
        'SO2':   [20.0, 25.0, 15.0, 18.0],
        'O3':    [50.0, 55.0, 40.0, 45.0],
        'Benzene': [2.0, 2.5, 1.5, 1.8],
        'Toluene': [5.0, 6.0, 3.0, 4.0],
        'Xylene':  [1.0, 1.2, 0.8, 0.9],
        'AQI':   [180.0, 210.0, 120.0, 150.0],
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ══════════════════════════════════════════════════════════════════
# fix_dates
# ══════════════════════════════════════════════════════════════════

class TestFixDates:

    def test_valid_dates_converted_to_datetime(self):
        """Normal ISO date strings should become pandas Timestamp objects."""
        df = make_df()
        result = fix_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result['Date']), \
            "Date column should be datetime after fix_dates()"

    def test_unparseable_date_row_dropped(self):
        """A row with a bad date string should be removed."""
        df = make_df()
        df.loc[1, 'Date'] = 'not-a-date'
        result = fix_dates(df)
        assert len(result) == 3, "Bad-date row should be dropped"

    def test_no_nat_in_output(self):
        """No NaT values should remain after fix_dates()."""
        df = make_df()
        result = fix_dates(df)
        assert result['Date'].isna().sum() == 0, "No NaT should remain"

    def test_row_count_unchanged_when_all_valid(self):
        """Row count stays the same when all dates are valid."""
        df = make_df()
        result = fix_dates(df)
        assert len(result) == len(df)


# ══════════════════════════════════════════════════════════════════
# handle_missing_values
# ══════════════════════════════════════════════════════════════════

class TestHandleMissingValues:

    def test_missing_aqi_row_dropped(self):
        """A row where AQI is NaN must be removed."""
        df = make_df()
        df.loc[0, 'AQI'] = np.nan
        df = fix_dates(df)
        result = handle_missing_values(df)
        assert len(result) == 3, "Row with NaN AQI should be dropped"

    def test_missing_city_row_dropped(self):
        """A row where City is NaN must be removed."""
        df = make_df()
        df.loc[0, 'City'] = np.nan
        df = fix_dates(df)
        result = handle_missing_values(df)
        assert len(result) == 3

    def test_pollutant_nan_filled(self):
        """Missing pollutant values should be filled (no NaN left)."""
        df = make_df()
        df.loc[0, 'PM2.5'] = np.nan
        df = fix_dates(df)
        result = handle_missing_values(df)
        assert result['PM2.5'].isna().sum() == 0, "PM2.5 NaNs should be filled"

    def test_aqi_bucket_column_removed(self):
        """If an AQI_Bucket column exists it should be dropped (we add our own)."""
        df = make_df()
        df['AQI_Bucket'] = 'Moderate'
        df = fix_dates(df)
        result = handle_missing_values(df)
        assert 'AQI_Bucket' not in result.columns

    def test_all_rows_kept_when_no_missing(self):
        """No rows dropped when data is complete."""
        df = fix_dates(make_df())
        result = handle_missing_values(df)
        assert len(result) == len(df)


# ══════════════════════════════════════════════════════════════════
# remove_impossible_values
# ══════════════════════════════════════════════════════════════════

class TestRemoveImpossibleValues:

    def test_negative_pollutant_clamped_to_zero(self):
        """Negative PM2.5 is physically impossible; it should become 0."""
        df = make_df()
        df.loc[0, 'PM2.5'] = -10.0
        df = fix_dates(df)
        df = handle_missing_values(df)
        result = remove_impossible_values(df)
        assert result['PM2.5'].min() >= 0, "No negative pollutant values allowed"

    def test_aqi_above_999_removed(self):
        """AQI values above 999 are data-entry errors and should be dropped."""
        df = make_df()
        df.loc[0, 'AQI'] = 5000.0
        df = fix_dates(df)
        df = handle_missing_values(df)
        result = remove_impossible_values(df)
        assert result['AQI'].max() <= 999

    def test_aqi_below_zero_removed(self):
        """AQI cannot be negative."""
        df = make_df()
        df.loc[0, 'AQI'] = -5.0
        df = fix_dates(df)
        df = handle_missing_values(df)
        result = remove_impossible_values(df)
        assert result['AQI'].min() >= 0

    def test_valid_values_unchanged(self):
        """All-valid data should pass through without any rows removed."""
        df = fix_dates(make_df())
        df = handle_missing_values(df)
        before = len(df)
        result = remove_impossible_values(df)
        assert len(result) == before


# ══════════════════════════════════════════════════════════════════
# add_aqi_category
# ══════════════════════════════════════════════════════════════════

class TestAddAqiCategory:

    def test_column_created(self):
        """AQI_Category column should exist after add_aqi_category()."""
        df = fix_dates(make_df())
        df = handle_missing_values(df)
        df = remove_impossible_values(df)
        result = add_aqi_category(df)
        assert 'AQI_Category' in result.columns

    def test_good_category_for_low_aqi(self):
        """AQI = 30 should map to 'Good'."""
        df = make_df(AQI=[30.0, 30.0, 30.0, 30.0])
        df = fix_dates(df)
        df = handle_missing_values(df)
        df = remove_impossible_values(df)
        result = add_aqi_category(df)
        cats = result['AQI_Category'].astype(str).unique()
        assert 'Good' in cats, f"Expected Good, got {cats}"

    def test_severe_category_for_high_aqi(self):
        """AQI = 450 should map to 'Severe'."""
        df = make_df(AQI=[450.0, 450.0, 450.0, 450.0])
        df = fix_dates(df)
        df = handle_missing_values(df)
        df = remove_impossible_values(df)
        result = add_aqi_category(df)
        cats = result['AQI_Category'].astype(str).unique()
        assert 'Severe' in cats

    def test_all_rows_have_a_category(self):
        """Every row should have a non-null category."""
        df = fix_dates(make_df())
        df = handle_missing_values(df)
        df = remove_impossible_values(df)
        result = add_aqi_category(df)
        null_count = result['AQI_Category'].isna().sum()
        assert null_count == 0, f"{null_count} rows have no category"

    @pytest.mark.parametrize("aqi,expected", [
        (25,   'Good'),
        (75,   'Satisfactory'),
        (150,  'Moderate'),
        (250,  'Poor'),
        (350,  'Very Poor'),
        (450,  'Severe'),
        (600,  'Severe+'),
    ])
    def test_category_boundaries(self, aqi, expected):
        """Each CPCB boundary maps to the correct category label."""
        df = make_df(AQI=[float(aqi)] * 4)
        df = fix_dates(df)
        df = handle_missing_values(df)
        df = remove_impossible_values(df)
        result = add_aqi_category(df)
        cats = result['AQI_Category'].astype(str).unique()
        assert expected in cats, f"AQI {aqi} should be {expected}, got {cats}"


# ══════════════════════════════════════════════════════════════════
# save_to_sqlite
# ══════════════════════════════════════════════════════════════════

class TestSaveToSqlite:

    def _make_clean_df(self):
        df = fix_dates(make_df())
        df = handle_missing_values(df)
        df = remove_impossible_values(df)
        df = add_aqi_category(df)
        return df

    def test_table_created(self):
        """The air_quality table should exist in the DB after saving."""
        df = self._make_clean_df()
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, 'test.db')
            save_to_sqlite(df, db)
            conn = sqlite3.connect(db)
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            conn.close()
        assert ('air_quality',) in tables

    def test_row_count_matches(self):
        """Number of rows in DB should equal number of rows in DataFrame."""
        df = self._make_clean_df()
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, 'test.db')
            save_to_sqlite(df, db)
            conn = sqlite3.connect(db)
            n = conn.execute('SELECT COUNT(*) FROM air_quality').fetchone()[0]
            conn.close()
        assert n == len(df)

    def test_date_stored_as_string(self):
        """Dates in the DB should be strings in YYYY-MM-DD format."""
        df = self._make_clean_df()
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, 'test.db')
            save_to_sqlite(df, db)
            conn = sqlite3.connect(db)
            row = conn.execute('SELECT Date FROM air_quality LIMIT 1').fetchone()
            conn.close()
        # Should look like '2019-01-01'
        assert isinstance(row[0], str)
        assert len(row[0]) == 10
        assert row[0][4] == '-'
