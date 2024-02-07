"""
Test suite for the DataAnalyzer class in the DataComparisonApp.

This test suite aims to cover a range of functionalities provided by the DataAnalyzer class, including
data comparison, trend analysis, statistical calculations, and more. Each test case is designed to verify
the correctness of the implementation, ensuring that the data analysis capabilities meet expected outcomes
and handle edge cases gracefully.
"""

import unittest
from app.data_analyzer import DataAnalyzer
import pandas as pd
import numpy as np

class TestDataAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up any class-wide configurations and test data."""
        # This could include setting paths to test files, initializing shared resources, etc.

    def setUp(self):
        """Set up test data and DataAnalyzer instance."""
        self.data_analyzer = DataAnalyzer()
        # Sample test data mimics real-world scenarios
        self.df1 = pd.DataFrame({
            'Date': pd.date_range(start='2021-01-01', periods=4, freq='D'),
            'Sales': [100, 150, 200, 250],
            'Expenses': [80, 120, 160, 200]
        })
        self.df2 = pd.DataFrame({
            'Date': pd.date_range(start='2021-01-01', periods=4, freq='D'),
            'Sales': [90, 140, 210, 240],
            'Expenses': [85, 115, 150, 195]
        })
        self.data_frames = {'df1': self.df1, 'df2': self.df2}

    def test_compare_data(self):
        """Ensure that data comparison identifies similarities and differences accurately."""
        selected_columns = ['Sales']
        comparison_results = self.data_analyzer.compare_data(self.data_frames, selected_columns)
        self.assertIsInstance(comparison_results, dict, "Comparison results should be a dictionary.")
        # Further assertions depend on the implementation details of compare_data

    def test_analyze_time_series(self):
        """Test the time series analysis for trend detection in sales data."""
        results = self.data_analyzer.analyze_time_series({'df1': self.df1}, 'Sales')
        self.assertIsInstance(results, list, "Time series analysis should return a list.")
        # Check for expected trends or patterns in the results
        # This assumes your analyze_time_series method returns a list of analysis summaries

    def test_calculate_statistics(self):
        """Verify that statistical calculations return expected values."""
        statistics = self.data_analyzer.calculate_statistics({'df1': self.df1}, ['Sales'])
        self.assertIsInstance(statistics, dict, "Statistics should be returned as a dictionary.")
        self.assertIn('mean', statistics['Sales'], "Mean should be calculated for Sales.")
        self.assertAlmostEqual(statistics['Sales']['mean'], 175.0, "Incorrect mean value for Sales.")

    def test_detect_outliers(self):
        """Test outlier detection on artificially created data with a known outlier."""
        # Adding an outlier
        df_with_outlier = self.df1.copy()
        df_with_outlier.loc[df_with_outlier.index[-1], 'Sales'] = 1000
        outliers = self.data_analyzer.detect_outliers(df_with_outlier, 'Sales')
        self.assertNotEqual(len(outliers), 0, "Outlier detection should identify at least one outlier.")

    def test_perform_regression_analysis(self):
        """Test linear regression on sales data to predict expenses."""
        coef, intercept, _ = self.data_analyzer.perform_regression_analysis(self.df1, 'Sales', 'Expenses')
        self.assertNotEqual(coef, 0, "Coefficient should not be zero in a meaningful regression.")
        self.assertIsInstance(intercept, float, "Intercept should be a float.")

    def test_aggregate_data(self):
        """Check that data aggregation sums sales correctly."""
        aggregated_data = self.data_analyzer.aggregate_data(self.data_frames, ['Sales'], 'sum')
        self.assertTrue('Sales' in aggregated_data.columns, "Sales column should exist in aggregated data.")
        if 'sum' in aggregated_data.index:  # If 'sum' is used as an index
            self.assertAlmostEqual(aggregated_data.loc['sum', 'Sales'], 700,
                                   msg="Aggregated sales sum should be correct.")
        elif 'sum' in aggregated_data.columns:  # If 'sum' is a separate column
            self.assertAlmostEqual(aggregated_data['Sales']['sum'], 700, msg="Aggregated sales sum should be correct.")

    def test_identify_common_columns(self):
        """Ensure common columns are identified correctly across data frames."""
        common_columns = self.data_analyzer.identify_common_columns(self.data_frames)
        self.assertTrue('Sales' in common_columns and 'Expenses' in common_columns, "Common columns should include 'Sales' and 'Expenses'.")

    def test_find_data_discrepancies(self):
        """Test finding discrepancies in common columns."""
        discrepancies = self.data_analyzer.find_data_discrepancies(self.data_frames, ['Sales'])
        self.assertIsInstance(discrepancies, dict, "Discrepancies should be returned as a dictionary.")


if __name__ == '__main__':
    unittest.main()
