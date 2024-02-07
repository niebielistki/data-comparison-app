import unittest
from app.data_analyzer import DataAnalyzer
import pandas as pd
import os

test_data_directory = '/Users/wiktoria/PycharmProjects/DataComparisonApp/tests/data/data_6'

class TestDataAnalyzer(unittest.TestCase):

    def setUp(self):
        self.data_analyzer = DataAnalyzer()

    def test_compare_data(self):
        # Test the compare_data method of DataAnalyzer
        test_data_frames = {"file1.csv": pd.DataFrame(), "file2.csv": pd.DataFrame()}  # Replace with dummy or real data
        selected_columns = ["Column1", "Column2"]
        comparison_results = self.data_analyzer.compare_data(test_data_frames, selected_columns)
        self.assertIsInstance(comparison_results, pd.DataFrame)
        # Add more assertions to check the correctness of comparison results

    def test_find_data_discrepancies(self):
        # Test the find_data_discrepancies method of DataAnalyzer
        test_data_frames = self.load_test_data()
        common_columns = self.data_analyzer.identify_common_columns(test_data_frames)
        discrepancies = self.data_analyzer.find_data_discrepancies(test_data_frames, common_columns)
        self.assertIsInstance(discrepancies, dict)
        # Add more assertions to check the correctness of discrepancies results

    def test_calculate_statistics(self):
        # Test the calculate_statistics method of DataAnalyzer
        test_data_frames = self.load_test_data()
        common_columns = self.data_analyzer.identify_common_columns(test_data_frames)
        statistics = self.data_analyzer.calculate_statistics(test_data_frames, common_columns)
        self.assertIsInstance(statistics, dict)
        # Assert that statistics contain expected keys like 'mean', 'median', 'mode'
        for column_stats in statistics.values():
            self.assertIn('mean', column_stats)
            self.assertIn('median', column_stats)
            self.assertIn('mode', column_stats)

    def load_test_data(self):
        # Utility function to load test data
        test_data_frames = {}
        for file_name in os.listdir(test_data_directory):
            if file_name.endswith('.csv'):
                file_path = os.path.join(test_data_directory, file_name)
                test_data_frames[file_name] = pd.read_csv(file_path)
        return test_data_frames

if __name__ == '__main__':
    unittest.main()














