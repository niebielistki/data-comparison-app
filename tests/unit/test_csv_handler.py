import unittest
import os
import shutil
import pandas as pd
from app.csv_handler import CSVHandler  # Adjust the import path based on your project structure


class TestCSVHandler(unittest.TestCase):
    """
    Unit tests for the CSVHandler class.

    Tests cover reading CSV files, identifying common columns, mapping columns to files,
    validating column content, analyzing comparison scenarios, data types, and generating
    comparison possibilities and suggestions.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up a temporary directory and populate it with CSV files for testing.
        This method runs once before all tests.
        """
        cls.test_dir = "test_data"
        os.makedirs(cls.test_dir, exist_ok=True)
        # Create sample CSV files for testing
        data = {'Email': ['test@example.com', 'example@test.com'],
                'Age': [25, 30]}
        cls.sample_csv_path = os.path.join(cls.test_dir, "sample.csv")
        pd.DataFrame(data).to_csv(cls.sample_csv_path, index=False)

        # Initialize CSVHandler
        cls.csv_handler = CSVHandler()

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the temporary directory after all tests.
        """
        shutil.rmtree(cls.test_dir)

    def test_read_csv_files(self):
        """
        Test reading CSV files.
        """
        data_frames = self.csv_handler.read_csv_files(self.test_dir)
        self.assertIsInstance(data_frames, dict)
        self.assertTrue(data_frames, "Expected non-empty dictionary of data frames.")
        self.assertIn("sample.csv", data_frames)

    def test_map_columns_to_files(self):
        """
        Test mapping columns to the files they are contained in.
        """
        data_frames = self.csv_handler.read_csv_files(self.test_dir)
        column_file_map = self.csv_handler.map_columns_to_files(data_frames)
        self.assertIsInstance(column_file_map, dict)
        self.assertTrue(all(column in column_file_map for column in ['Email', 'Age']),
                        "Expected columns to be mapped to files.")

    def test_validate_column_content(self):
        """
        Test validating column content against specified criteria.
        """
        data_frames = self.csv_handler.read_csv_files(self.test_dir)
        validation_results = self.csv_handler.validate_column_content(data_frames, 'Email', 'email')
        self.assertIsInstance(validation_results, dict)
        # Check that each Series in validation_results has all True values
        for result in validation_results.values():
            self.assertTrue(result.all(), "Expected all email validations to pass.")

    def test_analyze_comparison_scenarios(self):
        """
        Test analyzing comparison scenarios based on column-file mappings.
        """
        data_frames = self.csv_handler.read_csv_files(self.test_dir)
        column_file_map = self.csv_handler.map_columns_to_files(data_frames)
        comparison_scenarios = self.csv_handler.analyze_comparison_scenarios(column_file_map)
        self.assertIsInstance(comparison_scenarios, dict)
        self.assertIn('with_common_columns', comparison_scenarios)

    def test_analyze_data_types(self):
        """
        Test analyzing data types in the given CSV files.
        """
        data_frames = self.csv_handler.read_csv_files(self.test_dir)
        data_type_report = self.csv_handler.analyze_data_types(data_frames)
        self.assertIsInstance(data_type_report, dict)
        self.assertTrue('sample.csv' in data_type_report, "Expected 'sample.csv' in data type report.")

    def test_generate_comparison_possibilities(self):
        """
        Test generating comparison possibilities from comparison scenarios.
        """
        data_frames = self.csv_handler.read_csv_files(self.test_dir)
        column_file_map = self.csv_handler.map_columns_to_files(data_frames)
        comparison_scenarios = self.csv_handler.analyze_comparison_scenarios(column_file_map)
        comparison_possibilities = self.csv_handler.generate_comparison_possibilities(comparison_scenarios)
        self.assertIsInstance(comparison_possibilities, dict)

    def test_suggest_comparisons(self):
        """
        Test suggesting comparisons based on comparison possibilities.
        """
        data_frames = self.csv_handler.read_csv_files(self.test_dir)
        column_file_map = self.csv_handler.map_columns_to_files(data_frames)
        comparison_scenarios = self.csv_handler.analyze_comparison_scenarios(column_file_map)
        comparison_possibilities = self.csv_handler.generate_comparison_possibilities(comparison_scenarios)
        suggestions = self.csv_handler.suggest_comparisons(comparison_possibilities)
        self.assertIsInstance(suggestions, dict)


if __name__ == '__main__':
    unittest.main()
