"""
This script tests the functionalities of the DataComparisonApp class, focusing on its ability to initialize properly,
handle data loading, perform analysis, and display results.
"""

import unittest
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtCore import Qt
from unittest.mock import patch
from app.main_app import DataComparisonApp
from app.results_widget import ResultsWidget
import pandas as pd
import sys
import os

class TestDataComparisonApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize QApplication once for the test suite
        cls.app = QApplication(sys.argv)

    def setUp(self):
        # Create an instance of DataComparisonApp before each test
        self.app = DataComparisonApp()

    def test_initial_state(self):
        """Test the initial state of the application."""
        # Verify that the main UI components are initialized
        self.assertIsNotNone(self.app.loadButton, "Load button should be initialized.")
        self.assertIsNotNone(self.app.analyzeButton, "Analyze button should be initialized.")
        self.assertIsNotNone(self.app.resultsWidget, "Results widget should be initialized.")

    @patch('PyQt5.QtWidgets.QFileDialog.getOpenFileNames')
    def test_loadCSVFiles(self, mock_getOpenFileNames):
        """Test loading CSV files into the application with mocking."""
        mock_file_paths = ['tests/data/data_1/sales.csv', 'tests/test_data/data_2/team_info.csv']
        mock_getOpenFileNames.return_value = (mock_file_paths, '')

        self.app.loadCSVFiles()

        for file_path in mock_file_paths:
            self.assertIn(file_path, self.app.data_frames)
            self.assertIsInstance(self.app.data_frames[file_path], pd.DataFrame)

    def test_performAnalysis(self):
        """Simulate analysis trigger and check for results display."""
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        self.app.data_frames['test_data.csv'] = test_df

        with patch.object(self.app.analyzer, 'classify_and_analyze_data', return_value=(pd.DataFrame(), [])):
            self.app.performAnalysis()
            self.assertTrue(self.app.resultsWidget.isUpdated)

    @patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName')
    def test_exportResults(self, mock_getSaveFileName):
        # Assuming self.app is the correct instance of your application
        mock_save_path = 'tests/exported_results/test_export.csv'
        mock_getSaveFileName.return_value = (mock_save_path, '')

        # Populate self.app.resultsWidget.df with a dummy DataFrame before exporting
        self.app.resultsWidget.df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

        # Call the export function
        self.app.resultsWidget.exportToCSV()

        # Verify the file was created
        self.assertTrue(os.path.exists(mock_save_path))

        # Cleanup
        os.remove(mock_save_path)

    def tearDown(self):
        """Cleanup any changes made to the filesystem or application state."""

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()

if __name__ == '__main__':
    unittest.main()
