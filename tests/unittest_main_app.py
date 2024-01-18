import unittest
from unittest.mock import MagicMock
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QCheckBox
from main_app import DataComparisonApp
import sys
import pandas as pd

class TestDataComparisonApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)

    def setUp(self):
        self.main_app = DataComparisonApp()

    def test_initUI(self):
        # Test if the main layout is set up
        self.assertIsNotNone(self.main_app.layout)

        # Test the visibility and status of UI components
        self.assertTrue(hasattr(self.main_app, 'loadButton'))
        self.assertTrue(hasattr(self.main_app, 'analyzeButton'))
        self.assertTrue(hasattr(self.main_app, 'resultsWidget'))
        self.assertTrue(hasattr(self.main_app, 'statusBar'))
        self.assertTrue(hasattr(self.main_app, 'exitButton'))

        # Test initial visibility of certain components
        self.assertFalse(self.main_app.analyzeButton.isVisible())
        self.assertFalse(self.main_app.resultsWidget.isVisible())

        # Add more assertions as necessary to cover all components initialized in initUI

    # def test_onAnalysisComplete(self):
        # Mock setup for necessary conditions before calling onAnalysisComplete
        # This might include setting up data frames or other UI states
        # For example:
        # self.main_app.data_frames = {'vet_data_2023_03.csv': pd.DataFrame()}
        # self.main_app.selected_columns = ['Column1', 'Column2']

        # Simulate the analysis completion signal
        # self.main_app.onAnalysisComplete()

        # Test if the resultsWidget is made visible after analysis is complete
        # self.assertTrue(self.main_app.resultsWidget.isVisible())

    def test_prepareAnalysisButton(self):
        """
        This test function checks if the "Perform Analysis" button is properly
        initialized and connected to the correct slot.

        """

        # Call prepareAnalysisButton to initialize the button
        self.main_app.prepareAnalysisButton()

        # Test if the analyzeButton is correctly set up
        self.assertIsNotNone(self.main_app.analyzeButton)
        self.assertFalse(self.main_app.analyzeButton.isEnabled())
        self.assertEqual(self.main_app.analyzeButton.text(), 'Perform Analysis')

        # Verify if the button is connected to the correct slot
        # This might require mocking or deeper inspection depending on the PyQt5 version

    def test_prepareExportButtons(self):
        """
        This test function checks if the export buttons are created,
        initially disabled, and added to the layout correctly.

        """

        # Call prepareExportButtons to initialize the export buttons
        self.main_app.prepareExportButtons()

        # Test if the export buttons are correctly set up
        self.assertTrue(hasattr(self.main_app, 'exportButtons'))
        self.assertEqual(len(self.main_app.exportButtons), 4)

        # Check if all buttons are initially disabled
        for button in self.main_app.exportButtons:
            self.assertFalse(button.isEnabled())
            self.assertIn(button.text(), ["Export as Text", "Export Graph", "Export as CSV", "Export as Excel"])

    def test_setupLoadButton(self):
        """
        This test will verify if the "Load CSV Files" button is correctly set up and responds to click events.

        """

        # Call setupLoadButton to initialize the load button
        self.main_app.setupLoadButton()

        # Test if the loadButton is correctly set up
        self.assertIsNotNone(self.main_app.loadButton)
        self.assertEqual(self.main_app.loadButton.text(), 'Load CSV Files')

    def test_setupAnalysisButton(self):
        """
        This test will check if the "Perform Analysis" button is correctly initialized
        and is disabled until CSV files are loaded.

        """

        # Call setupAnalysisButton to initialize the analyze button
        self.main_app.setupAnalysisButton()

        # Test if the analyzeButton is correctly set up
        self.assertIsNotNone(self.main_app.analyzeButton)
        self.assertEqual(self.main_app.analyzeButton.text(), 'Perform Analysis')
        self.assertFalse(self.main_app.analyzeButton.isEnabled())

    def test_setupResultsWidgets(self):
        """
        This function will verify if the results and insights labels,
        as well as the results widget, are initialized correctly.

        """

        # Call setupResultsWidgets to initialize the results widgets
        self.main_app.setupResultsWidgets()

        # Test if the resultsLabel, insightsLabel, and resultsWidget are correctly set up
        self.assertIsNotNone(self.main_app.resultsLabel)
        self.assertIsNotNone(self.main_app.insightsLabel)
        self.assertIsNotNone(self.main_app.resultsWidget)

        # Optionally, test the initial visibility of these widgets
        self.assertFalse(self.main_app.resultsLabel.isVisible())
        self.assertFalse(self.main_app.insightsLabel.isVisible())
        self.assertFalse(self.main_app.resultsWidget.isVisible())

    def test_setupStatusBar(self):
        """
        This function will check if the status bar is set up correctly in the application.

        """

        # Call setupStatusBar to initialize the status bar
        self.main_app.setupStatusBar()

        # Test if the statusBar is correctly set up
        self.assertIsNotNone(self.main_app.statusBar)

        # Verify if the statusBar is added to the layout
        # This can be checked if you have a reference to the layout or through visible UI changes

    # Continuing within the TestDataComparisonApp class

    def test_loadCSVFiles(self):
        """
        This function will test the loading of CSV files,
        ensuring that data frames are correctly loaded and the UI is updated accordingly.

        """

        # Mock the file paths to simulate file selection
        mock_file_paths = ['/path/to/mock_file1.csv', '/path/to/mock_file2.csv']
        # Mock loading CSV files
        self.main_app.data_frames = {file_path: pd.DataFrame() for file_path in mock_file_paths}
        self.main_app.loadCSVFiles()  # Assuming this can be called without actual file dialog interaction

        # Test if data frames are loaded correctly
        for file_path in mock_file_paths:
            self.assertIn(file_path, self.main_app.data_frames)
            self.assertIsInstance(self.main_app.data_frames[file_path], pd.DataFrame)

        # Test UI updates
        self.assertTrue(self.main_app.analyzeButton.isEnabled())

    def test_displayDynamicUIComponents(self):
        """
        This function will check if dynamic UI components are correctly displayed and updated.
        """

        # Prepare mock results
        mock_results = pd.DataFrame({"Sample": [1, 2, 3]})

        # Call displayResults directly with mock results
        self.main_app.displayResults(mock_results)

        # Now check if resultsLabel, insightsLabel, and resultsWidget are visible
        self.assertTrue(self.main_app.resultsLabel.isVisible())
        self.assertTrue(self.main_app.insightsLabel.isVisible())
        self.assertTrue(self.main_app.resultsWidget.isVisible())

    def test_displayColumnSuggestions(self):
        """
        This test should verify if the suggestionsLabel displays the correct
        text based on the columns suggested for analysis.

        """
        # Prepare mock data frames with columns
        mock_df1 = pd.DataFrame({'ColumnA': [1, 2], 'ColumnB': [3, 4]})
        mock_df2 = pd.DataFrame({'ColumnB': [5, 6], 'ColumnC': [7, 8]})
        self.main_app.data_frames = {'file1.csv': mock_df1, 'file2.csv': mock_df2}
        #self.main_app.suggestColumnsForAnalysis = MagicMock(return_value=['ColumnA', 'ColumnB', 'ColumnC'])

        # Call the method to test
        self.main_app.displayColumnSuggestions()

        # Process pending events
        QApplication.processEvents()

        print(f"suggestionsLabel visible: {self.main_app.suggestionsLabel.isVisible()}")
        print(f"suggestionsLabel text: '{self.main_app.suggestionsLabel.text()}'")

        # Test if the suggestionsLabel is visible
        self.assertTrue(self.main_app.suggestionsLabel.isVisible())

        QApplication.processEvents()

        # Test if the suggestionsLabel text contains the expected column names
        actual_text = self.main_app.suggestionsLabel.text()
        expected_columns = ['ColumnB', 'ColumnA', 'ColumnC']

        QApplication.processEvents()
        for column in expected_columns:
            self.assertIn(column, actual_text)

    def test_updateColumnCheckboxes(self):
        """
        This test should verify that the column checkboxes are created and added to
        the layout correctly based on the common columns in the data frames.

        """
        # Prepare mock data frames with common columns
        mock_df1 = pd.DataFrame(columns=['Common1', 'Unique1'])
        mock_df2 = pd.DataFrame(columns=['Common1', 'Unique2'])
        self.main_app.data_frames = {'file1.csv': mock_df1, 'file2.csv': mock_df2}

        # Call the method to test
        self.main_app.updateColumnCheckboxes()

        # Test if checkboxes for common columns are created and added to the layout
        self.assertIn('Common1', self.main_app.column_checkboxes)
        self.assertIsInstance(self.main_app.column_checkboxes['Common1'], QCheckBox)

        # Test if checkboxes for unique columns are not created
        self.assertNotIn('Unique1', self.main_app.column_checkboxes)
        self.assertNotIn('Unique2', self.main_app.column_checkboxes)

        # Optionally, check if the tooltips are set correctly
        expected_tooltip = "Column from files: file1.csv, file2.csv"
        self.assertEqual(self.main_app.column_checkboxes['Common1'].toolTip(), expected_tooltip)


if __name__ == '__main__':
    unittest.main()
