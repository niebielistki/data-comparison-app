#! python3
# main_app.py

import os
import sys
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QCheckBox, QApplication, QStatusBar)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal, QObject
from collections import defaultdict
from PyQt5.QtWidgets import QFileDialog
from csv_handler import CSVHandler
from data_analyzer import DataAnalyzer
from results_widget import ResultsWidget
from PyQt5.QtWidgets import QScrollArea
from PyQt5.QtWidgets import QComboBox
import numpy as np
import traceback

class DataComparisonApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data Comparison App')
        self.csv_handler = CSVHandler()
        self.analyzer = DataAnalyzer()
        self.resultsWidget = ResultsWidget(self)
        self.data_frames = {}
        self.selected_columns = []
        self.column_checkboxes = {}

        # Initialize the UI components (which includes setting up layout)
        self.initUI()

        # Connect signal only once
        self.analyzer.analysisComplete.connect(self.resultsWidget.updateAndShow)

    def initUI(self):
        print("initUI called")
        # Main layout for the entire application
        self.layout = QVBoxLayout(self)

        # Load button at the top
        self.setupLoadButton()

        # Automatic analysis checkbox and label
        self.autoAnalysisLayout = QHBoxLayout()
        self.setupAutomaticAnalysisCheckbox()
        self.setupSuggestedColumnsLabel()
        self.layout.addLayout(self.autoAnalysisLayout)

        # Dynamic layout for column checkboxes
        self.dynamicLayout = QVBoxLayout()
        self.setupColumnCheckboxes()
        self.layout.addLayout(self.dynamicLayout)

        # Initialize time series priority UI components
        self.setupTimeSeriesPriorityUI()

        # "Perform Analysis" button
        self.prepareAnalysisButton()
        self.layout.addWidget(self.analyzeButton)

        # Insights Label
        self.insightsLabel = QLabel("Insights will be shown here.", self)
        self.insightsLabel.setAlignment(Qt.AlignCenter)
        self.insightsLabel.hide()
        self.layout.addWidget(self.insightsLabel)

        # Initialize and add the ResultsWidget
        # self.resultsWidget = ResultsWidget(self)
        self.layout.addWidget(self.resultsWidget)

        self.resultsWidget.hide()

        # Status bar at the bottom
        self.setupStatusBar()

        # Exit button at the bottom
        self.setupExitButton()

        # Initially hide elements that should only be shown after loading CSV files
        self.checkAutomaticAnalysis.hide()
        self.analyzeButton.hide()

    def onAnalysisComplete(self, analysis_results):
        # Assuming analysis_results is a dictionary with 'numerical_results' and 'textual_results'
        self.resultsWidget.updateAndShow(analysis_results)

    def prepareAnalysisButton(self):
        self.analyzeButton = QPushButton('Perform Analysis', self)
        self.analyzeButton.clicked.connect(self.performAnalysis)
        self.analyzeButton.setEnabled(False)
        # Do not add the button to self.layout here

    def setupLoadButton(self):
        self.loadButton = QPushButton('Load CSV Files', self)
        self.loadButton.clicked.connect(self.loadCSVFiles)
        self.layout.addWidget(self.loadButton)

    def setupResultsWidgets(self):
        # Make the insights label and results widget visible
        # self.resultsLabel.show()
        self.insightsLabel.show()
        self.resultsWidget.show()

    def setupStatusBar(self):
        self.statusBar = QStatusBar()
        self.layout.addWidget(self.statusBar)

    def loadCSVFiles(self):
        print("Loading CSV Files...")
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Select CSV files", "", "CSV Files (*.csv)")
        if file_paths:
            successfully_loaded = False
            for file_path in file_paths:
                try:
                    df = pd.read_csv(file_path)
                    self.data_frames[file_path] = df
                    self.statusBar.showMessage(f"Loaded CSV file: {file_path}", 5000)
                    successfully_loaded = True
                    print(f"File loaded: {file_path}")
                except Exception as e:
                    self.statusBar.showMessage(f"Error loading {file_path}: {e}", 10000)

            if successfully_loaded:
                self.updateColumnCheckboxes()
                self.displayPreAnalysisInsights()
                self.displayColumnSuggestions()

                # Call detect_time_columns here and update UI accordingly
                detected_time_columns = self.analyzer.detect_time_columns(self.data_frames)
                self.updateTimeSeriesPriorityUI(detected_time_columns)

                self.analyzeButton.setEnabled(True)
                self.analyzeButton.show()
                self.checkAutomaticAnalysis.show()
                self.performTextualAnalysis()  # Textual Analysis

    def updateTimeSeriesPriorityUI(self, detected_time_columns):
        # Convert the defaultdict to a regular dict for .keys() to work correctly
        detected_time_columns = dict(detected_time_columns)
        if len(detected_time_columns) > 1:
            self.timeSeriesPriorityComboBox.clear()
            self.timeSeriesPriorityComboBox.addItems(detected_time_columns.keys())
            self.timeSeriesPriorityLabel.show()
            self.timeSeriesPriorityComboBox.show()
        else:
            self.timeSeriesPriorityLabel.hide()
            self.timeSeriesPriorityComboBox.hide()

        # Reset the checkbox for automatic analysis
        self.checkAutomaticAnalysis.setChecked(False)

    def load_multiple_csv(self, folder_path):
        """
        Loads multiple CSV files from the specified folder and preprocesses them.
        :param folder_path: Path to the folder containing CSV files.
        """
        self.data_frames = {}
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                try:
                    # Utilize load_and_preprocess_data method of DataAnalyzer
                    preprocessed_df = self.analyzer.load_and_preprocess_data(file_path)
                    if not preprocessed_df.empty:
                        self.data_frames[file] = preprocessed_df
                    else:
                        print(f"Empty or invalid data in file: {file}")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    def performTextualAnalysis(self):
        """
        Initiates the textual analysis process for the loaded data.
        """
        if not self.data_frames:
            print("No data frames available for analysis.")
            return

        # Retrieve user-selected time series priority
        selected_time_series_priority = self.timeSeriesPriorityComboBox.currentText() if self.timeSeriesPriorityComboBox else None

        # Perform the analysis using the analyzer's analyze_time_series method
        analysis_results = self.analyzer.analyze_time_series(self.data_frames, selected_time_series_priority)

        # Check if time-related analysis was performed
        if "Time Series Analysis" in analysis_results:
            # Display the results in the UI
            self.displayTextualAnalysisResults(analysis_results)
        else:
            print("No time-related analysis to display.")

    def displayTextualAnalysisResults(self, results):
        """
        Displays the results of textual analysis in the UI.
        """
        if not results:
            print("No results to display.")
            self.statusBar.showMessage("No textual analysis results available.")
            return

        # Update the results widget with the analysis results
        self.resultsWidget.updateAndShow({'textual_results': results})

    def setupTimeSeriesPriorityUI(self):
        # Create a label for the time series priority section
        self.timeSeriesPriorityLabel = QLabel("Prioritize Time Series Analysis:", self)
        self.layout.addWidget(self.timeSeriesPriorityLabel)

        # Create a combo box for time series prioritization
        self.timeSeriesPriorityComboBox = QComboBox(self)
        self.timeSeriesPriorityComboBox.addItems(
            ['Date', 'Month', 'Year'])  # Default options, update these based on detected columns
        self.layout.addWidget(self.timeSeriesPriorityComboBox)

        # Initially hide these components, they should be shown after loading CSV files and detecting time columns
        self.timeSeriesPriorityLabel.hide()
        self.timeSeriesPriorityComboBox.hide()

    def displayDynamicUIComponents(self):
        # Clear previous dynamic UI components if any
        self.clearLayout(self.dynamicLayout)

        # Suggest columns for analysis
        self.suggestColumnsForAnalysis()

        # Set up the dynamic UI components
        self.setupAutomaticAnalysisCheckbox()
        self.updateColumnCheckboxes()

        # Make the insights label and results widget visible
        # self.resultsLabel.show()
        self.insightsLabel.show()
        self.resultsWidget.show()

    def displayColumnSuggestions(self):
        # This method will display the suggested columns in the insights label
        column_suggestions = self.suggestColumnsForAnalysis()
        if column_suggestions:
            suggestion_text = "Suggested columns for analysis based on data completeness:\n" + \
                              "\n".join(f"- {column}" for column in column_suggestions)
            self.suggestionsLabel.setText(suggestion_text)
            self.suggestionsLabel.show()
        else:
            self.suggestionsLabel.hide()


    def suggestColumnsForAnalysis(self):
        # Suggest columns for analysis based on data completeness
        column_completeness = defaultdict(int)
        for df in self.data_frames.values():
            for column in df.columns:
                # Count non-null values
                non_null_count = df[column].count()
                if non_null_count > 0:
                    column_completeness[column] += non_null_count

        # Sort columns based on completeness and return the top suggestions
        sorted_columns = sorted(column_completeness.items(), key=lambda item: item[1], reverse=True)
        suggested_columns = [column for column, _ in sorted_columns[:5]]  # Get top 5 suggestions
        return suggested_columns

    def setupAutomaticAnalysisCheckbox(self):
        self.checkAutomaticAnalysis = QCheckBox("Perform automatic analysis", self)
        self.checkAutomaticAnalysis.stateChanged.connect(self.automatedAnalysisStateChanged)
        self.layout.addWidget(self.checkAutomaticAnalysis)

    def setupSuggestedColumnsLabel(self):
        # This label will be used as a placeholder and will be shown/hidden as needed
        self.suggestionsLabel = QLabel("", self)
        self.layout.addWidget(self.suggestionsLabel)
        self.suggestionsLabel.hide()

    def setupColumnCheckboxes(self):
        # Initialize the container for column checkboxes
        self.columnCheckboxesLayout = QVBoxLayout()  # Add this line to initialize the layout
        self.dynamicLayout.addLayout(self.columnCheckboxesLayout)

    def updateColumnCheckboxes(self):
        # Clear existing checkboxes
        self.clearLayout(self.columnCheckboxesLayout)
        self.column_checkboxes = {}

        common_columns = self.csv_handler.identify_common_columns(self.data_frames)
        for column in common_columns:
            checkbox = QCheckBox(column, self)
            checkbox.stateChanged.connect(self.checkboxChanged)
            # Pass self.data_frames as an argument
            checkbox.setToolTip(
                f"Column from files: {', '.join(self.csv_handler.files_containing_column(column, self.data_frames))}")
            self.column_checkboxes[column] = checkbox
            self.columnCheckboxesLayout.addWidget(checkbox)

        # Adjust layout formatting for better readability
        self.columnCheckboxesLayout.addStretch(1)

    def checkboxChanged(self, state):
        # Update the class member with the currently selected columns
        self.selected_columns = [checkbox.text() for checkbox in self.column_checkboxes.values() if checkbox.isChecked()]
        print("checkboxChanged called. Currently selected columns:", self.selected_columns)

    def performAnalysis(self):
        try:
            print("Performing analysis...")
            print("Data frames available for analysis:", self.data_frames)

            # Check if the automatic analysis checkbox is checked
            if self.checkAutomaticAnalysis.isChecked():
                print("Automatic column selection is enabled.")
                # Call the method to automatically select columns
                self.selected_columns = self.analyzer.automatic_column_selection(self.data_frames)
                print("Selected columns for automatic analysis:", self.selected_columns)

            # Filter the data_frames to only include the selected columns
            filtered_data_frames = {path: df[self.selected_columns] for path, df in self.data_frames.items() if
                                    set(self.selected_columns).issubset(df.columns)}

            print("Filtered data frames for analysis:", filtered_data_frames)

            if filtered_data_frames:
                print("Filtered data frames are not empty, proceeding with analysis.")
                numerical_results, textual_results = self.analyzer.classify_and_analyze_data(filtered_data_frames)

                # Debug prints to inspect types and contents
                print("numerical_results type:", type(numerical_results))
                print("numerical_results content:", numerical_results)
                print("textual_results type:", type(textual_results))
                print("textual_results content:", textual_results)

                # Pass the results directly to updateAndShow
                print("Analysis completed. Updating results widget...")
                self.resultsWidget.updateAndShow(
                    {'numerical_results': numerical_results, 'textual_results': textual_results})

            else:
                print("Filtered data frames are empty.")
                self.statusBar.showMessage("Please select columns for analysis.")


        except KeyError as ke:
            print(f"A KeyError occurred: {ke}")
            self.handleError(f"Analysis failed due to a missing column: {ke}")

        except AttributeError as ae:
            print(f"An AttributeError occurred: {ae}")
            self.handleError(f"Analysis failed due to a wrong attribute: {ae}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            self.handleError(f"An unexpected error occurred during analysis: {e}")

    def handleError(self, message):
        print(message)  # Ensure that error messages are printed to the console for debugging
        self.statusBar.showMessage(message)

    def setupButtons(self):
        self.exitButton = QPushButton('Exit', self)
        self.exitButton.clicked.connect(self.close)
        self.layout.addWidget(self.exitButton)

    def clearLayout(self, layout):
        while layout is not None and layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def getColumnMetaData(self, column):
        # Get meta-information for the column from all dataframes
        meta_info = f"Column: {column}\n"
        for file_path, df in self.data_frames.items():
            if column in df.columns:
                meta_info += f"From file: {os.path.basename(file_path)}, Non-null values: {df[column].count()}\n"
        return meta_info.strip()

    def displayColumnMetaData(self):
        # Display meta-information about each column
        for column, checkbox in self.column_checkboxes.items():
            meta_info = self.getColumnMetaData(column)
            checkbox.setToolTip(meta_info)

    def displayPreAnalysisInsights(self):
        selected_columns = [checkbox.text() for checkbox in self.column_checkboxes.values() if checkbox.isChecked()]
        insights_text = "<ul>"

        if selected_columns:
            # Assuming calculate_statistics is implemented and returns expected results
            stats = self.analyzer.calculate_statistics(self.data_frames, selected_columns)
            for column in selected_columns:
                column_stats = stats.get(column, {})
                insights_text += f"<li>Column: {column}<ul>"
                for key, value in column_stats.items():
                    insights_text += f"<li>{key.capitalize()}: {value}</li>"
                insights_text += "</ul></li>"

        insights_text += "</ul>"
        self.insightsLabel.setText(insights_text.strip())

        #Other methods that don't mention in your correct code example.

    def setupExitButton(self):
        self.exitButton = QPushButton('Exit', self)
        self.exitButton.clicked.connect(self.close)
        self.layout.addWidget(self.exitButton)

    def automatedAnalysisStateChanged(self, state):
        if state == Qt.Checked:
            # Disable manual column checkboxes as we are using automatic selection
            for checkbox in self.column_checkboxes.values():
                checkbox.setEnabled(False)
        else:
            # Enable manual column checkboxes when automatic selection is not used
            for checkbox in self.column_checkboxes.values():
                checkbox.setEnabled(True)

    def displayNLPResults(self, nlp_results):
        """
        Displays the results of NLP analysis in the GUI.
        """
        # Convert the NLP results to a human-readable string
        results_text = "NLP Analysis Results:\n"
        for column, analysis in nlp_results.items():
            results_text += f"\nColumn: {column}\n"
            for key, value in analysis.items():
                results_text += f"{key.capitalize()}: {value}\n"

        # Display in the insights label or another dedicated UI component
        self.insightsLabel.setText(results_text)

    def displayResults(self, results):
        """
        Displays analysis results in the UI.
        """
        try:
            numerical_data = results.get('numerical_results', pd.DataFrame())
            textual_data = results.get('textual_results', "")

            print(
                f"Preparing to display results...\nNumerical data available for display: {isinstance(numerical_data, pd.DataFrame) and not numerical_data.empty}\nTextual data available for display: {isinstance(textual_data, (pd.DataFrame, str)) and textual_data != ''}")

            # Display numerical data if available
            if isinstance(numerical_data, pd.DataFrame) and not numerical_data.empty:
                print("Displaying numerical data...")
                self.resultsWidget.displayGraphResults(numerical_data)

            # Display textual data if available
            if isinstance(textual_data, pd.DataFrame) and not textual_data.empty:
                print("Displaying textual DataFrame data...")
                self.resultsWidget.displayTextResults(textual_data)
            elif isinstance(textual_data, str) and textual_data:
                print("Displaying textual string data...")
                self.resultsWidget.displayTextResults(textual_data)

        except Exception as e:
            print(f"An error occurred during result display: {e}")
            traceback.print_exc()  # Added to print detailed stack trace


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = DataComparisonApp()
    main.show()
    sys.exit(app.exec_())