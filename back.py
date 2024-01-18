#! python3
# main_app.py

import os
import sys
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QCheckBox, QApplication, QStatusBar)
from PyQt5.QtCore import Qt
from collections import defaultdict

from csv_handler import CSVHandler
from data_analyzer import DataAnalyzer
from results_widget import ResultsWidget

class DataComparisonApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data Comparison App')
        self.csv_handler = CSVHandler()
        self.analyzer = DataAnalyzer()
        self.data_frames = {}
        self.selected_columns = []
        self.column_checkboxes = {}
        self.dynamicLayout = QVBoxLayout()  # Initialize here before using it
        self.initUI()
        self.setupDynamicUIComponents()  # This will ensure dynamic components are created.
        self.hideDynamicComponents()
    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.dynamicLayout)  # Add dynamic layout to the main layout here
        self.setupLoadButton()
        self.setupResultsWidgets()
        self.setupExportButtons()
        self.setupStatusBar()
        self.setupExitButton()
        self.setLayout(self.layout)

    def setupLoadButton(self):
        self.loadButton = QPushButton('Load CSV Files', self)
        self.loadButton.clicked.connect(self.loadCSVFiles)
        self.layout.addWidget(self.loadButton)

    def setupResultsWidgets(self):
        self.resultsLabel = QLabel('Analysis Results:', self)
        self.resultsLabel.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.resultsLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.resultsLabel)
        self.resultsLabel.hide()  # Initially hidden

        self.insightsLabel = QLabel("Insights will be shown here.", self)
        self.insightsLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.insightsLabel)
        self.insightsLabel.hide()  # Initially hidden

        self.resultsWidget = ResultsWidget(self)
        self.layout.addWidget(self.resultsWidget)
        self.resultsWidget.hide()  # Initially hidden
        # The export buttons themselves will be setup in setupExportButtons()

    def setupExportButtons(self):
        self.exportButtonsLayout = QHBoxLayout()
        self.exportTextButton = QPushButton("Export as Text", self)
        self.exportGraphButton = QPushButton("Export Graph", self)
        self.exportCSVButton = QPushButton("Export as CSV", self)
        self.exportExcelButton = QPushButton("Export as Excel", self)
        self.exportButtons = [
            self.exportTextButton, self.exportGraphButton,
            self.exportCSVButton, self.exportExcelButton
        ]
        for button in self.exportButtons:
            button.setEnabled(False)
            button.hide()  # Initially hidden
            self.exportButtonsLayout.addWidget(button)
        self.layout.addLayout(self.exportButtonsLayout)

    def setupStatusBar(self):
        self.statusBar = QStatusBar()
        self.layout.addWidget(self.statusBar)

    def setupExitButton(self):
        # Exit button setup to ensure it is placed at the bottom
        self.exitButton = QPushButton('Exit', self)
        self.exitButton.clicked.connect(self.close)
        self.layout.addWidget(self.exitButton)

    def loadCSVFiles(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Select CSV files", "", "CSV Files (*.csv)")
        if file_paths:
            self.clearDynamicUIComponents()  # Clear existing dynamic components
            success = False
            for file_path in file_paths:
                try:
                    df = pd.read_csv(file_path)
                    self.data_frames[file_path] = df
                    self.statusBar.showMessage(f"Loaded CSV file: {file_path}", 5000)
                    success = True
                except Exception as e:
                    self.statusBar.showMessage(f"Error loading {file_path}: {e}", 10000)
                    # Do not return, just notify the user and continue with other files
            if success:
                self.displayColumnSuggestions()  # This will show the suggestions label with text
                self.displayDynamicUIComponents()

    def setupDynamicUIComponents(self):
        # Create dynamic components if they haven't been set up already
        if not hasattr(self, 'analyzeButton'):
            self.analyzeButton = QPushButton('Perform Analysis', self)
            self.analyzeButton.clicked.connect(self.performAnalysis)
            self.dynamicLayout.addWidget(self.analyzeButton)
            self.analyzeButton.setEnabled(False)

            self.checkAutomaticAnalysis = QCheckBox("Perform automatic analysis", self)
            self.checkAutomaticAnalysis.stateChanged.connect(self.automatedAnalysisStateChanged)
            self.dynamicLayout.addWidget(self.checkAutomaticAnalysis)

            self.suggestionsLabel = QLabel("Suggested columns for analysis based on data completeness:", self)
            self.dynamicLayout.addWidget(self.suggestionsLabel)

            self.columnCheckboxesLayout = QVBoxLayout()
            self.dynamicLayout.addLayout(self.columnCheckboxesLayout)

    def displayDynamicUIComponents(self):
        # This function should make sure that all dynamic components are made visible
        self.checkAutomaticAnalysis.show()
        self.suggestionsLabel.show()
        self.analyzeButton.setEnabled(False)  # Enable should be false until columns are selected
        self.analyzeButton.show()
        self.updateColumnCheckboxes()

        # Make sure export buttons are still disabled and hidden until analysis is performed
        for button in self.exportButtons:
            button.setEnabled(False)  # Keep them disabled
            button.hide()

        self.resultsLabel.show()
        print(f"Is the 'Analysis Results' label visible? {self.resultsLabel.isVisible()}")
    def hideDynamicComponents(self):
        # This function should hide all dynamic components that are not needed immediately upon startup
        if hasattr(self, 'checkAutomaticAnalysis'):
            self.checkAutomaticAnalysis.hide()
        if hasattr(self, 'suggestionsLabel'):
            self.suggestionsLabel.hide()
        if hasattr(self, 'analyzeButton'):
            self.analyzeButton.hide()
        for checkbox in self.column_checkboxes.values():
            checkbox.hide()  # Hide all the checkboxes
        self.resultsLabel.hide()
        for button in self.exportButtons:
            button.hide()
        for button in self.exportButtons:
            button.hide()  # Also hide the layout containing the buttons
        self.resultsWidget.hide()

    def clearLayout(self, layout):
        # Remove all items from the layout properly
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                widget = item.widget()
                layout.removeItem(item)  # Remove the item from the layout
                widget.deleteLater()
            elif item.layout():
                self.clearLayout(item.layout())
                layout.removeItem(item)  # Remove the layout item

    def setupAnalysisButton(self):
        # This method now only creates the button without adding it to any layout
        self.analyzeButton = QPushButton('Perform Analysis', self)
        self.analyzeButton.clicked.connect(self.performAnalysis)
        self.analyzeButton.setEnabled(False)

    def clearDynamicUIComponents(self):
        if hasattr(self, 'dynamicLayout'):
            self.clearLayout(self.dynamicLayout)

    def displayColumnSuggestions(self):
        # This method will display the suggested columns in the insights label
        column_suggestions = self.suggestColumnsForAnalysis()
        if column_suggestions:
            suggestion_text = "Suggested columns for analysis based on data completeness:\n" + \
                              "\n".join(f"- {column}" for column in column_suggestions)
            self.suggestionsLabel.setText(suggestion_text)  # Set text to the correct label
            self.suggestionsLabel.show()  # Show the suggestions label
            if hasattr(self, 'analyzeButton'):
                self.analyzeButton.setEnabled(True)  # Enable the analyze button if it exists
        else:
            self.suggestionsLabel.setText("No columns suggested for analysis.")  # Show a message if no suggestions
            self.suggestionsLabel.show()

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
            checkbox.setToolTip(
                f"Column from files: {', '.join(self.csv_handler.files_containing_column(column, self.data_frames))}")
            self.column_checkboxes[column] = checkbox
            self.columnCheckboxesLayout.addWidget(checkbox)
            checkbox.show()

    def checkboxChanged(self, state):
        # Update the class member with the currently selected columns
        self.selected_columns = [checkbox.text() for checkbox in self.column_checkboxes.values() if
                                 checkbox.isChecked()]
        print("Currently selected columns:", self.selected_columns)  # For debugging

    def performAnalysis(self):
        try:
            if getattr(self, 'checkAutomaticAnalysis', None) and self.checkAutomaticAnalysis.isChecked():
                self.selected_columns = self.suggestColumnsForAnalysis()
                if not self.selected_columns:
                    self.statusBar.showMessage("No suitable columns found for automatic analysis.")
                    return

            if not self.selected_columns:
                self.statusBar.showMessage("No columns selected for analysis.")
                return

            comparison_results = self.analyzer.compare_data(self.data_frames, self.selected_columns)
            self.displayResults(comparison_results)
            self.resultsWidget.enableExportButtons()

        except AttributeError as e:
            self.handleError(f"An error occurred: {e}")
        except Exception as e:
            self.handleError(f"An error occurred during analysis: {e}")

    def setupButtons(self):
        self.exitButton = QPushButton('Exit', self)
        self.exitButton.clicked.connect(self.close)
        self.layout.addWidget(self.exitButton)

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

        #Other methods that don't mention in your correct code example

    def automatedAnalysisStateChanged(self, state):
        # If the checkbox is checked, disable manual column selection
        if state == Qt.Checked:
            for checkbox in self.column_checkboxes.values():
                checkbox.setEnabled(False)
        else:  # Enable manual column selection when checkbox is unchecked
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
            if isinstance(results, str):
                self.resultsWidget.resultsLabel.setText(results)
            else:
                self.resultsWidget.renderResults(results)
                self.resultsWidget.show()

        except AttributeError as e:
            self.handleError(f"An error occurred: {e}")
        except Exception as e:
            self.handleError(f"An error occurred during result display: {e}")

    def handleError(self, message):
        print(message)
        self.statusBar.showMessage(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = DataComparisonApp()
    main.show()
    sys.exit(app.exec_())