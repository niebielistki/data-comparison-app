#! python3
# results_widget.py

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QTabWidget, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QTextOption
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import traceback

class ResultsWidget(QWidget):
    def __init__(self, parent=None):
        super(ResultsWidget, self).__init__(parent)
        self.df = None
        self.initUI()

    def initUI(self):
        # Main vertical layout
        vbox = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.textTab = QWidget()  # Tab for textual results (tables)
        self.graphTab = QWidget()  # Tab for graphical results (plots)

        # Add tabs to the tab widget
        self.tabs.addTab(self.textTab, "Text")
        self.tabs.addTab(self.graphTab, "Graph")
        self.tabs.setStyleSheet("QTabBar::tab { width: 100px; }")

        # Graph Tab layout
        graphLayout = QVBoxLayout()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        graphLayout.addWidget(self.canvas)
        self.graphTab.setLayout(graphLayout)

        # Text Tab layout
        textLayout = QVBoxLayout()
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setLayout(QVBoxLayout())
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scrollAreaWidgetContents.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        textLayout.addWidget(self.scrollArea)
        self.textTab.setLayout(textLayout)

        # Results label above tabs
        self.resultsLabel = QLabel("Analysis Results", self)
        self.resultsLabel.setStyleSheet("font-size: 24px; font-weight: bold; margin-top: 5px;")
        vbox.addWidget(self.resultsLabel)

        # Add tabs to the main layout
        vbox.addWidget(self.tabs)

        # Horizontal layout for export buttons
        hbox = QHBoxLayout()
        self.initExportButtons(hbox)
        vbox.addLayout(hbox)  # This adds export buttons below the tabs

        self.setLayout(vbox)

    def initExportButtons(self, layout):
        # Create export buttons
        self.exportTextButton = QPushButton("Export as Text", self)
        self.exportGraphButton = QPushButton("Export Graph", self)
        self.exportCSVButton = QPushButton("Export as CSV", self)
        self.exportExcelButton = QPushButton("Export as Excel", self)

        # Connect buttons to their functions
        self.exportTextButton.clicked.connect(self.exportAsText)
        self.exportGraphButton.clicked.connect(self.exportGraph)
        self.exportCSVButton.clicked.connect(self.exportToCSV)
        self.exportExcelButton.clicked.connect(self.exportToExcel)

        # Add buttons to the layout
        layout.addWidget(self.exportTextButton)
        layout.addWidget(self.exportGraphButton)
        layout.addWidget(self.exportCSVButton)
        layout.addWidget(self.exportExcelButton)

        # Disable buttons initially
        self.enableExportButtons(False)

    def initScrollArea(self, layout):
        # Initialize the scroll area for displaying tables
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setLayout(QVBoxLayout())
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea.setMinimumHeight(400)
        self.scrollArea.hide()  # Initially hide it
        layout.addWidget(self.scrollArea)

    def enableExportButtons(self, enable=True):
        # Enable or disable export buttons
        self.exportTextButton.setEnabled(enable)
        self.exportGraphButton.setEnabled(enable)
        self.exportCSVButton.setEnabled(enable)
        self.exportExcelButton.setEnabled(enable)

    def renderResults(self, results):
        """
        This method handles the display of analysis results and summary in the appropriate tabs.
        """

        # Original handling for comparison results and summary
        df = results.get('comparison_results', pd.DataFrame())
        summary = results.get('summary', "")
        self.displaySummary(summary)

        # New handling for numerical and textual results
        numerical_data = results.get('numerical_results', pd.DataFrame())
        textual_data = results.get('textual_results', "")

        # Determine the type of data and update UI components accordingly
        if not df.empty:
            if 'Graph' in df:
                # Display the graph
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                df['Graph'].plot(kind='bar', ax=ax)
                self.canvas.draw()
                self.canvas.setVisible(True)
                self.scrollArea.setVisible(False)
                self.tabs.setCurrentIndex(self.tabs.indexOf(self.graphTab))
            elif 'Text' in df:
                # Display the table
                self.displayAsTable(df['Text'])
                self.scrollArea.setVisible(True)
                self.canvas.setVisible(False)
                self.tabs.setCurrentIndex(self.tabs.indexOf(self.textTab))

        elif not numerical_data.empty:
            print("Displaying graphical data...")
            self.displayGraphResults(numerical_data)

        elif textual_data:
            print("Displaying textual data...")
            self.displayTextResults(textual_data)

            # Enable export buttons if DataFrame is not empty
        self.enableExportButtons(not df.empty and not numerical_data.empty and textual_data != "")

    def format_analysis_as_html(self, results_sections):
        html_output = "<html><head><style>"
        html_output += "table { width:100%; border-collapse: collapse; }"
        html_output += "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }"
        html_output += "</style></head><body>"

        for section in results_sections:
            html_output += "<h2>{}</h2>".format(section['title'])
            if isinstance(section['content'], list):
                for item in section['content']:
                    if isinstance(item, dict) and 'title' in item and 'content' in item:
                        html_output += "<h3>{}</h3>".format(item['title'])
                        html_output += item['content']
                    else:
                        html_output += "<p>{}</p>".format(item)
            elif isinstance(section['content'], dict):
                for key, value in section['content'].items():
                    html_output += "<h3>{}</h3>".format(key)
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            html_output += "<p><b>{}</b>: {}</p>".format(sub_key, sub_value)
                    else:
                        html_output += "<p>{}</p>".format(value)
            else:
                html_output += "<p>{}</p>".format(section['content'])

        html_output += "</body></html>"
        return html_output

    def displayTextResults(self, results_sections):
        """
        Handles the display of textual analysis results with keys formatted for better readability.
        """
        try:
            if not results_sections:
                print("No textual data to display.")
                return

            # Prepends an arrow to each section title for display
            for section in results_sections:
                section['title'] = f"▶︎ {section['title']}"

            html_results = self.format_analysis_as_html(results_sections)

            # Display the results
            text_widget = QTextEdit()
            text_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            text_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            text_widget.setReadOnly(True)
            text_widget.setHtml(html_results)
            self.scrollAreaWidgetContents.layout().addWidget(text_widget)
            self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()  # This line prints the traceback of the exception

    def displayGraphResults(self, numerical_data):
        """
        Handles the display of graphical analysis results.
        """
        if numerical_data.empty:
            print("No numerical data to display.")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # You may want to adjust the kind of plot based on the data
        kind_of_plot = 'bar'  # Placeholder, adjust as needed

        numerical_data.plot(kind=kind_of_plot, ax=ax)
        self.canvas.draw()
        self.scrollAreaWidgetContents.layout().addWidget(self.canvas)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

    def displayAsTable(self, data):
        """
        Displays the given data in a table format within the ResultsWidget.
        """
        # Clear the layout of any existing widgets
        while self.scrollAreaWidgetContents.layout().count():
            child = self.scrollAreaWidgetContents.layout().takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Create a QTextEdit to display HTML table content
        table_widget = QTextEdit()
        table_widget.setReadOnly(True)
        table_widget.setHtml(data.to_html(classes='results-table'))
        self.scrollAreaWidgetContents.layout().addWidget(table_widget)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

    def exportAsText(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Results As Text", "",
                                                  "Text Files (*.txt);;All Files (*)", options=options)
        if fileName:
            with open(fileName, 'w') as f:
                # Assuming df is your DataFrame with results
                f.write(self.df.to_string())

    def exportGraph(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Graph", "",
                                                  "PNG Files (*.png);;All Files (*)", options=options)
        if fileName:
            self.figure.savefig(fileName)

    def exportToCSV(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Results As CSV", "", "CSV Files (*.csv);;All Files (*)",
                                                  options=options)
        if fileName:
            self.df.to_csv(fileName, index=False)

    def exportToExcel(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Results As Excel", "",
                                                  "Excel Files (*.xlsx);;All Files (*)", options=options)
        if fileName:
            self.df.to_excel(fileName, index=False)

    def displaySummary(self, summary):
        """
        Displays the generated summary in the 'Text' tab.
        """
        # Clear any existing widgets in the layout
        layout = self.scrollAreaWidgetContents.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Create a new QLabel for the summary text
        summary_content = QLabel(summary)
        summary_content.setWordWrap(True)
        summary_content.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        # Add the summary content to the scrollAreaWidgetContents layout
        layout.addWidget(summary_content)

    def updateAndShow(self, data):
        """
        Update the ResultsWidget with the provided data and display it.
        """

        # Handle numerical results
        numerical_data = data.get('numerical_results', pd.DataFrame())
        if not numerical_data.empty:
            self.displayGraphResults(numerical_data)

        # Handle textual results
        textual_data = data.get('textual_results', [])
        if textual_data:  # assuming textual_data is a list of dictionaries
            self.displayTextResults(textual_data)

        # Show the widget and enable export buttons if there are results
        self.show()
        self.enableExportButtons(not numerical_data.empty or bool(textual_data))











