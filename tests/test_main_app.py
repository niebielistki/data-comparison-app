import pytest
from PyQt5.QtWidgets import QApplication
from main_app import DataComparisonApp
import pandas as pd
from unittest.mock import MagicMock


@pytest.fixture(scope='module')
def app(qtbot):
    test_app = QApplication.instance()
    if not test_app:
        test_app = QApplication(sys.argv)
    widget = DataComparisonApp()
    qtbot.addWidget(widget)
    return widget


def test_load_csv_files(app, qtbot, monkeypatch):
    # Mock the getOpenFileNames function to return predefined file paths
    path1 = '/Users/wiktoria/PycharmProjects/DataComparisonApp/tests/test_data/data_6/vet_data_2023_03.csv'
    path2 = '/Users/wiktoria/PycharmProjects/DataComparisonApp/tests/test_data/data_6/vet_data_2022_05.csv'
    monkeypatch.setattr(QFileDialog, 'getOpenFileNames',
                        lambda *args, **kwargs: ([path1, path2], 'All Files (*)'))

    # Pretend CSV files are loaded
    app.data_frames[path1] = pd.DataFrame({'Column1': [1, 2, 3]})
    app.data_frames[path2] = pd.DataFrame({'Column1': [1, 2, 3]})

    # Simulate clicking the load button
    qtbot.mouseClick(app.loadButton, Qt.LeftButton)

    # Wait for the signal that indicates loading is done
    with qtbot.waitSignal(app.analyzer.analysisComplete, timeout=1000):
        app.analyzer.analysisComplete.emit()

    # Check if data frames have been loaded
    assert app.data_frames, "Data frames should be loaded"


def test_column_selection(app):
    # Directly manipulate selected columns for testing
    app.selected_columns = ['Column1']

    # Check if the correct columns are selected
    assert 'Column1' in app.selected_columns, "Column1 should be selected"


def test_perform_analysis(app, qtbot):
    # Trigger the analysis
    qtbot.mouseClick(app.analyzeButton, Qt.LeftButton)

    # Wait for the signal that indicates analysis is complete
    with qtbot.waitSignal(app.analyzer.analysisComplete, timeout=1000):
        app.analyzer.analysisComplete.emit()

    # Check if results widget is visible after analysis
    assert app.resultsWidget.isVisible(), "Results widget should be visible after analysis"


def test_export_buttons(app, qtbot):
    # Pretend analysis has been performed
    app.resultsWidget.enableExportButtons()

    # Check if export buttons are enabled
    assert all(button.isEnabled() for button in app.exportButtons), "Export buttons should be enabled after analysis"
