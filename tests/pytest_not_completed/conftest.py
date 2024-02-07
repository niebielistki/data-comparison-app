# conftest.py
import pytest
import os

@pytest.fixture(scope='session')
def base_test_data_dir():
    return os.path.join(os.path.dirname(__file__), 'data')

@pytest.fixture(params=['data_1', 'data_2', 'data_3', 'data_4', 'data_5'])
def data_directory(base_test_data_dir, request):
    return os.path.join(base_test_data_dir, request.param)
