"""Configuration and fixtures for pytest."""
import os
import sys
from pathlib import Path
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Fixtures can be added here to be available across test modules

@pytest.fixture(scope="session")
def project_root():
    """Return the absolute path to the project root directory."""
    return Path(__file__).parent.absolute()

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    test_data = Path(__file__).parent / 'test_data'
    test_data.mkdir(exist_ok=True)
    return test_data
