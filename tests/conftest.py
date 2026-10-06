"""
Pytest configuration and shared fixtures for the test suite.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def model():
    from churn.model import train

    return train()
