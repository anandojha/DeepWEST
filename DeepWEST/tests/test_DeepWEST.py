"""
Unit and regression test for the DeepWEST package.
"""

# Import package, test suite, and other packages as needed
import DeepWEST
import pytest
import sys

def test_DeepWEST_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "DeepWEST" in sys.modules
