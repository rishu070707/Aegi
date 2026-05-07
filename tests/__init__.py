"""
tests/ -- Unit and integration tests for Aegi weapon detection system.

This directory contains test suites for post-processing pipeline components.

Running tests:
    python -m pytest tests/ -v                # Run all tests
    python -m pytest tests/test_post_processing.py -v  # Run specific test file
    python -m pytest tests/test_post_processing.py::TestTemporalConsistencyFilter -v  # Run specific test class

Installing test dependencies:
    pip install pytest pytest-cov
"""
