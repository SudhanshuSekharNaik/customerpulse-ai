"""Pytest configuration and sys.path setup."""

import os
import sys

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
