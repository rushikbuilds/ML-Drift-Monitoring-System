"""Dashboard data helpers and the (optional, legacy) Python client for the drift monitoring API."""
try:
    from .client import ApiClient
except ImportError:  # httpx lives in requirements-legacy.txt; the Streamlit dashboard does not need it
    ApiClient = None

__all__ = ["ApiClient"]
