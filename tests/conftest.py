import pytest


@pytest.fixture(autouse=True)
def clear_lru_cache():
    yield
    try:
        from app.artifacts import get_artifacts
        get_artifacts.cache_clear()
    except Exception:
        pass