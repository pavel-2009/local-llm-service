import pytest
from fastapi.testclient import TestClient

from src.local_llm_service.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
