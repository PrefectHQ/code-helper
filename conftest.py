import pytest
from prefect.testing.utilities import prefect_test_harness

pytest_plugins = ["pytest_asyncio"]


"""
@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield
"""
