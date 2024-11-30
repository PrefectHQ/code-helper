import os
from prefect.testing.utilities import prefect_test_harness
import pytest_asyncio

from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy import pool

from code_helper.models import async_drop_db

pytest_plugins = ["pytest_asyncio"]

# Only necessary when we're testing Prefect flows, tasks, etc.
"""
@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        yield
"""


@pytest_asyncio.fixture
async def SessionLocal():
    """Create a test database and handle setup/teardown"""
    # Test database configuration
    TEST_DB_URL = (
        "postgresql+asyncpg://code_helper:help-me-code@localhost:5432/code_helper_test"
    )

    # Set environment variable for test database
    os.environ["DATABASE_URL"] = TEST_DB_URL

    # Import models after setting DATABASE_URL
    from code_helper.models import init_db

    ECHO_SQL_QUERIES = os.getenv("CODE_HELPER_ECHO_SQL_QUERIES", "false").lower() == "true"
    test_engine = create_async_engine(
        TEST_DB_URL,
        echo=ECHO_SQL_QUERIES,
        poolclass=pool.NullPool  # Prevent connection pool issues
    )

    await init_db(test_engine)

    SessionLocal = sessionmaker(
        bind=test_engine, expire_on_commit=False, class_=AsyncSession
    )

    try:
        yield SessionLocal
    finally:
        await async_drop_db(test_engine)
        await test_engine.dispose()


@pytest_asyncio.fixture
async def db_session(SessionLocal):
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
