import pytest
import aiofiles
import os
import shutil
import hashlib
import pickle
from code_helper.file_cache import (
    file_cache,
    DEFAULT_CACHE_DIR,
    DEFAULT_STRATEGY,
    build_cache_key,
)

TEST_CACHE_DIR = "test_cache_dir"


@pytest.fixture(autouse=True)
def clean_cache_dir():
    if os.path.exists(TEST_CACHE_DIR):
        shutil.rmtree(TEST_CACHE_DIR)
    yield
    shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)


@pytest.mark.asyncio
async def test_file_cache_decorator():
    @file_cache(TEST_CACHE_DIR)
    async def sample_coro(x, y):
        return x + y

    result1 = await sample_coro(1, 2)
    assert result1 == 3

    result2 = await sample_coro(1, 2)
    assert result2 == 3

    assert os.path.exists(TEST_CACHE_DIR)

    key = build_cache_key(
        sample_coro._coro,
        args=(1, 2),
        kwargs={},
        strategy=DEFAULT_STRATEGY
    )
    key_hash = hashlib.md5(pickle.dumps(key)).hexdigest()
    cache_file = os.path.join(TEST_CACHE_DIR, key_hash + ".cache")

    assert os.path.exists(cache_file)


@pytest.mark.asyncio
async def test_file_cache_decorator_default_cache_dir():
    @file_cache
    async def sample_coro(x, y):
        return x + y

    result1 = await sample_coro(1, 2)
    assert result1 == 3

    result2 = await sample_coro(1, 2)
    assert result2 == 3

    assert os.path.exists(DEFAULT_CACHE_DIR)

    key = build_cache_key(
        sample_coro._coro,
        args=(1, 2),
        kwargs={},
        strategy=DEFAULT_STRATEGY
    )
    key_hash = hashlib.md5(pickle.dumps(key)).hexdigest()
    cache_file = os.path.join(DEFAULT_CACHE_DIR, key_hash + ".cache")

    assert os.path.exists(cache_file)

    async with aiofiles.open(cache_file, "rb") as f:
        content = await f.read()
        _, cached_result = pickle.loads(content)
        assert cached_result == 3

    os.unlink(cache_file)


@pytest.mark.asyncio
async def test_cache_with_kwargs():
    @file_cache(TEST_CACHE_DIR)
    async def sample_coro(x, y, z=1):
        return x + y + z

    result1 = await sample_coro(1, 2, z=3)
    assert result1 == 6

    result2 = await sample_coro(1, 2, z=3)
    assert result2 == 6

    assert os.path.exists(TEST_CACHE_DIR)

    key = build_cache_key(
        sample_coro._coro,
        args=(1, 2),
        kwargs={"z": 3},
        strategy=DEFAULT_STRATEGY
    )
    key_hash = hashlib.md5(pickle.dumps(key)).hexdigest()
    cache_file = os.path.join(TEST_CACHE_DIR, key_hash + ".cache")

    assert os.path.exists(cache_file)

    async with aiofiles.open(cache_file, "rb") as f:
        content = await f.read()
        _, cached_result = pickle.loads(content)
        assert cached_result == 6
