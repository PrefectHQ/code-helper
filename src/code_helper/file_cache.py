import functools
import hashlib
import os
import pickle
import time

import aiofiles

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def file_cache(cache_dir=DEFAULT_CACHE_DIR, ttl=None):
    if callable(cache_dir):
        return file_cache(DEFAULT_CACHE_DIR, ttl)(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    def decorator(coro):
        @functools.wraps(coro)
        async def wrapper(*args, **kwargs):
            func_code = coro.__code__.co_code
            key = (func_code, args, tuple(sorted(kwargs.items())))
            key_hash = hashlib.md5(pickle.dumps(key)).hexdigest()
            cache_file = os.path.join(cache_dir, key_hash + ".cache")

            try:
                async with aiofiles.open(cache_file, "rb") as f:
                    content = await f.read()
                    cache_entry = pickle.loads(content)
                    cache_time, result = cache_entry

                    # Check if the cached entry has expired
                    if ttl is not None and (time.time() - cache_time) > ttl:
                        raise FileNotFoundError
                    return result
            except (FileNotFoundError, EOFError):
                result = await coro(*args, **kwargs)
                cache_entry = (time.time(), result)
                async with aiofiles.open(cache_file, "wb") as f:
                    pickled_cache = pickle.dumps(cache_entry)
                    await f.write(pickled_cache)
                return result

        wrapper._coro = coro

        return wrapper

    return decorator
