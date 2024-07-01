import functools
import pickle

import aiofiles


async def file_cache(cache_file):
    try:
        async with aiofiles.open(cache_file, "rb") as f:
            content = await f.read()
            cache = pickle.loads(content)
    except (FileNotFoundError, EOFError):
        cache = {}

    async def decorator(coro):
        @functools.wraps(coro)
        async def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                return cache[key]
            else:
                result = await coro(*args, **kwargs)
                cache[key] = result
                async with open(cache_file, "wb") as f:
                    pickled_cache = pickle.dumps(cache)
                    await f.write(pickled_cache)
                return result

        return wrapper

    return decorator
