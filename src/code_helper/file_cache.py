import enum
import functools
import hashlib
import os
import pickle
import shutil
import time

import aiofiles

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


class CacheStrategy(enum.Flag):
    name = enum.auto()
    args = enum.auto()
    kwargs = enum.auto()
    code = enum.auto()
    docstring = enum.auto()


DEFAULT_STRATEGY = (
    CacheStrategy.name | CacheStrategy.args | CacheStrategy.kwargs | CacheStrategy.code
)


def build_cache_key(coro, args, kwargs, strategy):
    key = []

    if CacheStrategy.name in strategy:
        key.append(coro.__name__)

    if CacheStrategy.args in strategy:
        key.extend(args)

    if CacheStrategy.kwargs in strategy:
        key.extend(tuple(sorted(kwargs.items())))

    if CacheStrategy.code in strategy:
        key.append(coro.__code__.co_code)

    if CacheStrategy.docstring in strategy:
        key.append(coro.__doc__)

    return key


def file_cache(
    cache_dir=DEFAULT_CACHE_DIR,
    file_extension=".cache",
    cache_strategy=DEFAULT_STRATEGY,
    ttl=None,
):
    # If the cache_dir is a callable, then file_cache is being used as a
    # decorator without arguments.
    if callable(cache_dir):
        return file_cache()(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    def decorator(coro):
        @functools.wraps(coro)
        async def wrapper(*args, **kwargs):

            key = build_cache_key(coro, args, kwargs, cache_strategy)
            key_hash = hashlib.md5(pickle.dumps(key)).hexdigest()
            cache_file = os.path.join(cache_dir, key_hash + file_extension)

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


def bust_file_cache(
    cache_dir=DEFAULT_CACHE_DIR,
):
    shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
