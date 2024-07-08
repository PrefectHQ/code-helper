import asyncio
from datetime import datetime
import json
from functools import wraps
from logging import getLogger
import os
import sys
from typing import Any

import aiofiles
import click
from dotenv import load_dotenv
import marvin
from sqlalchemy import select, delete

from code_helper.code_fragment_extractor import (
    extract_code_fragments_from_file_content,
    extract_imports_from_file_content,
)
from code_helper.models import (
    Document,
    DocumentFragment,
    get_session,
    create_document,
    create_document_fragment,
)
from code_helper.file_cache import file_cache, CacheStrategy

load_dotenv()

IGNORED_PATHS = [".git", "env", "venv", "__pycache__"]

logger = getLogger(__name__)

instructor_lock = asyncio.Lock()
_model = None

marvin.settings.openai.chat.completions.model = "gpt-3.5-turbo"

AI_FN_CACHE_STRATEGY = (
    CacheStrategy.name
    | CacheStrategy.args
    | CacheStrategy.kwargs
    | CacheStrategy.docstring
)


async def get_embedding_model():
    global _model
    if _model is None:
        async with instructor_lock:
            from InstructorEmbedding import INSTRUCTOR

            _model = INSTRUCTOR("hkunlp/instructor-xl")
    return _model


# @task(
#     persist_result=True,
#     cache_key_fn=task_input_hash,
#     cache_expiration=timedelta(days=7),
#     tags=["openai"],
# )
@file_cache(cache_strategy=AI_FN_CACHE_STRATEGY)
@marvin.fn
async def summarize_code(code: str) -> str:
    """
    You are preparing summarizations of code fragments to help in a search
    pipeline. Summaries will help developers use natural language to describe
    what they want. Developers may use general technical terms like "orm" or
    "model" or the specific names of functions and classes in the codebase. Or
    they may also ask questions about the code, either by naming classes or
    functions or by providing example code. You need to summarize the code by
    referring to what it does specifically and then inferring major components
    it relates to, such as CLI commands, API endpoints, data models, or
    validation schemas.
    """


# @task(
#     persist_result=True,
#     cache_key_fn=task_input_hash,
#     cache_expiration=timedelta(days=7),
#     tags=["openai"],
# )
@file_cache(cache_strategy=AI_FN_CACHE_STRATEGY)
@marvin.fn
async def extract_metadata(code: str) -> str:
    """
    Extra metadata from code fragments to help with natural language search.

    If the code you extract metadata from is one function or class, extract
    metadata such as whether the code is a function or class. Extract the module
    name. If the code is a class, extract the class name and the parent class.
    Extract the class's major component (CLI commands, API endpoints, data
    models, or schemas, utilities, tests, etc.). If the code appears to be an
    entire file, extract metadata such as the major components it relates to,
    such as CLI commands or database models, and any other metadata relevant to
    the file as a whole that would be useful to natural language search. Include
    the names of functions and classes.

    DO NOT extract metadata describing import statements.

    Return a JSON object of metadata classifications to extracted values inside
    a string.

    Examples:
        {
            "type": "function",
            "module": "utils",
            "name": "get_user",
            "component": "utilities"
        }

        {
            "type": "class",
            "module": "prefect.client.schemas.users",
            "name": "User",
            "parent_class": "BaseModel",
            "component": "validation schemas"
        }

        {
            "type": "class",
            "module": "prefect_cloud.models.orm",
            "name": "User",
            "parent_class": "Base",
            "component": "database models"
        }

        {
            "type": "function",
            "module": "prefect_cloud.models.actors",
            "name": "read_actor",
            "component": "database models"
        }

        {
            "type": "file",
            "component": "CLI commands"
        }
    """


# @task(
#     persist_result=True,
#     cache_key_fn=task_input_hash,
#     cache_expiration=timedelta(days=7),
#     tags=["model"],
# )
@file_cache
async def embed_text_with_instructor(text: str):
    instruction = "Represent the code snippet:"
    model = await get_embedding_model()
    embeddings = model.encode([[instruction, text]])
    return embeddings[0].tolist()


# @task(
#     persist_result=True,
#     cache_key_fn=task_input_hash,
#     cache_expiration=timedelta(days=7),
# )
@file_cache
async def clean_text(text):
    return text.replace("\x00", "")


async def with_retries(llm_task: Any, text: str) -> Any:
    """
    Send text to an LLM task and use various attempts to retry if it fails.

    TODO: Should probably just be a decorator.
    TODO: One finalizer.
    """
    original_model = marvin.settings.openai.chat.completions.model

    # Exceptions from OpenAI are always openai.BadResponseError, but for some
    # reason, trying to catch that doesn't work reliably. It's possible that
    # multiple copies of the exception class are in memory due to lazy loading.
    try:
        return await llm_task(text)
    except:
        logger.exception(f"Error running {llm_task}. Retrying with gpt-4o.")
    finally:
        marvin.settings.openai.chat.completions.model = original_model

    try:
        marvin.settings.openai.chat.completions.model = "gpt-4o"
        return await llm_task(text)
    except:
        logger.exception(
            f"Error running {llm_task} (attempt #2). Retrying with less context."
        )
    finally:
        marvin.settings.openai.chat.completions.model = original_model

    try:
        marvin.settings.openai.chat.completions.model = "gpt-4o"
        return await llm_task(text[: len(text) // 2])
    except:
        logger.exception(
            f"Error running {llm_task} (attempt #3). Retrying with less context."
        )
    finally:
        marvin.settings.openai.chat.completions.model = original_model

    try:
        marvin.settings.openai.chat.completions.model = "gpt-4o"
        return await llm_task(text[: len(text) // 4])
    except:
        logger.exception(f"Error running {llm_task} (attempt #4). Giving up.")
        return None
    finally:
        marvin.settings.openai.chat.completions.model = original_model


# @task(
#     persist_result=True,
#     cache_key_fn=task_input_hash,
#     cache_expiration=timedelta(days=7),
#     refresh_cache=True,
#     tags=["file"],
# )
async def process_file(
    filepath, session, replace_existing=False, reindex_updated=False
) -> bool:
    # with transaction(commit_mode=CommitMode.EAGER):
    try:
        async with aiofiles.open(filepath, "r") as f:
            file_content = await f.read()
    except UnicodeDecodeError:
        print(f"Error reading file: {filepath}")
        return False

    # stmt = (
    #     select(func.count()).select_from(Document).where(Document.filepath == filepath)
    # )
    query = await session.execute(
        select(Document.updated_at).where(Document.filepath == filepath)
    )
    result: datetime = query.scalar()

    if result is not None:
        if not replace_existing:
            print(f"File already exists: {filepath}")
            return False
        if reindex_updated:
            updated_at = datetime.fromtimestamp(os.path.getmtime(filepath))
            if updated_at <= result:
                # print(f"File is not updated: {filepath}")
                return False
            print(f"File is updated: {filepath}. Reindexing.")
        await session.execute(
            delete(DocumentFragment).where(
                DocumentFragment.document.has(filepath=filepath)
            )
        )
        await session.execute(delete(Document).where(Document.filepath == filepath))
    else:
        print(f"File does not exist in the database. Indexing: {filepath}")
        return False

    cleaned_content = await clean_text(file_content)

    # Summarize the file as a whole for TF-IDF/keyword search
    file_summary = await with_retries(summarize_code, cleaned_content)

    # 1. Create embeddings for the file as a whole.
    file_vector = await embed_text_with_instructor(cleaned_content)

    # 2. Create embeddings for each code fragment.
    fragments = extract_code_fragments_from_file_content(cleaned_content)

    async with asyncio.TaskGroup() as tg:
        fragment_vectors_tasks = [
            tg.create_task(embed_text_with_instructor(frag)) for frag in fragments
        ]
    fragment_vectors = [f.result() for f in fragment_vectors_tasks]

    updated_at = datetime.fromtimestamp(os.path.getmtime(filepath))
    metadata_raw = await with_retries(extract_metadata, cleaned_content)

    try:
        file_metadata = json.loads(metadata_raw)
    except (ValueError, json.JSONDecodeError, TypeError) as e:
        logger.error(
            f"Error extracting metadata: {e}. Raw metadata: {metadata_raw}. Using empty metadata."
        )
        file_metadata = {}

    file_metadata["imports"] = extract_imports_from_file_content(cleaned_content)

    async with asyncio.TaskGroup() as tg:
        fragment_metadata_tasks = [
            tg.create_task(with_retries(extract_metadata, frag)) for frag in fragments
        ]
    fragment_metadata = [f.result() for f in fragment_metadata_tasks]

    document = await create_document(
        session,
        filename=os.path.basename(filepath),
        filepath=filepath,
        file_content=cleaned_content,
        vector=file_vector,
        summary=file_summary,
        meta=file_metadata,
        updated_at=updated_at,
    )

    for fragment, vector, metadata in zip(
        fragments, fragment_vectors, fragment_metadata
    ):
        await create_document_fragment(
            session,
            document_id=document.id,
            fragment_content=fragment,
            vector=vector,
            updated_at=updated_at,
            meta=metadata,
        )

    await session.commit()

    print(f"Processed file: {filepath}")

    return True


# async def reset_concurrency_limits():
#     """
#     This is a hack because concurrency limits aren't releasing when tasks
#     crash/fail.
#     """
#     async with get_client() as client:
#         await client.update_global_concurrency_limit(
#             "openai", GlobalConcurrencyLimitUpdate(active_slots=0)
#         )
#         await client.update_global_concurrency_limit(
#             "model", GlobalConcurrencyLimitUpdate(active_slots=0)
#         )
#         await client.update_global_concurrency_limit(
#             "file", GlobalConcurrencyLimitUpdate(active_slots=0)
#         )


def async_command(f):
    """Wrapper necessary because Click doesn't support async"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


# @flow(persist_result=True)
@click.command()
@click.argument("code_dirs", nargs=-1)
@click.option(
    "--reindex",
    is_flag=True,
    help="Index files even if they already exist in the database",
)
@click.option(
    "--reindex-updated",
    is_flag=True,
    help="Index files if the version on disk is newer than the version in the database",
)
@click.option(
    "--limit", help="Limit reindexing to a specific number of files", type=int
)
@async_command
async def process_files(
    code_dirs: list[str],
    reindex: bool = False,
    reindex_updated: bool = False,
    limit: int = -1,
):
    """
    Process all Python files in the given directories and insert/update the
    embeddings in the database.
    """
    filepaths = set()
    processed = 0

    async with get_session() as session:
        if not code_dirs:
            print("At least one directory or file path must be provided.")
            sys.exit(1)

        # await reset_concurrency_limits()

        for code_dir in code_dirs:
            if not os.path.exists(code_dir):
                print(f"Path does not exist: {code_dir}")
                sys.exit(1)

            if os.path.isfile(code_dir):
                filepaths.add(code_dir)
            elif os.path.isdir(code_dir):
                filepaths = set()
                for root, dirs, files in os.walk(code_dir):
                    dirs[:] = [d for d in dirs if d not in IGNORED_PATHS]
                    for file in files:
                        if file.endswith(".py") and file != "__init__.py":
                            filepath = os.path.join(root, file)
                            filepaths.add(filepath)
            for filepath in filepaths:
                indexed = await process_file(
                    filepath, session, reindex, reindex_updated
                )
                if not indexed:
                    continue
                processed += 1
                if 0 < limit <= processed:
                    print("Limit reached")
                    return
            print("Code files processed and inserted/updated successfully.")
        else:
            print(f"Invalid path: {code_dir}")
            sys.exit(1)


if __name__ == "__main__":
    process_files()
