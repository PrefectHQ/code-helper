import asyncio
import logging
import os
import sys
from datetime import datetime
from functools import wraps
from logging import getLogger
from typing import Any, Coroutine

import aiofiles
import click
import marvin
import torch
from dotenv import load_dotenv
from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.orm.attributes import flag_modified
from transformers import AutoModel, AutoTokenizer

from code_helper.code_fragment_extractor import (
    extract_code_fragments_from_file_content,
    extract_metadata_from_fragment,
    extract_metadata_from_node,
)
from code_helper.file_cache import CacheStrategy, bust_file_cache, file_cache
from code_helper.models import (
    Document,
    DocumentFragment,
    create_document,
    create_document_fragment,
    get_session,
)

load_dotenv()

IGNORED_PATHS = [".git", "env", "venv", "__pycache__"]

logger = getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("Starting create_embeddings_async")

instructor_lock = asyncio.Lock()
_model = None
_tokenizer = None

marvin.settings.openai.chat.completions.model = "gpt-3.5-turbo"

AI_FN_CACHE_STRATEGY = (
    CacheStrategy.name
    | CacheStrategy.args
    | CacheStrategy.kwargs
    | CacheStrategy.docstring
)


async def get_embedding_model():
    global _model, _tokenizer
    if _model is None:
        async with instructor_lock:
            # Load CodeBERT model and tokenizer
            _tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            _model = AutoModel.from_pretrained("microsoft/codebert-base")
    return _model, _tokenizer


@file_cache(cache_strategy=AI_FN_CACHE_STRATEGY)
@marvin.fn
async def summarize_code(code_fragment: str) -> str:
    """
    Summarize code to enhance a search pipeline,
    enabling developers to find relevant code by using natural language descriptions.
    The summary should clearly explain the specific purpose and functionality of the
    code, highlighting key actions, inputs, and outputs. Additionally, identify any
    major components related to the code, such as CLI commands, API endpoints, data
    models, validation schemas, or ORM interactions.

    When summarizing:
    - Use general technical terms (e.g., "ORM," "validation schema") as appropriate.
    - Refer to specific classes, functions, or example code if they are central to
      the fragment.
    - Aim for clarity and brevity, ensuring each summary is actionable and easy
      to interpret in search results.
    - Do not start the summary with "Summary" or "Summarize." Instead, jump straight
      into the summary.

    Remember that developers may search using various terms, so the summary should
    anticipate and cover potential queries based on functionality and component
    types.

    Return a summary.

    **Example outputs:**
    -   "A module containing classes for an Event Log, Deck, Propulsion System, and
        Ship in a space travel simulation. The classes include methods for logging
       events, managing decks, firing thrusters, and controlling ships. Key features
        include handling events, calculating acceleration, managing fuel
        consumption, and simulating space travel dynamics."

    -   "A database model class for a user object with attributes for username, email,
        and password. The class includes methods for hashing passwords and validating
        user credentials."

    -   "A function for generating a random password reset token for a user."

    -   "A function representing a CLI command for listing all users in the database."
    """


@file_cache
async def generate_embeddings(text: str):
    logger.info("Generating embeddings")
    model, tokenizer = await get_embedding_model()

    # Change max_length to 512 (CodeBERT's maximum) and add proper padding
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,  # Changed from 768 to 512
        padding=True,  # Changed from 'max_length' to True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].tolist()


@file_cache
async def clean_text(text):
    return text.replace("\x00", "")


async def run_with_retries(
    llm_task: Coroutine,
    inputs: list[Any],
    max_retries: int = 3,
    timeout: float = 30.0,
    models: list[str] = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"],
    initial_concurrency: int = 10,
) -> list[Any]:
    """
    Execute an LLM task with multiple inputs and dynamic concurrency control.
    Reduces concurrency limit on failure until successful or falls back to
    synchronous processing.
    """
    original_model = marvin.settings.openai.chat.completions.model
    errors = []

    if not inputs:
        return []

    try:
        concurrency = initial_concurrency
        while concurrency > 0:
            for model in models:
                marvin.settings.openai.chat.completions.model = model
                for attempt in range(max_retries):
                    try:
                        results = [None] * len(inputs)
                        sem = asyncio.Semaphore(concurrency)

                        async def process_item(index: int, text: str):
                            async with sem:
                                try:
                                    async with asyncio.timeout(timeout):
                                        result = await llm_task(text)
                                        results[index] = result[0] if isinstance(result, list) else result
                                except (asyncio.TimeoutError, Exception) as e:
                                    error_msg = f"Failed processing item {index} with model {model}: {str(e)}"
                                    logger.error(error_msg)
                                    errors.append(error_msg)
                                    raise

                        # Process all texts concurrently with semaphore control
                        async with asyncio.TaskGroup() as tg:
                            for i, text in enumerate(inputs):
                                tg.create_task(process_item(i, text))

                        # If we get here, all tasks completed successfully
                        return [r for r in results if r is not None]

                    except Exception as e:
                        if attempt == max_retries - 1:  # Last attempt for this model
                            if concurrency == 1:  # Try next model
                                logger.error(
                                    f"Last attempt failed with model {model} and concurrency {concurrency}"
                                )
                                continue
                            # Reduce concurrency and start over with first model
                            concurrency = max(concurrency // 2, 1)
                            break

                        logger.error(
                            f"Attempt {attempt + 1}/{max_retries} failed with model {model} and concurrency {concurrency}: {str(e)}"
                        )
                        await asyncio.sleep(1)  # Add backoff delay
                        continue

        if not errors:
            errors.append("Unknown error occurred during processing")

        raise Exception(f"All retry attempts failed:\n{'\n'.join(errors)}")

    finally:
        marvin.settings.openai.chat.completions.model = original_model


async def read_and_validate_file(
    filepath: str, session, replace_existing: bool, reindex_updated: bool
) -> tuple[str, datetime] | None:
    """Handle file reading and validation of processing conditions."""
    try:
        async with aiofiles.open(filepath, "r") as f:
            file_content = await f.read()
    except UnicodeDecodeError:
        logger.error(f"Error reading file: {filepath}")
        return None

    if not file_content:
        logger.error(f"File is empty: {filepath}")
        return None

    query = await session.execute(
        select(Document.updated_at).where(Document.filepath == filepath)
    )
    result: datetime = query.scalar()
    print(f"Result: {result}")

    if result is not None:
        if not replace_existing:
            logger.error(f"File already indexed in database and not replacing: {filepath}")
            return None
        # Delete existing document and its fragments
        logger.info(f"Deleting existing document: {filepath}")
        await session.execute(
            delete(DocumentFragment).where(
                DocumentFragment.document.has(filepath=filepath)
            )
        )
        await session.execute(delete(Document).where(Document.filepath == filepath))
        await session.commit()
        if reindex_updated:
            updated_at = datetime.fromtimestamp(os.path.getmtime(filepath))
            if updated_at <= result:
                return None
            logger.info(f"File is updated: {filepath}. Reindexing.")

    updated_at = datetime.fromtimestamp(os.path.getmtime(filepath))
    return file_content, updated_at


async def process_document_content(content: str) -> tuple[str, list, dict, str]:
    """Process the document content to generate embeddings and metadata."""
    cleaned_content = await clean_text(content)
    file_summary = await run_with_retries(summarize_code, [cleaned_content])
    file_vector = await generate_embeddings(file_summary[0])
    file_metadata = extract_metadata_from_fragment(cleaned_content)
    return cleaned_content, file_vector, file_metadata, file_summary[0]


async def process_fragments(
    content: str,
) -> tuple[list[str], list[str], list[list[float]], list[dict]]:
    """Process code fragments to generate embeddings and metadata."""
    fragment_tuples = extract_code_fragments_from_file_content(content)
    nodes, fragments = zip(*fragment_tuples) if fragment_tuples else ([], [])

    # Generate summaries for fragments
    fragment_summaries = await run_with_retries(summarize_code, fragments)

    # Generate embeddings from summaries instead of raw fragments
    async with asyncio.TaskGroup() as tg:
        fragment_vectors_tasks = [
            tg.create_task(generate_embeddings(summary))
            for summary in fragment_summaries
        ]
    fragment_vectors = [f.result() for f in fragment_vectors_tasks]

    fragment_metadata = [extract_metadata_from_node(node) for node in nodes]

    return fragments, fragment_summaries, fragment_vectors, fragment_metadata


async def process_file(
    filepath: str, session, replace_existing=False, reindex_updated=False
) -> bool:
    """Main processing function that coordinates the stages."""
    # Stage 1: File Reading and Validation
    result = await read_and_validate_file(
        filepath, session, replace_existing, reindex_updated
    )
    if result is None:
        return False
    file_content, updated_at = result

    # Stage 2: Document Processing
    (
        cleaned_content,
        file_vector,
        file_metadata,
        file_summary,
    ) = await process_document_content(file_content)

    # Calculate path information
    path_parts = filepath.split(os.sep)
    if (
        path_parts[0] == ""
    ):  # Remove empty string at start if path begins with separator
        path_parts = path_parts[1:]
    filename = path_parts[-1]
    parent_path = os.sep.join(path_parts[:-1])

    # Build hierarchy metadata
    hierarchy_meta = {"children": [filename]}

    # Stage 3: Fragment Processing
    fragments, summaries, fragment_vectors, fragment_metadata = await process_fragments(
        cleaned_content
    )

    # Stage 4: Database Operations
    document = await create_document(
        session,
        filename=filename,
        filepath=filepath,
        path_array=path_parts,
        vector=file_vector,
        summary=file_summary,
        meta=file_metadata,
        hierarchy_meta=hierarchy_meta,
        updated_at=updated_at,
    )

    # Update parent's children list if parent exists
    if parent_path:
        parent_doc = await session.execute(
            select(Document).where(Document.filepath == parent_path)
        )
        parent_doc = parent_doc.scalar_one_or_none()
        if parent_doc:
            current_meta = parent_doc.hierarchy_meta or {"children": []}
            if "children" not in current_meta:
                current_meta["children"] = []
            if filename not in current_meta["children"]:
                current_meta["children"].append(filename)

            await session.execute(
                update(Document)
                .where(Document.filepath == parent_path)
                .values(hierarchy_meta=current_meta)
            )

    # Process fragments with hierarchy information
    for idx, (fragment, summary, vector, metadata) in enumerate(
        zip(fragments, summaries, fragment_vectors, fragment_metadata)
    ):
        # For methods, ensure parent_classes is set
        if metadata.get("type") == "function" and metadata.get("parent"):
            if "parent_classes" not in metadata:
                metadata["parent_classes"] = [metadata["parent"]]

        hierarchy_meta = {
            "parent": metadata.get("parent"),  # For methods within classes
            "siblings": [],  # Will contain other fragments at same level
            "index": idx,
        }

        await create_document_fragment(
            session,
            document_id=document.id,
            fragment_content=fragment,
            summary=summary,
            vector=vector,
            meta=metadata,
            hierarchy_meta=hierarchy_meta,
            updated_at=updated_at,
        )

    # Update sibling relationships for fragments
    fragments_query = await session.execute(
        select(DocumentFragment).where(DocumentFragment.document_id == document.id)
    )
    fragments_in_doc = fragments_query.scalars().all()

    fragment_map = {}
    for frag in fragments_in_doc:
        parent = frag.meta.get("parent")  # Use meta instead of fragment_meta
        if parent not in fragment_map:
            fragment_map[parent] = []
        name = frag.meta.get("name")
        fragment_map[parent].append(name)
        print(f"Adding {name} to parent {parent}")

    # Update each fragment with its siblings
    for frag in fragments_in_doc:
        parent = frag.meta.get("parent")
        siblings = fragment_map.get(parent, [])
        if not frag.meta:
            frag.meta = {}
        frag.meta["siblings"] = [s for s in siblings if s != frag.meta.get("name")]
        flag_modified(frag, "meta")
        session.add(frag)

    await session.commit()
    print(f"Processed file: {filepath}")
    return True


async def get_directory_contents(session, directory_path: str):
    """Get all files and subdirectories in a directory."""
    path_parts = directory_path.split(os.sep)
    return await session.execute(
        select(Document).where(
            and_(
                func.array_length(Document.path_array, 1) == len(path_parts) + 1,
                Document.path_array[: len(path_parts)].contains(path_parts),
            )
        )
    )


async def get_file_siblings(session, filepath: str):
    """Get all files in the same directory."""
    doc = await session.execute(select(Document).where(Document.filepath == filepath))
    if doc is None:
        return []

    parent_path = doc.hierarchy_meta["parent"]
    if not parent_path:
        return []

    return await session.execute(
        select(Document).where(Document.hierarchy_meta["parent"] == parent_path)
    )


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
    "--limit",
    help="Limit reindexing to a specific number of files",
    type=int,
    default=-1,
)
@click.option(
    "--bust-cache",
    is_flag=True,
    help="Bust the cache for all functions",
)
@async_command
async def main(
    code_dirs: list[str],
    reindex: bool = False,
    reindex_updated: bool = False,
    limit: int = -1,
    bust_cache: bool = False,
):
    """
    Process all Python files in the given directories and insert/update the
    embeddings in the database.
    """
    filepaths = set()
    processed = 0

    if bust_cache:
        bust_file_cache()

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
                        if file.endswith(".py"):
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
