import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from functools import wraps
from logging import getLogger
from typing import Any

import aiofiles
import click
import marvin
import torch
from dotenv import load_dotenv
from sqlalchemy import delete, select, update, and_, func
from transformers import AutoModel, AutoTokenizer

from code_helper.code_fragment_extractor import (
    extract_code_fragments_from_file_content,
    extract_imports_from_file_content,
)
from code_helper.file_cache import CacheStrategy, file_cache
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
async def summarize_code(code_fragments: list[str]) -> list[str]:
    """
    Summarize the code fragments `code_fragments` to enhance a search pipeline,
    enabling developers to find relevant code by using natural language
    descriptions. The summary should clearly explain the specific purpose and
    functionality of the code, highlighting key actions, inputs, and outputs.
    Additionally, identify any major components related to the code, such as CLI
    commands, API endpoints, data models, validation schemas, or ORM interactions.

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

    **Example outputs:**

    - A module containing classes for an Event Log, Deck, Propulsion System, and
      Ship in a space travel simulation. The classes include methods for logging
      events, managing decks, firing thrusters, and controlling ships. Key features
      include handling events, calculating acceleration, managing fuel
      consumption, and simulating space travel dynamics.

    - A database model class for a user object with attributes for username, email,
      and password. The class includes methods for hashing passwords and validating
      user credentials.

    - A function for generating a random password reset token for a user.

    - A function representing a CLI command for listing all users in the database.
    """


@file_cache(cache_strategy=AI_FN_CACHE_STRATEGY)
@marvin.fn
async def extract_metadata(code_fragments: list[str]) -> list[dict[str, Any]]:
    """
    Extract metadata from the code fragments `code_fragments` to support natural
    language search. Return a list of dictionaries, each containing metadata for a
    code fragment.

    For each code fragment:
    - Identify if it is a function, class, or module.
        - If a fragment contains multiple functions and/or classes, it is a module.
    - If it represents a function:
        - Extract the function name and primary component type
        - Identify parameters, return types, and decorators
        - Note if it's async/sync and access level
        - List exceptions raised
    - If it represents a class:
        - Extract the class name and parent class
        - List major attributes and their types
        - Note if abstract/concrete
        - Identify implemented interfaces
        - Note design patterns used
    - If it represents a module:
        - Extract the primary component type
        - Include any relevant metadata for search
        - List functions and classes
        - Note module category and dependencies
        - Identify configuration constants

    *Exclude* metadata related to import statements.

    Return a JSON object with metadata classifications and values in string format.
    This JSON object should be valid and parsable by Python's `json` module.

    **Example output:**
    ```json
    {
        "type": "function",
        "name": "get_user",
        "component": "utilities",
        "is_async": false,
        "access": "public",
        "parameters": ["user_id: int", "include_deleted: bool = False"],
        "return_type": "User",
        "raises": ["NotFoundError", "ValidationError"],
        "decorators": ["@validate_input"]
    },
    {
        "type": "class",
        "name": "User",
        "parent_class": "BaseModel",
        "component": "data models",
        "attributes": {
            "id": "int",
            "username": "str",
            "email": "str"
        },
        "is_abstract": false,
        "interfaces": ["Serializable"],
        "design_pattern": "Active Record"
    },
    {
        "type": "module",
        "component": "API controllers",
        "classes": ["UserController"],
        "functions": ["validate_token"],
        "category": "web",
        "config_constants": ["MAX_USERS", "TOKEN_EXPIRY"],
        "dependencies": ["auth", "database"]
    }
    ```
    """


@file_cache
async def generate_embeddings(text: str):
    logger.info("Generating embeddings")
    model, tokenizer = await get_embedding_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].tolist()


@file_cache
async def clean_text(text):
    return text.replace("\x00", "")


async def batch_process_with_retries(
    llm_task: Any,
    texts: list[str],
    max_retries: int = 3,
    timeout: float = 30.0,
    models: list[str] = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"],
) -> list[Any]:
    """
    Execute an LLM task with batched inputs and dynamic batch size adjustment.
    Reduces batch size on failure until successful or single item processing.
    """
    original_model = marvin.settings.openai.chat.completions.model
    errors = []

    try:
        if not texts:
            return []

        batch_size = len(texts)
        while batch_size > 0:
            for model in models:
                marvin.settings.openai.chat.completions.model = model
                for attempt in range(max_retries):  # Add retry loop
                    try:
                        results = []
                        for i in range(0, len(texts), batch_size):
                            batch = texts[i : i + batch_size]
                            try:
                                async with asyncio.timeout(timeout):
                                    result = await llm_task(batch)
                                    results.extend(
                                        result if isinstance(result, list) else [result]
                                    )
                            except (asyncio.TimeoutError, Exception) as e:
                                error_msg = f"Attempt {attempt + 1}/{max_retries} failed with model {model} and batch size {batch_size}: {str(e)}"
                                logger.error(error_msg)
                                errors.append(error_msg)
                                raise
                        return results  # Success! Return results
                    except Exception as e:
                        if attempt == max_retries - 1:  # Last attempt for this model
                            if batch_size == 1:  # Try next model
                                logger.error(
                                    f"Last attempt failed with model {model} and batch size {batch_size}"
                                )
                                continue
                            # Reduce batch size and start over with first model
                            batch_size = max(batch_size // 2, 1)
                            break
                        # Not last attempt, continue with next retry
                        logger.error(
                            f"Attempt {attempt + 1}/{max_retries} failed with model {model} and batch size {batch_size}: {str(e)}"
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
        print(f"Error reading file: {filepath}")
        return None

    if not file_content:
        print(f"File is empty: {filepath}")
        return None

    query = await session.execute(
        select(Document.updated_at).where(Document.filepath == filepath)
    )
    result: datetime = query.scalar()

    if result is not None:
        if not replace_existing:
            print(f"File already exists: {filepath}")
            return None
        # Delete existing document and its fragments
        print(f"Deleting existing document: {filepath}")
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
            print(f"File is updated: {filepath}. Reindexing.")

    updated_at = datetime.fromtimestamp(os.path.getmtime(filepath))
    return file_content, updated_at


async def process_document_content(content: str) -> tuple[str, list, dict, str]:
    """Process the document content to generate embeddings and metadata."""
    cleaned_content = await clean_text(content)
    file_summary = await batch_process_with_retries(summarize_code, [cleaned_content])
    file_vector = await generate_embeddings(cleaned_content)

    metadata_raw = await batch_process_with_retries(extract_metadata, [cleaned_content])
    if not metadata_raw:
        logger.error("No metadata returned")
        file_metadata = {}
    else:
        try:
            file_metadata = json.loads(metadata_raw[0])
        except (ValueError, json.JSONDecodeError, TypeError) as e:
            logger.error(
                f"Error extracting metadata: {e}. Raw metadata: {metadata_raw}. Using empty metadata."
            )
            file_metadata = {}

    file_metadata["imports"] = extract_imports_from_file_content(cleaned_content)
    return cleaned_content, file_vector, file_metadata, file_summary[0]


async def process_fragments(content: str) -> tuple[list, list, list]:
    """Process code fragments to generate embeddings and metadata."""
    fragments = extract_code_fragments_from_file_content(content)

    async with asyncio.TaskGroup() as tg:
        fragment_vectors_tasks = [
            tg.create_task(generate_embeddings(frag)) for frag in fragments
        ]
        # Use batch processing for metadata
        fragment_metadata = await batch_process_with_retries(
            extract_metadata, fragments
        )

    fragment_vectors = [f.result() for f in fragment_vectors_tasks]

    # Parse the JSON metadata strings into dictionaries
    parsed_metadata = []
    for metadata_str in fragment_metadata:
        try:
            metadata = json.loads(metadata_str)
        except (ValueError, json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error parsing fragment metadata: {e}. Using empty metadata.")
            metadata = {}
        parsed_metadata.append(metadata)

    return fragments, fragment_vectors, parsed_metadata


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
    fragments, fragment_vectors, fragment_metadata = await process_fragments(
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
    for idx, (fragment, vector, metadata) in enumerate(
        zip(fragments, fragment_vectors, fragment_metadata)
    ):
        hierarchy_meta = {
            "parent": metadata.get("parent_class"),  # For methods within classes
            "siblings": [],  # Will contain other fragments at same level
            "index": idx,
        }

        await create_document_fragment(
            session,
            document_id=document.id,
            fragment_content=fragment,
            vector=vector,
            updated_at=updated_at,
            meta=metadata,
            hierarchy_meta=hierarchy_meta,
        )

    # Update sibling relationships for fragments
    fragments_query = await session.execute(
        select(DocumentFragment).where(DocumentFragment.document_id == document.id)
    )
    fragments_in_doc = (
        fragments_query.scalars().all()
    )  # Get actual DocumentFragment objects

    fragment_map = {}
    for frag in fragments_in_doc:
        parent = frag.meta.get("parent")  # Use meta instead of fragment_meta
        if parent not in fragment_map:
            fragment_map[parent] = []
        fragment_map[parent].append(frag.meta.get("name"))

    # Update each fragment with its siblings
    for frag in fragments_in_doc:
        parent = frag.meta.get("parent")
        siblings = fragment_map.get(parent, [])
        if not frag.meta:
            frag.meta = {}
        frag.meta["siblings"] = [s for s in siblings if s != frag.meta.get("name")]
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
@async_command
async def main(
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
