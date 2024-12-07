import ast
import asyncio
from dataclasses import dataclass
import logging
import os
import sys
from datetime import UTC, datetime
from functools import wraps
from logging import getLogger
from typing import Any, Coroutine, Dict, Optional

import aiofiles
import click
import marvin
from prefect import flow, task
import tiktoken
import torch
from dotenv import load_dotenv
from sqlalchemy import and_, delete, func, select
from transformers import AutoModel, AutoTokenizer
from openai import RateLimitError, BadRequestError
from prefect.cache_policies import TASK_SOURCE, INPUTS, RUN_ID, CachePolicy
from prefect.context import TaskRunContext


from code_helper.code_fragment_extractor import (
    extract_code_fragments_from_file_content,
    extract_metadata_from_fragment,
    extract_metadata_from_node,
)
from code_helper.file_cache import CacheStrategy, bust_file_cache
from code_helper.models import (
    Document,
    DocumentFragment,
    create_document,
    create_document_fragment,
    get_session,
)

load_dotenv()

# Default paths to ignore
DEFAULT_IGNORED_PATHS = [".git", "env", "venv", "__pycache__"]
DEFAULT_FILE_EXTENSIONS = [".py"]

logger = getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info("Starting create_embeddings_async")

embedding_model_lock = asyncio.Lock()
_model = None
_tokenizer = None

marvin.settings.openai.chat.completions.model = "gpt-4o-mini"

AI_FN_CACHE_STRATEGY = (
    CacheStrategy.name
    | CacheStrategy.args
    | CacheStrategy.kwargs
    | CacheStrategy.docstring
)


@dataclass
class TaskDocstring(CachePolicy):
    """
    Policy for computing a cache key based on the task docstring.
    """

    def compute_key(
        self,
        task_ctx: TaskRunContext,
        inputs: Optional[Dict[str, Any]],
        flow_parameters: Optional[Dict[str, Any]],
        **kwargs,
    ) -> Optional[str]:
        if not task_ctx:
            return None
        return task_ctx.task.fn.__doc__
   

AI_FN = TaskDocstring() + INPUTS
IGNORE_SESSION = TASK_SOURCE + RUN_ID + (INPUTS - 'session')

# Default token limits for OpenAI models based on docs
DEFAULT_MODEL_TOKEN_LIMITS = {
    # GPT-4o models
    "gpt-4o": 128000,
    "chatgpt-4o-latest": 128000,
    
    # GPT-4o mini models
    "gpt-4o-mini": 128000,
    
    # GPT-4o Realtime + Audio Beta
    "gpt-4o-realtime-preview": 128000,
    "gpt-4o-audio-preview": 128000,
    
    # o1 models
    "o1-preview": 128000,
    "o1-mini": 128000,
    
    # GPT-4 Turbo and GPT-4
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-instruct": 4096,
    
    # GPT base models
    "babbage-002": 16384,
    "davinci-002": 16384,
    
    # Moderation models
    "omni-moderation-latest": 32768,
    "omni-moderation-2024-09-26": 32768,
    "text-moderation-latest": 32768,
    "text-moderation-stable": 32768,
    "text-moderation-007": 32768,
}


def get_model_token_limit(model: str) -> int:
    """
    Get the token limit for a model, allowing for environment variable override.
    Environment variables should be in the format: MODEL_NAME_MAX_TOKENS
    Example: GPT_4_MAX_TOKENS=16000
    """
    # Convert model name to env var format (e.g., gpt-4-32k -> GPT_4_32K_MAX_TOKENS)
    env_var_name = f"{model.upper().replace('-', '_')}_MAX_TOKENS"
    
    # Try to get limit from environment variable
    env_limit = os.getenv(env_var_name)
    if env_limit:
        try:
            return int(env_limit)
        except ValueError:
            logger.warning(f"Invalid token limit in environment variable {env_var_name}: {env_limit}")
    
    # Fallback to default limits
    default_limit = DEFAULT_MODEL_TOKEN_LIMITS.get(model)
    if default_limit is None:
        # If model not found in defaults, use gpt-4o-mini's limit as a safe default
        logger.warning(f"Unknown model {model}, using default token limit of 128000")
        return 128000
    
    return default_limit


async def run_with_retries(
    llm_task: Coroutine,
    inputs: list[Any],
    max_retries: int = 3,
    timeout: float = 30.0,
    initial_concurrency: int = 10,
) -> list[Any]:
    """
    Execute an LLM task with multiple inputs and dynamic concurrency control.
    Handles rate limiting, timeouts, and token limits appropriately.
    """
    errors = []
    if not inputs:
        return []

    # Get the current model's token limit
    model = marvin.settings.openai.chat.completions.model
    max_tokens = get_model_token_limit(model)
    
    try:
        concurrency = initial_concurrency
        results = [None] * len(inputs)
        retry_counts = {i: 0 for i in range(len(inputs))}  # Track retries per item
        
        while concurrency > 0:
            try:
                sem = asyncio.Semaphore(concurrency)
                failed_items = set()  # Track which items failed in this batch

                async def process_item(index: int, text: str):
                    if retry_counts[index] >= max_retries:
                        logger.info(f"Skipping item {index} after {max_retries} retries")
                        return  # Skip if max retries reached
                        
                    async with sem:
                        try:
                            # Estimate tokens before making the API call
                            llm_task_doc_token_count = await estimate_tokens(llm_task.__doc__, model)
                            token_count = await estimate_tokens(text, model)
                            if token_count + llm_task_doc_token_count > max_tokens:
                                # Truncate text to fit within token limit
                                encoding = tiktoken.encoding_for_model(model)
                                truncated_tokens = encoding.encode(text)[:max_tokens-(llm_task_doc_token_count+100)]  # Leave room for system message
                                text = encoding.decode(truncated_tokens)
                                logger.warning(f"Text truncated for item {index} to fit within {max_tokens} token limit")
                                
                            async with asyncio.timeout(timeout):
                                result = await llm_task(text)
                                results[index] = result[0] if isinstance(result, list) else result
                                
                        except (RateLimitError, BadRequestError, asyncio.TimeoutError, Exception) as e:
                            attempt_info = f"attempt {retry_counts[index] + 1}/{max_retries}"
                            if isinstance(e, RateLimitError):
                                error_msg = f"Rate limit exceeded for item {index} ({attempt_info}): {str(e)}"
                            elif isinstance(e, BadRequestError):
                                error_msg = f"Invalid request for item {index} ({attempt_info}): {str(e)}"
                            elif isinstance(e, asyncio.TimeoutError):
                                error_msg = f"Timeout processing item {index} ({attempt_info}): {str(e)}"
                            else:
                                error_msg = f"Failed processing item {index} ({attempt_info}): {str(e)}"
                            
                            logger.error(error_msg)
                            errors.append(error_msg)
                            retry_counts[index] += 1
                            failed_items.add(index)
                            raise

                # Process all texts concurrently with semaphore control
                async with asyncio.TaskGroup() as tg:
                    for i, text in enumerate(inputs):
                        if retry_counts[i] < max_retries and results[i] is None:
                            tg.create_task(process_item(i, text))

                # Check if we're done or need to retry some items
                if not any(count < max_retries and results[i] is None for i, count in retry_counts.items()):
                    # Either all items succeeded or hit max retries
                    break

                # If we had rate limit errors, reduce concurrency
                if any(isinstance(e, RateLimitError) for e in errors[-len(failed_items):]):
                    if concurrency == 1:
                        await asyncio.sleep(1)  # Wait before retrying with minimum concurrency
                        continue
                    concurrency = max(concurrency // 2, 1)
                    logger.info(f"Reducing concurrency to {concurrency} due to rate limiting")
                
                # Add backoff delay between retries
                await asyncio.sleep(min(2 ** (max(retry_counts.values()) - 1), 8))  # Exponential backoff capped at 8 seconds
                continue

            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
                continue

        # Check results and raise if necessary
        failed_items = [i for i, r in enumerate(results) if r is None]
        if failed_items:
            failed_errors = [e for i, e in enumerate(errors) if i in failed_items]
            raise Exception(f"Failed to process items {failed_items} after {max_retries} retries:\n{'\n'.join(failed_errors)}")

        return [r for r in results if r is not None]

    except Exception as e:
        logger.error(f"Fatal error in run_with_retries: {str(e)}")
        raise


async def get_embedding_model():
    global _model, _tokenizer
    if _model is None:
        async with embedding_model_lock:
            # Load CodeBERT model and tokenizer
            _tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            _model = AutoModel.from_pretrained("microsoft/codebert-base")
    return _model, _tokenizer


@task(cache_policy=AI_FN)
@marvin.fn
async def summarize_code(code_fragment: str) -> str:
    """
    Summarize code to enhance a search pipeline, enabling developers to find
    relevant code by using natural language descriptions. The summary should
    clearly explain the specific purpose and functionality of the code,
    highlighting key actions, inputs, and outputs. Additionally, identify any
    major components related to the code, such as CLI commands, API endpoints,
    data models, validation schemas, or ORM interactions.

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


@task(cache_policy=INPUTS + TASK_SOURCE)
async def generate_embeddings(text: str):
    logger.info("Generating embeddings")
    model, tokenizer = await get_embedding_model()

    # Change max_length to 512 (CodeBERT's maximum) and add proper padding
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].tolist()


@task(cache_policy=INPUTS)
async def clean_text(text):
    return text.replace("\x00", "")


@task(cache_policy=INPUTS)
async def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Estimate the number of tokens in a text string for a given model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


@task(cache_policy=IGNORE_SESSION)
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

    # Get file modification time
    mtime = os.path.getmtime(filepath)
    file_updated_at = datetime.fromtimestamp(mtime, tz=UTC)

    # Execute query and get result
    result = await session.execute(
        select(Document.updated_at).where(Document.filepath == filepath)
    )
    existing_timestamp = result.scalar()

    # Return content of the file to index if:
    # - It's a new file (no existing timestamp)
    # - We're forcing reindex (replace_existing)
    # - File has been updated since last index (reindex_updated)
    if (existing_timestamp is None or 
        replace_existing or 
        (reindex_updated and file_updated_at > existing_timestamp)):
        return file_content, file_updated_at
    
    # Skip file if it exists in the database and doesn't need reindexing
    # (i.e., not a new file, not forcing reindex, and not updated since last index)
    return None


@task
async def process_document_content(content: str) -> tuple[str, list, dict, str]:
    """Process the document content to generate embeddings and metadata."""
    cleaned_content = await clean_text(content)
    file_summary = await run_with_retries(summarize_code, [cleaned_content])
    file_vector = await generate_embeddings(file_summary[0])
    file_metadata = extract_metadata_from_fragment(cleaned_content)
    return cleaned_content, file_vector, file_metadata, file_summary[0]


@task
async def process_fragments(
    content: str,
) -> tuple[list[str], list[str], list[list[float]], list[dict], list[dict]]:
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

    # Extract basic metadata
    fragment_metadata = [extract_metadata_from_node(node) for node in nodes]
    
    # Build hierarchy metadata
    hierarchy_metadata = []
    for node, meta in zip(nodes, fragment_metadata):
        hierarchy = {
            "id": meta.get("name"),
            "type": meta.get("type"),
            "parent": None,
            "children": []
        }
        
        # Check for parent class/function
        parent = getattr(node, "parent", None)
        while parent and isinstance(parent, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            hierarchy["parent"] = parent.name
            break
            
        # For classes, collect child methods and nested classes
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    hierarchy["children"].append(child.name)
                elif isinstance(child, ast.ClassDef):
                    hierarchy["children"].append(child.name)
                    
        hierarchy_metadata.append(hierarchy)

    return fragments, fragment_summaries, fragment_vectors, fragment_metadata, hierarchy_metadata


@flow
async def process_file(
    filepath: str, session, replace_existing=False, reindex_updated=False
) -> bool:
    """Main processing function that coordinates indexing stages."""
    # If replace_existing, delete the old document first
    if replace_existing:
        # Find and delete existing document
        result = await session.execute(
            select(Document).where(Document.filepath == filepath)
        )
        existing_doc = result.scalar_one_or_none()
        if existing_doc:
            # Delete associated fragments first
            await session.execute(
                delete(DocumentFragment).where(
                    DocumentFragment.document_id == existing_doc.id
                )
            )
            # Then delete the document
            await session.execute(
                delete(Document).where(Document.id == existing_doc.id)
            )
            await session.flush()  # Make changes visible within transaction without committing

    # Stage 1: Read and validate file
    result = await read_and_validate_file(
        filepath, session, replace_existing, reindex_updated
    )
    if result is None:
        return False
    file_content, updated_at = result

    # Stage 2: Process document content
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

    # Build hierarchy metadata
    hierarchy_meta = {"children": [filename]}

    # Stage 3: Chunking into fragments
    fragments, summaries, fragment_vectors, fragment_metadata, hierarchy_metadata = await process_fragments(
        cleaned_content
    )

    # Stage 4: Insert document and fragments into the database
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

    # Create fragments
    if fragments:
        for fragment, summary, vector, metadata, hierarchy in zip(
            fragments, summaries, fragment_vectors, fragment_metadata, hierarchy_metadata
        ):
            await create_document_fragment(
                session,
                document.id,
                fragment,
                summary,
                vector,
                metadata,
                hierarchy_meta=hierarchy,
            )

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
@click.option(
    "--ignored-paths",
    help="Comma-separated list of paths to ignore (default: .git,env,venv,__pycache__)",
    default=",".join(DEFAULT_IGNORED_PATHS),
)
@click.option(
    "--extensions",
    help="Comma-separated list of file extensions to process (default: .py)",
    default=",".join(DEFAULT_FILE_EXTENSIONS),
)
@async_command
async def main(
    code_dirs: list[str],
    reindex: bool = False,
    reindex_updated: bool = False,
    limit: int = -1,
    bust_cache: bool = False,
    ignored_paths: str = None,
    extensions: str = None,
):
    """
    Process code files in the given directories and insert/update the
    embeddings in the database.
    """
    filepaths = set()
    processed = 0

    if bust_cache:
        bust_file_cache()

    # Parse ignored paths from CLI option
    ignored_paths_list = [p.strip() for p in ignored_paths.split(",") if p.strip()]
    
    # Parse file extensions from CLI option
    extensions_list = [ext.strip() for ext in extensions.split(",") if ext.strip()]
    # Ensure all extensions start with a dot
    extensions_list = [ext if ext.startswith(".") else f".{ext}" for ext in extensions_list]

    async with get_session() as session:
        if not code_dirs:
            print("At least one directory or file path must be provided.")
            sys.exit(1)

        for code_dir in code_dirs:
            if not os.path.exists(code_dir):
                print(f"Path does not exist: {code_dir}")
                sys.exit(1)

            if os.path.isfile(code_dir):
                # For single files, check if the extension matches
                if any(code_dir.endswith(ext) for ext in extensions_list):
                    filepaths.add(code_dir)
            elif os.path.isdir(code_dir):
                filepaths = set()
                for root, dirs, files in os.walk(code_dir):
                    # Filter directories using the ignored paths from CLI
                    dirs[:] = [d for d in dirs if d not in ignored_paths_list]
                    for file in files:
                        # Check if file has any of the specified extensions
                        if any(file.endswith(ext) for ext in extensions_list):
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
            print(f"Code files processed and inserted/updated successfully. Extensions processed: {', '.join(extensions_list)}")
