import asyncio
from datetime import datetime, timedelta
import json
from logging import getLogger
import os
import sys

import aiofiles
from dotenv import load_dotenv
import marvin
from prefect import flow, get_client, task

from prefect.tasks import task_input_hash
from prefect.transactions import CommitMode, transaction
from prefect.utilities.annotations import quote

from code_fragment_extractor import (
    extract_code_fragments_from_file_content,
    extract_imports_from_file_content,
)
from models import Document, DocumentFragment, get_session

load_dotenv()

IGNORED_PATHS = [".git", "env", "venv", "__pycache__"]

logger = getLogger(__name__)

instructor_lock = asyncio.Lock()
_model = None

marvin.settings.openai.chat.completions.model = "gpt-3.5-turbo"


async def get_embedding_model():
    global _model
    if _model is None:
        async with instructor_lock:
            from InstructorEmbedding import INSTRUCTOR

            _model = INSTRUCTOR("hkunlp/instructor-xl")
    return _model


@task(
    persist_result=True,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=7),
    tags=["openai"],
)
@marvin.fn
async def summarize_code(code: str) -> str:
    """
    You are an LLM preparing summarizations of code fragments to help in a
    search pipeline. Summaries will help developers use natural language to
    describe what they want. Developers may use general technical terms like
    "orm" or "model" or the specific names of functions and classes in the
    codebase. Or they may also ask questions about the code, either by naming
    classes or functions or by providing example code. You need to summarize the
    code by referring to what it does specifically and then inferring major
    components it relates to, such as CLI commands, API endpoints, data models,
    or validation schemas.
    """


@task(
    persist_result=True,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=7),
    tags=["openai"],
)
@marvin.fn()
async def extract_metadata(code: str) -> str:
    """
    You are an LLM preparing metadata extraction for code fragments to help with
    natural language search. Developers will query for help writing code similar
    to code in the codebase from which these summaries are generated. Queries
    may also include questions about specific functions or classes in the code
    base, or provide example code. You need to generate metadata about code that
    will help these queries in the future.

    If the code you extract metadata from is one function or class, extract
    metadata such as whether the code is a function or class. Extract the module
    name. If the code is a class, extract the class name and the parent class.
    Extract the class's major component (CLI commands, API endpoints, data
    models, or schemas, utilities, tests, etc.). If the code appears to be an
    entire file, extract metadata such as the major components it relates to,
    such as CLI commands or database models, and any other metadata relevant to
    the file as a whole that would be useful to natural language search. Include
    the names of functions and classes.

    DO NOT extract metadata that is not useful for natural language search. DO
    NOT extract metadata describing import statements.

    Brevity is important: capture metadata that is useful for natural language
    search but avoid capturing metadata that is not useful for these purposes.
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


@task(
    persist_result=True,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=7),
    tags=["model"],
)
async def embed_text_with_instructor(text):
    instruction = "Represent the code snippet:"
    model = await get_embedding_model()
    embeddings = model.encode([[instruction, text]])
    return embeddings[0].tolist()


@task(
    persist_result=True,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=7),
)
async def clean_text(text):
    return text.replace("\x00", "")


@task(
    persist_result=True,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(days=7),
    refresh_cache=True,
)
async def process_file(filepath):
    with transaction(commit_mode=CommitMode.EAGER):
        session = get_session()

        try:
            async with aiofiles.open(filepath, "r") as f:
                file_content = await f.read()
        except UnicodeDecodeError:
            print(f"Error reading file: {filepath}")
            return

        cleaned_content = await clean_text(file_content)

        # Summarize the file as a whole for TF-IDF/keyword search
        file_summary = await summarize_code(cleaned_content)

        # 1. Create embeddings for the file as a whole.
        file_vector = await embed_text_with_instructor(quote(cleaned_content))

        # 2. Create embeddings for each code fragment.
        fragments = extract_code_fragments_from_file_content(cleaned_content)

        async with asyncio.TaskGroup() as tg:
            fragment_vectors_tasks = [
                tg.create_task(embed_text_with_instructor(quote(frag)))
                for frag in fragments
            ]
        fragment_vectors = [f.result() for f in fragment_vectors_tasks]

        updated_at = datetime.fromtimestamp(os.path.getmtime(filepath))
        try:
            file_metadata = json.loads(await extract_metadata(quote(cleaned_content)))
        except (ValueError, json.JSONDecodeError):
            file_metadata = {}

        file_metadata["imports"] = extract_imports_from_file_content(cleaned_content)

        async with asyncio.TaskGroup() as tg:
            fragment_metadata_tasks = [
                tg.create_task(extract_metadata(quote(frag))) for frag in fragments
            ]
        fragment_metadata = [f.result() for f in fragment_metadata_tasks]

        session.query(DocumentFragment).filter(
            DocumentFragment.document.has(filepath=filepath)
        ).delete()
        session.query(Document).filter_by(filepath=filepath).delete()

        document = Document(
            filename=os.path.basename(filepath),
            filepath=filepath,
            file_content=cleaned_content,
            vector=file_vector,
            summary=file_summary,
            meta=file_metadata,
            updated_at=updated_at,
        )
        session.add(document)
        session.flush()  # Flush to get the ID of the document record

        for fragment, vector, metadata in zip(
            fragments, fragment_vectors, fragment_metadata
        ):
            document_fragment = DocumentFragment(
                document_id=document.id,
                fragment_content=fragment,
                vector=vector,
                updated_at=updated_at,
                meta=metadata,
            )
            session.add(document_fragment)

        session.commit()
        session.close()

        print(f"Processed file: {filepath}")


async def reset_concurrency_limits():
    """
    This is a hack because concurrency limits aren't releasing when tasks
    crash/fail.
    """
    async with get_client() as client:
        await client.reset_concurrency_limit_by_tag("openai")
        await client.reset_concurrency_limit_by_tag("model")


@flow(persist_result=True)
async def process_files(code_dirs: list[str]):
    """
    Process all Python files in the given directories and insert/update the
    embeddings in the database.
    """
    if not code_dirs:
        print("At least one directory or file path must be provided.")
        sys.exit(1)

    await reset_concurrency_limits()

    tasks = []
    async with asyncio.TaskGroup() as tg:
        for code_dir in code_dirs:
            if not os.path.exists(code_dir):
                print(f"Path does not exist: {code_dir}")
                sys.exit(1)

            if os.path.isfile(code_dir):
                tasks.append(tg.create_task(process_file(code_dir)))
            elif os.path.isdir(code_dir):
                filepaths = []
                for root, dirs, files in os.walk(code_dir):
                    dirs[:] = [d for d in dirs if d not in IGNORED_PATHS]
                    for file in files:
                        if file.endswith(".py") and file != "__init__.py":
                            filepath = os.path.join(root, file)
                            filepaths.append(filepath)

                    for filepath in filepaths:
                        tasks.append(tg.create_task(process_file(filepath)))
            else:
                print(f"Invalid path: {code_dir}")
                sys.exit(1)

    print("Code files processed and inserted/updated successfully.")


if __name__ == "__main__":
    asyncio.run(process_files(sys.argv[1:]))
