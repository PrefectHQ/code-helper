import os
import sys
from datetime import datetime, timedelta
from threading import Lock
from dotenv import load_dotenv

import marvin
from prefect import flow, task
from prefect.tasks import task_input_hash
from prefect.utilities.annotations import quote

from code_fragment_extractor import extract_code_fragments_from_file_content
from models import get_session, Document, DocumentFragment

load_dotenv()

IGNORED_PATHS = ['.git', 'env', 'venv', '__pycache__']

instructor_lock = Lock()
_model = None
_summarization_pipeline = None

marvin.settings.openai.chat.completions.model = 'gpt-3.5-turbo'


def get_embedding_model():
    with instructor_lock:
        global _model
        if _model is None:
            from InstructorEmbedding import INSTRUCTOR
            _model = INSTRUCTOR('hkunlp/instructor-xl')
    return _model


# NOTE: Disabled because codetrans summary results of smaller code fragments were low
#       quality in testing.
# def get_summarization_pipeline():
#     with code_trans_lock:
#         global _summarization_pipeline
#         if _summarization_pipeline is None:
#             from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
#
#             _summarization_pipeline = SummarizationPipeline(
#                 model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_base_source_code_summarization_python"),
#                 tokenizer=AutoTokenizer.from_pretrained(
#                     "SEBIS/code_trans_t5_base_source_code_summarization_python", skip_special_tokens=True
#                 ),
#                 device=0
#             )
#     return _summarization_pipeline
#
#
# def summarize_code(code: str) -> str:
#     pipeline = get_summarization_pipeline()
#     return pipeline([code])


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=7), tags=["openai"])
@marvin.fn
def summarize_code(code: str) -> str:
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


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=7), tags=["openai"])
@marvin.fn()
def extract_metadata(code: str) -> str:
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


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=7), tags=["model"], refresh_cache=True)
def embed_text_with_instructor(text):
    instruction = "Represent the code snippet:"
    model = get_embedding_model()
    embeddings = model.encode([[instruction, text]])
    return embeddings[0].tolist()


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=7))
def clean_text(text):
    return text.replace('\x00', '')


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=7), refresh_cache=True)
def process_file(filepath):
    session = get_session()

    with open(filepath, "r") as f:
        try:
            file_content = f.read()
        except UnicodeDecodeError:
            print(f"Error reading file: {filepath}")
            return

    cleaned_content = clean_text(file_content)

    # Summarize the file as a whole for TF-IDF/keyword search
    file_summary = summarize_code(cleaned_content)

    # 1. Create embeddings for the file as a whole.
    file_vector = embed_text_with_instructor(quote(cleaned_content))

    # 2. Create embeddings for each code fragment.
    fragments = extract_code_fragments_from_file_content(cleaned_content)
    fragment_vectors = []
    vector_futures = [embed_text_with_instructor.submit(quote(frag)) for frag in fragments]
    for future in vector_futures:
        future.wait()
        fragment_vectors.append(future.result())

    updated_at = datetime.fromtimestamp(os.path.getmtime(filepath))
    file_metadata = extract_metadata(quote(cleaned_content))

    fragment_metadata = []
    fragment_metadata_futures = [extract_metadata.submit(quote(frag)) for frag in fragments]
    for future in fragment_metadata_futures:
        future.wait()
        fragment_metadata.append(future.result())

    session.query(DocumentFragment).filter(DocumentFragment.document.has(filepath=filepath)).delete()
    session.query(Document).filter_by(filepath=filepath).delete()

    document = Document(
        filename=os.path.basename(filepath),
        filepath=filepath,
        file_content=cleaned_content,
        vector=file_vector,
        summary=file_summary,
        meta=file_metadata,
        updated_at=updated_at
    )
    session.add(document)
    session.flush()  # Flush to get the ID of the document record

    for fragment, vector, metadata in zip(fragments, fragment_vectors, fragment_metadata):
        document_fragment = DocumentFragment(
            document_id=document.id,
            fragment_content=fragment,
            vector=vector,
            updated_at=updated_at,
            meta=metadata
        )
        session.add(document_fragment)

    # Commit the transaction and close the session
    session.commit()
    session.close()

    print(f"Processed file: {filepath}")


@flow
def process_files(code_dirs: list[str]):
    """
    Process all Python files in the given directories and insert/update the
    embeddings in the database.
    """
    if not code_dirs:
        print("At least one directory path must be provided.")
        sys.exit(1)

    # Iterate over files in the directory
    filepaths = []
    for code_dir in code_dirs:
        for root, dirs, files in os.walk(code_dir):
            # Exclude ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORED_PATHS]
            for file in files:
                if file.endswith(".py") and file != "__init__.py":  # Process only Python files excluding __init__.py
                    filepath = os.path.join(root, file)
                    filepaths.append(filepath)

    futures = [process_file.submit(filepath) for filepath in filepaths]

    # Run file processing tasks in parallel
    for future in futures:
        future.wait()

    print("Code files processed and inserted/updated successfully.")


if __name__ == "__main__":
    process_files(sys.argv[1:])
