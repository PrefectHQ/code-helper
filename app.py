import re
from dotenv import load_dotenv
from fastapi import FastAPI
from InstructorEmbedding import INSTRUCTOR
from numpy import extract
from models import get_session, hybrid_search
from schemas import SearchResponse, SearchRequest
from logging import getLogger

load_dotenv()

logger = getLogger(__name__)

# Lazy load the Instructor-XL model
_model = None


def get_model():
    global _model
    if _model is None:
        _model = INSTRUCTOR('hkunlp/instructor-xl')
    return _model


def vectorize_query(query_text: str):
    """
    Vectorize the query text using the Instructor-XL model
    """
    model = get_model()

    # Test using the same instruction we used for embedding code
    # instruction = "Represent the code snippet:"
    instruction = "Represent the search query:"
    return model.encode([instruction, query_text])[0].tolist()


app = FastAPI(
    title="Embeddings Search API",
    description="API to search for embeddings in a pgvector database.",
    version="1.0.0"
)


def extract_filenames(query_string: str):
    filename_regex = re.compile(r"([^/]+\.py)")
    matches = filename_regex.findall(query_string)
    return matches


@app.post("/v1/search_embeddings", response_model=SearchResponse)
def search_embeddings(request: SearchRequest):
    """
    Search for code fragments using a hybrid keyword and vector search approach.
    """
    query_text = request.query_text
    session = get_session()
    filenames = extract_filenames(query_text)

    try:
        query_vector = vectorize_query(query_text)
        results = hybrid_search(session, query_text, query_vector, filenames)

        return {"results": results}
    finally:
        session.close()

# Run the app with: make run-api
