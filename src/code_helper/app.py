import re
from logging import getLogger

from dotenv import load_dotenv
from fastapi import FastAPI

from code_helper.index import generate_embeddings
from code_helper.models import get_session, hybrid_search
from code_helper.schemas import SearchRequest, SearchResponse

load_dotenv()

logger = getLogger(__name__)


app = FastAPI(
    title="Code Search API",
    description="LLM-powered code search API.",
    version="0.1.0",
)


def extract_filenames(query_string: str):
    filename_regex = re.compile(r"([^/]+\.py)")
    matches = filename_regex.findall(query_string)
    return matches


@app.post("/v1/search_embeddings", response_model=SearchResponse)
async def search_embeddings(request: SearchRequest):
    """
    Search for code fragments using a hybrid keyword and vector search approach.
    """
    async with get_session() as session:
        query_text = request.query_text
        query_vector = await generate_embeddings(query_text)

        results = await hybrid_search(
            session, query_text, query_vector, limit=10
        )
        return {"results": results, "count": len(results)}


# Run the app with: make run-api
