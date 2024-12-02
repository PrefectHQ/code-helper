"""
Run the app with:
    $ make run-api
"""
from logging import getLogger
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from code_helper.index import generate_embeddings
from code_helper.models import get_session, hybrid_search, init_db_connection
from code_helper.schemas import SearchRequest, SearchResponse

load_dotenv()

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db_connection()
    yield


app = FastAPI(
    title="Code Search API",
    description="LLM-powered code search API.",
    version="1.0",
    lifespan=lifespan,
)

@app.post("/v1/code_search", response_model=SearchResponse)
async def code_search(request: SearchRequest):
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
