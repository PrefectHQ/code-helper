"""
Run the app with:
    $ make run-api
"""
from logging import getLogger
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastmcp import FastMCP

from code_helper.index import generate_embeddings
from code_helper.models import get_session, hybrid_search, init_db_connection
from code_helper.schemas import SearchRequest, SearchResponse, SearchResult

load_dotenv()

logger = getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastMCP):
    init_db_connection()
    yield


mcp = FastMCP("Code search MCP server", lifespan=lifespan)


@mcp.tool()
async def code_search(request: SearchRequest) -> SearchResponse:
    """Search for code fragments using a hybrid keyword and vector search approach."""
    async with get_session() as session:
        query_vector = await generate_embeddings(request.query_text)

        results = await hybrid_search(
            session, request.query_text, query_vector,
            limit=request.limit, offset=request.offset
        )
        return SearchResponse(results=results, count=len(results))
