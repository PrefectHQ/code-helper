from typing import Any

from pydantic import BaseModel


# Pydantic model for the request body
class SearchRequest(BaseModel):
    query_text: str


# Pydantic model for the response
class SearchResult(BaseModel):
    document_id: str
    score: float
    filename: str
    filepath: str
    fragment_content: str
    metadata: str


class SearchResponse(BaseModel):
    results: list[SearchResult]

