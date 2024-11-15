from typing import Any
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query_text: str


class SearchResult(BaseModel):
    document_id: str
    score: float
    filename: str
    filepath: str
    fragment_content: str
    metadata: dict[str, Any]
    imports: list[str]


class SearchResponse(BaseModel):
    results: list[SearchResult]
    count: int
