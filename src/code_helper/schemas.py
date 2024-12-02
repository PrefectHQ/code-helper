from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query_text: str
    filenames: Optional[List[str]] = None


class SearchResultMetadata(BaseModel):
    name: str
    type: str
    docstring: Optional[str] = None
    decorators: Optional[List[str]] = None
    is_async: Optional[bool] = None
    parameters: Optional[List[str]] = None
    return_type: Optional[str] = None
    parent: Optional[str] = None
    parent_classes: Optional[List[str]] = None


class SearchResult(BaseModel):
    id: int
    content: str
    summary: str
    metadata: SearchResultMetadata
    score: float
    document_id: str
    filename: str
    filepath: str
    fragment_content: str
    imports: List[str]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    count: int
