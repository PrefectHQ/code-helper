from dotenv import load_dotenv
from fastapi import FastAPI
from InstructorEmbedding import INSTRUCTOR
from sqlalchemy import text
from models import get_session
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


# Create the FastAPI app
app = FastAPI(
    title="Embeddings Search API",
    description="API to search for embeddings in a pgvector database.",
    version="1.0.0"
)


# Search embeddings endpoint
@app.post("/v1/search_embeddings", response_model=SearchResponse)
def search_embeddings(request: SearchRequest):
    query_text = request.query_text

    # Create a new database session
    session = get_session()

    try:
        # Vectorize the query text using the Instructor model
        model = get_model()
        instruction = "Represent the search query:"
        # Test using the same instruction we used for embedding code
        # instruction = "Represent the code snippet:"
        query_vector = model.encode([instruction, query_text])[0].tolist()

        # TODO: keyword search and index, fuse results with vector search

        # Perform document similarity search using pgvector
        # query_documents = text(
        #     """
        #     SELECT id, filepath, vector <-> :query_vector AS score
        #     FROM documents
        #     ORDER BY score DESC
        #     LIMIT 5
        #     """
        # )
        #
        # logger.debug("Query documents: %s", query_documents)
        #
        # # Execute the query for documents
        # document_results = session.execute(query_documents, {'query_vector': str(query_vector)}).fetchall()

        response = []

        # Perform the similarity search for fragments within the documents
        query_fragments = text(
            """
            SELECT document_fragments.vector, documents.id as document_id, document_fragments.meta as meta, documents.filename as filename, documents.filepath as filepath, document_fragments.fragment_content as fragment_content, document_fragments.vector <-> :query_vector as score
            FROM document_fragments
            LEFT JOIN documents ON documents.id = document_fragments.document_id
            ORDER BY document_fragments.vector
            LIMIT 10
            """
        )

        # Execute the query for fragments
        fragment_results = session.execute(
            query_fragments,
            {'query_vector': str(query_vector)}
        ).fetchall()

        for frag_row in fragment_results:
            response.append(
                {
                    "document_id": str(frag_row.document_id),
                    "score": frag_row.score,
                    "filename": frag_row.filename,
                    "filepath": frag_row.filepath,
                    "fragment_content": frag_row.fragment_content,
                    "metadata": frag_row.meta
                }
            )

        return {"results": response}
    finally:
        session.close()

# Run the app with: uvicorn filename:app --reload
