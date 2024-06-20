# Code Helper Search API

A search API that combines keyword-based and vector-based searches to find relevant code snippets and documents. It uses a PostgreSQL database with full-text search capabilities and vector search via pgvector. The results from both searches are combined using the Reciprocal Rank Fusion (RRF) algorithm to improve relevance.

## Features

- **Keyword Search**: Full-text search on document contents, summaries, and filenames.
- **Vector Search**: Semantic search using vector embeddings.
- **Combined Results**: Uses the Reciprocal Rank Fusion (RRF) algorithm to merge results from both keyword and vector searches.

## Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension
- SQLAlchemy
- Alembic for database migrations

## Installation

1. **Clone the Repository**

```sh
git clone https://github.com/PrefectHQ/code-helper-search-api.git
cd code-helper-search-api
```

2. **Create and Activate a Virtual Environment**

```sh
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. **Install Dependencies**

```sh
pip install -r prefect2.lock
```

4. **Configure the Database**

Update the `DATABASE_URL` in your configuration to point to your PostgreSQL database.

## Database Setup

**Initialize the Database**

```sh
alembic upgrade head
````

## Usage

### Running the Server

To start the server, run:

```sh
make run-api
```

### API Endpoints

#### Search Endpoint

- **URL**: `/v1/search_embeddings`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "query_text": "search query here"
  }
  ```
- **Response**:
  ```json
  {
    "results": [
      {
        "document_id": "1",
        "score": 0.123,
        "filename": "example.py",
        "filepath": "/path/to/example.py",
        "fragment_content": "def example_function(): ...",
        "metadata": {}
      },
      ...
    ]
  }
  ```
