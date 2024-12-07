# Code Search MCP Server with FastMCP

A search API that combines keyword-based and vector-based searches to find relevant code snippets and documents. The API is meant for consumption from an LLM library or tool for Retrieval Augmented Search (RAG) and is configured to run as an MCP server with FastMCP, usable by Claude.

## Features

- **Postgres** Uses PostgreSQL's native full-text search and pgvector.
- **Keyword Search**: Full-text search on document and chunk contents, summaries, and filenames.
- **Vector Search**: Semantic search using vector embeddings.
- **Hybrid Relevance**: Uses [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (RRF) to merge results from keyword and vector searches.
- **Recursive Summarization**: Shares some insights from [RAPTOR](https://arxiv.org/abs/2401.18059) - modules, classes, and functions are recursively summarized.

## Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension

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
pip install -r requirements.lock
```

I usually install the package too:

```sh
pip install -e .
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

But you most likely want to run the server in dev mode:

```sh
make run-api-dev
```

Then you can test it in the MCP Inspector at http://localhost:5173.

When you're ready to use it in Claude, you'll need to install the MCP server in Claude (see next section).

### Installing the MCP server in Claude

This server is more complex than the examples in the FastMCP docs, so I recommend editing your MCP server config by hand (in `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "Code search MCP server": {
      "command": "/Users/andrew/src/code-helper/env/bin/uv",
      "args": [
        "run",
        "-p",
        "/Users/andrew/src/code-helper/env/bin/python",
        "fastmcp",
        "run",
        "/Users/andrew/src/code-helper/src/code_helper/app.py"
      ]
    }
  }
}
```
**NOTE**: You'll need to replace the path to the Python binary and the path to the `app.py` file. After editing the config, restart Claude.
