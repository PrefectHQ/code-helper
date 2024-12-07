# Define variables
PYTHON = ./env/bin/python
ALEMBIC = alembic
PYTEST = pytest

# Default command
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

.PHONY: run-api
run-api: ## Run the API
	# uvicorn code_helper.app:app
	fastmcp run src/code_helper/app.py

.PHONY: run-api-dev
run-api-dev: ## Run the API in reload mode
	# uvicorn code_helper.app:app --reload
	fastmcp dev src/code_helper/app.py

.PHONY: create-embeddings
create-embeddings: ## Run the processing code recursively with a directory
	@echo "Running create_embeddings.py with directory: $(DIRECTORY)"
	$(PYTHON) create_embeddings.py $(DIRECTORY)

.PHONY: migrate
migrate: ## Run Alembic migrations
	$(ALEMBIC) upgrade head

.PHONY: create-migration
create-migration: ## Create a new Alembic migration
	@read -p "Enter migration message: " msg; \
	$(ALEMBIC) revision --autogenerate -m "$$msg"

.PHONY: tests
tests: ## Run tests with pytest
	$(PYTEST) tests

.PHONY: setup-db
setup-db: ## Setup the initial database
	$(ALEMBIC) upgrade head

.PHONY: drop-db
drop-db: ## Drop the database
	@read -p "Are you sure you want to drop the database? (y/N): " confirm; \
	if [ "$$confirm" = "y" ]; then \
		$(PYTHON) -c 'import asyncio; from code_helper.models import async_drop_db; asyncio.run(async_drop_db())' \
	else \
		echo "Drop database cancelled."; \
	fi

.PHONY: init-db
init-db: ## Initialize the database
	@read -p "Are you sure you want to initialize the database? (y/N): " confirm; \
	if [ "$$confirm" = "y" ]; then \
		$(PYTHON) -c 'import asyncio; from code_helper.models import init_db; asyncio.run(init_db())' \
	else \
		echo "Initialize database cancelled."; \
	fi

.PHONY: clean
clean: ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .venv
