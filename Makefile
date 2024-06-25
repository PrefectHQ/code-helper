# Define variables
PYTHON = python
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
	uvicorn app:app

.PHONY: run-api-dev
run-api: ## Run the API in reload mode
	uvicorn app:app --reload

.PHONY: create-embeddings
create-embeddings: ## Run the processing code recursively with a directory
	@echo "Running create_embeddings.py with directory: $(DIRECTORY)"
	$(PYTHON) create_embeddings.py $(DIRECTORY)

.PHONY: migrations
migrations: ## Run Alembic migrations
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
		$(PYTHON) -c 'from models import Base, engine; Base.metadata.drop_all(engine)'; \
	else \
		echo "Drop database cancelled."; \
	fi

.PHONY: clean
clean: ## Clean up temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .venv
