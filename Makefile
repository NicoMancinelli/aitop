.DEFAULT_GOAL := help
.PHONY: help bootstrap sync test lint fmt run watch doctor build clean

help:  ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1;36m%-12s\033[0m %s\n", $$1, $$2}'

bootstrap:  ## Create the dev environment (handles the iCloud venv trap)
	@./scripts/bootstrap.sh

sync:  ## Install/update dependencies
	uv sync --extra dev

test:  ## Run the test suite
	uv run pytest -q

lint:  ## Lint and check formatting
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt:  ## Auto-format
	uv run ruff format src tests
	uv run ruff check --fix src tests

run:  ## Run the dashboard
	uv run aitop

watch:  ## Run the dashboard in live mode
	uv run aitop --watch

doctor:  ## Show what telemetry is available here
	uv run aitop doctor

build:  ## Build sdist + wheel
	uv build

clean:  ## Remove build artefacts and caches
	rm -rf dist build .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
