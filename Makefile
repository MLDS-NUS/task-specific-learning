.PHONY: tests docs

dependencies:
	@echo "Initializing Git..."
	git init
	@echo "Installing dependencies..."
	poetry install

env: dependencies
	@echo "Activating virtual environment..."
	poetry shell

tests:
	pytest

docs:
	@echo "Save documentation to docs..."
	pdoc taskspec -o docs -d google --math
