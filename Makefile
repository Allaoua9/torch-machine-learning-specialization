install:
	poetry install

##
##    Maintain/QA
##───────────────────

lint:
	poetry run python -m ruff check .

fix:
	poetry run python -m ruff check --fix .

format:
	poetry run python -m ruff format .

test:
# poetry run python -m pytest ./tests

##
##      Cleanup
##───────────────────

clean: clean-dist clean-pyc clean-mypy ## Perform all clean recipes

clean-pyc: ## Remove python bytecode cache
	find . -type f -name '*.py[co]' -delete ;\
	find . -type d -name '__pycache__' -delete ;\
	find . -type d -name '.pytest_cache' -exec rm -rf {} +

clean-dist: ## Remove distribution folder
	rm -rf dist

clean-mypy: ## Remove mypy cache
	rm -rf .mypy_cache

	