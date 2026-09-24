vpath %.py src

TEST ?= public

all: run

run: install
	@uv run python -m src

install:
	@uv sync
	@cd llm_sdk && uv sync

debug: install
	@uv run python -m pdb src/__main__.py

clean:
	@find \( -name "__pycache__" -o -name ".mypy_cache" \) -prune -exec rm -rf {} \;
	@rm -rf data/output

lint:
	@flake8 src/ && mypy src/ --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --follow-imports=skip

prepare:
	@cd moulinette/ && uv run python -m moulinette prepare_exercises --set $(TEST)

generate:
	@uv run python -m src --input moulinette/data/input/function_calling_tests.json --functions_definition moulinette/data/input/functions_definition.json

grade:
	@cd moulinette/ && uv run python -m moulinette grade_student_answers ../data/output/function_calls.json --set $(TEST)