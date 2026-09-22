vpath %.py src

all: run

run: install
	@uv run python -m src

install:
	@uv sync

debug: install
	@uv run python -m pdb src/__main__.py

clean:
	@find \( -name "__pycache__" -o -name ".mypy_cache" \) -prune -exec rm -rf {} \;
	@rm -rf data/output

lint:
	flake8 . and mypy . --warn-return-any --warn-unused-ignores
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

prepare:
	@cd moulinette/ && uv run python -m moulinette prepare_exercises --set public

generate:
	@uv run python -m src --input moulinette/data/input/function_calling_tests.json --functions_definition moulinette/data/input/functions_definition.json

grade:
	@cd moulinette/ && uv run python -m moulinette grade_student_answers ../data/output/function_calls.json --set public