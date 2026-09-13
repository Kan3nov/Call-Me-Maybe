vpath %.py src

NAME = main.py

all: install run

run: $(NAME)
	@uv run src/main.py
install:
	@uv sync
clean:
	rm -rf __pycache__ .mypy_cache

lint:
	flake8 . and mypy . --warn-return-any --warn-unused-ignores
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
