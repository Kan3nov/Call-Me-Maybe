vpath %.py src

NAME = main.py
all: install run

run: $(NAME)
	@uv run src/main.py
install:
	@uv sync