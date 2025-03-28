#Makefile for Python code formatting

# Tools
PYTHON := python3
ISORT := isort
BLACK := black
RUFF := ruff

# Directories to format (current directory and all subdirectories)
SOURCE_DIR := .

# Formatting commands
format:
	@echo "Running isort..."
	@$(ISORT) $(SOURCE_DIR)
	@echo "Running black..."
	@$(BLACK) $(SOURCE_DIR)
	@echo "Running ruff format..."
	@$(RUFF) format $(SOURCE_DIR)