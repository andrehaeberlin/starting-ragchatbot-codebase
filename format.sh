#!/bin/bash

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

echo "Formatting Python code..."

# Fix import sorting / lint-fixable issues first, then apply Black's formatting
uv run ruff check --fix backend main.py
uv run black backend main.py

echo "Done."
