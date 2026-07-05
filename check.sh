#!/bin/bash

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "Error: backend directory not found"
    exit 1
fi

echo "Checking code formatting and lint rules..."

FAILED=0

uv run ruff check backend main.py || FAILED=1
uv run black --check --diff backend main.py || FAILED=1

if [ "$FAILED" -ne 0 ]; then
    echo "Code quality checks failed."
    exit 1
fi

echo "All checks passed."
