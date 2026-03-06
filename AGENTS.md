# Agent Instructions

## Package Management

This project uses [uv](https://github.com/astral-sh/uv) for Python dependency management.

- **Never** use pip directly. Always use uv commands instead.
- **Add dependency**: `uv add <package>`
- **Remove dependency**: `uv remove <package>`
- **Install all deps**: `uv sync`
- **Run commands**: `uv run <command>`

When adding/updating dependencies, edit `pyproject.toml` manually and run `uv sync`.
