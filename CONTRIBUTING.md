# Contributing

Thanks for helping improve SR-DeepSVDD!

## Setup
1. Fork & clone the repo.
2. Create a virtual env (Python 3.10+).
3. `pip install -e . -r requirements.txt`
4. Install dev tools: `pip install black ruff pytest mypy pre-commit` and run `pre-commit install`.

## Workflow
- Create a feature branch from `main`.
- Keep PRs focused and small.
- Add/adjust tests when changing behavior.
- Run `make lint && make test` before pushing.

## Coding style
- Black + Ruff enforced.
- Prefer type hints where practical.
- Avoid breaking the public CLI without deprecation notes.

