
    .PHONY: install dev lint format test typecheck run-sim run-adfa

    install:
	pip install -e .

    dev:
	pip install -r requirements.txt
	pip install black ruff pytest mypy pre-commit
	pre-commit install

    lint:
	ruff check .
	black --check . || true

    format:
	black .
	ruff check . --fix

    test:
	pytest -q

    typecheck:
	mypy src || true

    run-sim:
	python -m src.run_sr_deepsvdd --dataset banana --seed 42 || true

    run-adfa:
	python -m src.run_sr_deepsvdd_ADFA_v4 --prepare --adfa-root src/data/ADFA-LD --data-dir stc/data/adfa_features --ngram 1,3 --max-features 5000 --val-frac 0.2 --seed 42 || true
