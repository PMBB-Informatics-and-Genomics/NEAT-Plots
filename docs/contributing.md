# Contributing

Thank you for your interest in contributing to NEAT-Plots!

## Development setup

```bash
git clone https://github.com/RitchieLab/NEAT-Plots.git
cd NEAT-Plots
pip install -e ".[dev]"
```

## Running the test suite

```bash
pytest tests/                         # unit tests (fast, ~4 s)
pytest tests/integration/             # integration tests against the chr22 fixture
pytest tests/ tests/integration/ -v   # everything
```

All tests must pass before opening a pull request. The CI workflow runs the full suite on Python 3.9, 3.10, 3.11, and 3.12.

## Code style

We use **ruff** for linting and formatting:

```bash
ruff check neat_plots/        # lint
ruff format neat_plots/       # format
```

Configuration is in `pyproject.toml` under `[tool.ruff]`.

## Docstring style

All public methods should have [NumPy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings. The API reference pages are auto-generated from these docstrings via mkdocstrings.

## Building the documentation locally

```bash
pip install neat-plots[docs]
mkdocs serve        # live-reloading preview at http://127.0.0.1:8000
mkdocs build        # one-off static build into site/
```

## Adding a new plot type

1. Add any shared constants to `neat_plots/_constants.py`
2. If the plot type shares substantial logic with existing plots, add the shared methods to `neat_plots/_base.py`
3. Create `neat_plots/your_plot.py` with a class that inherits from `BasePlot`
4. Export the class from `neat_plots/__init__.py`
5. Write unit tests in `tests/test_your_plot.py`
6. Add a usage guide page in `docs/usage/`
7. Add a gallery figure if applicable

## Submitting a pull request

1. Fork the repository and create a feature branch
2. Make your changes, add tests, and verify `pytest` passes
3. Run `ruff check` and fix any issues
4. Open a PR against the `main` branch with a clear description of the change

## Reporting bugs

Please open an issue at [github.com/RitchieLab/NEAT-Plots/issues](https://github.com/RitchieLab/NEAT-Plots/issues) and include:

- Your Python version and NEAT-Plots version
- A minimal reproducible example
- The full traceback if applicable
