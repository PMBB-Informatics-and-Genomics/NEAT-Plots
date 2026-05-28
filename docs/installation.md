# Installation

## Requirements

- Python ≥ 3.9
- matplotlib ≥ 3.6, numpy ≥ 1.23, pandas ≥ 1.5, scipy ≥ 1.9

## From PyPI (recommended)

```bash
pip install neat-plots
```

## From conda-forge

```bash
conda install -c conda-forge neat-plots
```

!!! note
    The Bioconda recipe is under preparation. Once available, you can also install via `conda install -c bioconda neat-plots`.

## Development install

Clone the repository and install in editable mode with all developer extras:

```bash
git clone https://github.com/RitchieLab/NEAT-Plots.git
cd NEAT-Plots
pip install -e ".[dev]"
```

## Optional extras

| Extra | What it adds | Install |
|-------|-------------|---------|
| `gui`  | Qt-based interactive viewer (PySide6) | `pip install neat-plots[gui]` |
| `dev`  | pytest, ruff, mypy, build toolchain  | `pip install neat-plots[dev]` |
| `docs` | MkDocs + Material + mkdocstrings     | `pip install neat-plots[docs]` |

## Verifying the install

```python
import neat_plots
print(neat_plots.__version__)
```

## Backward compatibility

If your existing scripts use the legacy import style, they will continue to work without any changes:

```python
from manhattan_plot import ManhattanPlot   # still works
from manhattan_plot import BoroughsPlot    # still works
```

These re-export from `neat_plots` via a backward-compatibility shim included in the package.
