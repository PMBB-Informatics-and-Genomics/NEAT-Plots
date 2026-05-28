# Backward-compatibility shim.
# The canonical package is now `neat_plots`.
# Old import style still works:
#   from manhattan_plot import ManhattanPlot
#   from manhattan_plot import BoroughsPlot
from neat_plots import ManhattanPlot, BoroughsPlot  # noqa: F401

__all__ = ["ManhattanPlot", "BoroughsPlot"]
