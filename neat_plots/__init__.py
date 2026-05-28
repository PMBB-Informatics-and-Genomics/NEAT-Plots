"""
NEAT-Plots
==========

Publication-quality genomic visualization for GWAS, TWAS, ExWAS, and PheWAS.

Quick start
-----------
::

    from neat_plots import ManhattanPlot, BoroughsPlot

    mp = ManhattanPlot("sumstats.txt.gz", title="My GWAS")
    mp.load_data()
    mp.clean_data(col_map={"CHR": "#CHROM", "BP": "POS", "PVAL": "P"})
    mp.get_thinned_data()
    mp.update_plotting_parameters(vertical=True, sig=5e-8, merge_genes=True)
    mp.full_plot(save="manhattan.png")
    mp.qq_plot(save="qq.png")
"""

from .manhattan_plot import ManhattanPlot
from .boroughs_plot import BoroughsPlot

__all__ = [
    "ManhattanPlot",
    "BoroughsPlot",
]

__version__ = "0.2.0"
