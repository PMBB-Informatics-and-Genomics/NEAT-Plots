# NEAT-Plots

**Publication-quality genomic visualization for GWAS, TWAS, ExWAS, and PheWAS.**

<div class="badge-row">
  <img src="https://img.shields.io/pypi/v/neat-plots?color=purple" alt="PyPI">
  <img src="https://img.shields.io/pypi/pyversions/neat-plots?color=purple" alt="Python versions">
  <img src="https://img.shields.io/github/license/PMBB-Informatics-and-Genomics/NEAT-Plots" alt="License">
  <img src="https://img.shields.io/github/actions/workflow/status/PMBB-Informatics-and-Genomics/NEAT-Plots/docs.yml?label=docs" alt="Docs">
</div>

NEAT-Plots (**N**otation of **E**ffects **A**cross **T**raits) is a Python package that turns GWAS, TWAS, ExWAS, and PheWAS summary statistics into annotated, publication-ready figures with minimal code.

---

## What can it make?

=== "Manhattan plot"
    ![Manhattan plot — horizontal](assets/img/manhattan_horizontal.png)

=== "Vertical Manhattan"
    ![Manhattan plot — vertical](assets/img/manhattan_vertical.png)

=== "QQ plot"
    ![QQ plot](assets/img/qq_plot.png)

=== "Multi-series QQ"
    ![Multi-series QQ plot](assets/img/qq_multiseries.png)

=== "Boroughs / TWAS"
    ![TWAS boroughs plot](assets/img/boroughs_twas.png)

---

## Five-line quick start

```python
from neat_plots import ManhattanPlot

mp = ManhattanPlot("my_gwas.tsv.gz", title="My GWAS Study")
mp.prepare(col_map={"CHR": "#CHROM", "BP": "POS", "PVAL": "P"})
mp.update_plotting_parameters(sig=5e-8, merge_genes=True)
mp.full_plot(save="manhattan.png")
mp.qq_plot(save="qq.png")
```

---

## Key features

- **One-line pipeline** — `prepare()` handles load → clean → annotate → thin in a single call
- **Memory-efficient** — chunked loader (`prepare(chunked=True)`) for large files in HPC / Nextflow pipelines
- **Horizontal and vertical** Manhattan plots with automatic chromosome ticks and gene annotation tables
- **Faceted boroughs plots** for TWAS / multi-tissue results
- **Multi-series QQ plots** — overlay P-value distributions for subgroups or methods
- **Fully customizable** colors, thresholds, and point sizes via `update_plotting_parameters()` and JSON color palettes
- **Backward compatible** — existing scripts using `from manhattan_plot import ManhattanPlot` continue to work unchanged

---

## Installation

```bash
pip install neat-plots
```

See [Installation](installation.md) for conda, development install, and optional GUI extras.

---

## Citation

> Guare LA *et al.* NEAT-Plots: publication-quality genomic visualization for GWAS, TWAS, ExWAS, and PheWAS. *BioData Mining* (in preparation).
