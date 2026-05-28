# ExWAS Plots

Exome-wide association studies (ExWAS) follow the same API as GWAS Manhattan plots, with two common patterns: **singles** (one variant per row) and **regions** (gene-level results grouped by annotation or MAF).

## ExWAS singles

```python
from neat_plots import ManhattanPlot

mp = ManhattanPlot("exwas_singles.tsv.gz", title="ExWAS — Rare LoF Variants")
mp.prepare(
    col_map={"CHR": "#CHROM", "BP": "POS", "SNP": "ID", "P": "P"}
)

# Bonferroni threshold based on number of tests
n_tests   = len(mp.df)
threshold = 0.05 / n_tests
mp.update_plotting_parameters(
    sig=threshold,
    sug=threshold * 10,
    merge_genes=True,
    vertical=False,
)
mp.full_plot(save="exwas_singles.png")
mp.qq_plot(save="exwas_qq.png")
```

---

## ExWAS regions (gene-level, grouped by annotation × MAF)

For gene-level ExWAS where results are pre-grouped by annotation category and MAF bin, use `BoroughsPlot` with a `WRAP` column encoding the group:

```python
from neat_plots import BoroughsPlot

bp = BoroughsPlot("exwas_regions.tsv.gz", title="ExWAS — Gene-level")
bp.prepare(
    col_map={
        "CHR":        "#CHROM",
        "POS":        "POS",
        "GENE":       "ID",
        "P_VALUE":    "P",
        "ANNOT_MAF":  "WRAP",     # e.g. "LoF_rare", "Missense_common"
    }
)
bp.update_plotting_parameters(sig=2.5e-6, merge_genes=True)
bp.full_plot(save="exwas_regions.png", legend_loc="top")
```

---

## Looping over multiple annotation × MAF combinations

The typical ExWAS pipeline generates one plot per annotation/MAF combination. The `prepare()` call handles all the data wrangling each iteration:

```python
from neat_plots import ManhattanPlot
import pandas as pd

full_results = pd.read_table("exwas_all.tsv.gz")
annotations  = full_results["ANNOTATION"].unique()

for annot in annotations:
    subset_path = f"/tmp/exwas_{annot}.tsv"
    full_results[full_results["ANNOTATION"] == annot].to_csv(
        subset_path, sep="\t", index=False
    )

    mp = ManhattanPlot(subset_path, title=f"ExWAS — {annot}")
    mp.prepare()

    n_tests   = len(mp.df)
    threshold = 0.05 / n_tests
    mp.update_plotting_parameters(sig=threshold, merge_genes=True)
    mp.full_plot(save=f"exwas_{annot}.png", with_table=True)
    mp.qq_plot(save=f"exwas_{annot}_qq.png")
```

---

## Coloring by effect direction

For ExWAS results that include effect sizes, color significant signals by direction:

```python
mp.update_plotting_parameters(signal_color_col="BETA")
```

For categorical coloring (e.g. protective vs. risk), use any string column and NEAT-Plots will assign discrete colors automatically.

---

## Signal plot (zoomed loci only)

To show only the significant loci without the full-genome background:

```python
mp.signal_plot(save="exwas_signals.png")
```

This creates a compact figure with each signal as a separate panel, sorted by genomic position — useful for supplement figures.
