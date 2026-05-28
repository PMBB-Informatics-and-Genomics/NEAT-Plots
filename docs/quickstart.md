# Quick Start

## Minimal GWAS example

The fastest path from summary statistics to a publication figure:

```python
from neat_plots import ManhattanPlot

mp = ManhattanPlot("sumstats.tsv.gz", title="My GWAS")
mp.prepare(col_map={"CHR": "#CHROM", "BP": "POS", "PVAL": "P"})
mp.update_plotting_parameters(sig=5e-8, merge_genes=True)
mp.full_plot(save="manhattan.png")
mp.qq_plot(save="qq.png")
```

That's it. `prepare()` handles reading, cleaning, and thinning the data in one call. `full_plot()` draws the scatter, highlights significant signals, and adds the annotation table.

---

## Column names

NEAT-Plots expects three core columns internally:

| Internal name | Meaning |
|---|---|
| `#CHROM` | Chromosome (integer 1–23; X is recoded to 23) |
| `POS` | Base-pair position (GRCh37 or GRCh38) |
| `P` | P-value (raw, not −log₁₀) |
| `ID` | Variant / gene identifier (used for annotations) |

If your file uses different names, pass a `col_map`:

```python
mp.prepare(col_map={
    "CHR":    "#CHROM",
    "BP":     "POS",
    "SNP":    "ID",
    "P_META": "P",
})
```

If your file stores −log₁₀(P) instead of raw P, use the `logp` argument:

```python
mp.prepare(logp="LOG10P")
```

---

## The standard workflow in detail

```python
from neat_plots import ManhattanPlot

# 1. Construct — just stores the path
mp = ManhattanPlot("sumstats.tsv.gz", title="My GWAS")

# 2. Load — reads the file into mp.df
mp.load_data()

# 3. Clean — renames columns, filters to chr 1-23, fixes zeros
mp.clean_data(col_map={"CHR": "#CHROM", "BP": "POS", "PVAL": "P"})

# 4. (Optional) Annotate — merge gene names
mp.add_annotations(annot_df)

# 5. Thin — de-duplicates on rounded (x, y) pixel coordinates
mp.get_thinned_data()

# 6. Configure — set thresholds, orientation, colors, etc.
mp.update_plotting_parameters(
    sig=5e-8,
    sug=1e-5,
    merge_genes=True,
    vertical=False,          # horizontal (default) or vertical
)

# 7. Plot
mp.full_plot(save="manhattan.png")
mp.qq_plot(save="qq.png")
```

Steps 2–5 can be collapsed into a single `prepare()` call — see [One-line pipeline](#one-line-pipeline) below.

---

## One-line pipeline

```python
mp.prepare(
    col_map={"CHR": "#CHROM", "BP": "POS", "PVAL": "P"},
    annot_df=my_gene_table,         # optional
)
```

For very large files (50M+ variants), use the memory-efficient chunked mode:

```python
mp.prepare(
    col_map={"CHR": "#CHROM", "BP": "POS", "PVAL": "P"},
    chunked=True,
    chunksize=500_000,
)
```

See [Memory-Efficient Loading](usage/memory.md) for details.

---

## Next steps

- [GWAS Manhattan & QQ](usage/manhattan.md) — all Manhattan and QQ options
- [TWAS / Boroughs Plot](usage/boroughs.md) — multi-tissue faceted plots
- [ExWAS Plots](usage/exwas.md) — ExWAS singles and region plots
- [Customization](usage/customization.md) — colors, thresholds, fonts
