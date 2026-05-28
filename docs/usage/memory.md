# Memory-Efficient Loading

For large summary-statistics files (50M+ variants, common in whole-genome or multi-ancestry studies), the default pipeline can require 3–4× the file size in RAM during the `sort()` step inside `get_thinned_data()`. NEAT-Plots provides a chunked loading path that reduces peak RAM to approximately **1–2× the thinned result size** — typically a 10–50× reduction.

---

## The problem with large files

The standard pipeline:

```
load_data()         → self.df  (full file in RAM, ~1× file size)
clean_data()        → in-place operations, still ~1×
get_thinned_data()  → sort_values() creates a full sorted copy → peaks at ~2–3×
```

For a 10 GB summary-stats file, this can require 25–30 GB of RAM.

---

## Chunked loading with `prepare(chunked=True)`

```python
mp = ManhattanPlot("mega_gwas_50M_variants.tsv.gz", title="Mega GWAS")
mp.prepare(
    col_map={"CHR": "#CHROM", "BP": "POS", "PVAL": "P"},
    chunked=True,
    chunksize=500_000,    # tune to your available RAM
)
```

The algorithm:

1. Read `chunksize` rows at a time
2. Clean and thin each chunk independently
3. Concatenate only the thinned rows (typically 1–5% of the original)
4. One final global de-duplication pass
5. Store all raw P-values in `self._all_p_values` so `qq_plot()` remains accurate

After this call, `self.df` and `self.thinned` both point to the small thinned result — the full file is **never assembled in RAM**.

---

## Memory scaling

| Variants | Thinned (typical) | Standard peak RAM | Chunked peak RAM |
|----------|------------------|-------------------|-----------------|
| 1M       | ~50k             | ~3 GB             | ~0.2 GB         |
| 10M      | ~200k            | ~30 GB            | ~0.5 GB         |
| 50M      | ~500k            | ~150 GB           | ~1 GB           |

*Estimates assume ~200 bytes/variant in memory. Actual usage varies with column count.*

---

## Choosing chunksize

A chunk of 500,000 rows with ~10 columns uses approximately 400 MB RAM. Adjust based on your system:

| Available RAM | Recommended chunksize |
|--------------|----------------------|
| 8 GB         | 200,000              |
| 16 GB        | 500,000              |
| 32 GB+       | 1,000,000            |

---

## Nextflow / HPC usage

In a Nextflow pipeline you can run NEAT-Plots inside a process with a much smaller memory allocation:

```nextflow
process plot_manhattan {
    memory '8 GB'

    script:
    """
    python - << 'EOF'
    from neat_plots import ManhattanPlot
    mp = ManhattanPlot("${sumstats}", title="${params.title}")
    mp.prepare(
        col_map={"CHR": "#CHROM", "BP": "POS", "PVAL": "P"},
        chunked=True,
        chunksize=200_000,
    )
    mp.update_plotting_parameters(sig=5e-8, merge_genes=True)
    mp.full_plot(save="manhattan.png")
    mp.qq_plot(save="qq.png")
    EOF
    """
}
```

---

## QQ plot accuracy in chunked mode

`qq_plot()` automatically uses `self._all_p_values` (every non-NaN P-value collected during chunked loading) when it is available, so the QQ plot is based on the full distribution — not just the thinned subset:

```python
mp.prepare(chunked=True)
mp.qq_plot(save="qq.png")   # uses all 50M P-values internally
```

---

## Saving and reloading the thinned result

Once thinned, save the result to avoid re-running the chunked load:

```python
mp.save_thinned_df("thinned.pickle")

# Later — reload for styling iterations
mp2 = ManhattanPlot("thinned.pickle", title="My GWAS")
mp2.load_data()
mp2.get_thinned_data()   # already thinned, instant
mp2.full_plot(save="manhattan_v2.png")
```
