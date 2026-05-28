# Customization

## Plotting parameters

All visual parameters are set via `update_plotting_parameters()`. You can call it multiple times — only the keys you pass are updated.

```python
mp.update_plotting_parameters(
    sig          = 5e-8,       # genome-wide significance threshold
    sug          = 1e-5,       # suggestive threshold
    annot_thresh = 5e-8,       # P-value cutoff for annotation table
    ld_block     = 4e5,        # LD window half-width (bp) for signal merging
    merge_genes  = True,       # collapse nearby signals into one annotation
    annotate     = True,       # show the annotation table
    invert       = False,      # flip the y-axis (for paired upper/lower plots)
    max_log_p    = None,       # cap the y-axis at this −log₁₀(P) value
    vertical     = False,      # False = horizontal, True = vertical
    title        = "My GWAS",
)
```

---

## Color palette

### Checking current colors

```python
mp.check_plotting_parameters()
```

### Resetting to defaults

```python
mp.reset_colors()
```

### Changing individual colors

Set any color attribute directly on the instance:

```python
mp.DARK_CHR_COLOR    = "#2d6a8a"   # odd chromosomes
mp.LIGHT_CHR_COLOR   = "#6db0cf"   # even chromosomes
mp.NOVEL_HIT_COLOR   = "#c0392b"   # novel significant hits
mp.REP_HIT_COLOR     = "#e67e22"   # replicated hits (gold)
mp.FIFTH_COLOR       = "#8e44ad"   # significance line
```

### Loading a palette from JSON

Create a JSON file with any subset of color keys:

```json
{
    "DARK_CHR_COLOR":  "#2d6a8a",
    "LIGHT_CHR_COLOR": "#6db0cf",
    "NOVEL_HIT_COLOR": "#c0392b",
    "REP_HIT_COLOR":   "#e67e22",
    "COLOR_MAP":       "plasma"
}
```

Then load it:

```python
mp.config_colors("my_palette.json")
```

---

## Color map for continuous columns

When using `signal_color_col` with a numeric column, the color map is controlled by `COLOR_MAP` (any matplotlib colormap name):

```python
mp.COLOR_MAP = "viridis"
mp.update_plotting_parameters(signal_color_col="BETA")
```

---

## Point size

```python
mp.MIN_PT_SZ  = 5    # minimum scatter point size
mp.MAX_PT_SZ  = 200  # maximum scatter point size (for size-mapped columns)
```

---

## Full parameter reference

See the [API reference for ManhattanPlot](../api/manhattan_plot.md) for all accepted keyword arguments and their types.
