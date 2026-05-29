# NEAT-Plots

**Notation of Effects Across Traits** — publication-quality genomic visualization for GWAS, TWAS, ExWAS, and PheWAS.

📖 **Full documentation: [PMBB-Informatics-and-Genomics.github.io/NEAT-Plots](https://PMBB-Informatics-and-Genomics.github.io/NEAT-Plots)**

---

## Quick install

```bash
pip install neat-plots
```

## Quick start

```python
from neat_plots import ManhattanPlot

mp = ManhattanPlot("my_gwas.tsv.gz", title="My GWAS Study")
mp.prepare(col_map={"CHR": "#CHROM", "BP": "POS", "PVAL": "P"})
mp.update_plotting_parameters(sig=5e-8)
mp.full_plot(save="manhattan.png")
mp.qq_plot(save="qq.png")
```

See the [documentation site](https://PMBB-Informatics-and-Genomics.github.io/NEAT-Plots) for the full usage guide, gallery, and API reference.

## Citation

> Guare LA *et al.* NEAT-Plots: publication-quality genomic visualization for GWAS, TWAS, ExWAS, and PheWAS. *BioData Mining* (in preparation).
