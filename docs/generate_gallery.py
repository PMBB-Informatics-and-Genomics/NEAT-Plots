"""
Regenerate gallery PNGs from the synthetic chr22 integration fixture.

Run from the repo root:

    python docs/generate_gallery.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import sys
import os
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from neat_plots import ManhattanPlot, BoroughsPlot
from neat_plots._base import BasePlot

IMG  = pathlib.Path(__file__).parent / "assets" / "img"
GWAS = pathlib.Path(__file__).parent.parent / "tests" / "integration" / "chr22_synthetic_gwas.tsv.gz"


def make_horizontal(path=GWAS, out=IMG / "manhattan_horizontal.png"):
    plt.close("all")
    mp = ManhattanPlot(str(path), title="Chr22 GWAS — Horizontal")
    mp.prepare()
    mp.update_plotting_parameters(vertical=False, sig=5e-8, merge_genes=True)
    mp.full_plot(save=str(out), save_res=120)
    plt.close("all")
    print(f"  {out.name}")


def make_vertical(path=GWAS, out=IMG / "manhattan_vertical.png"):
    plt.close("all")
    mp = ManhattanPlot(str(path), title="Chr22 GWAS — Vertical")
    mp.prepare()
    mp.update_plotting_parameters(vertical=True, sig=5e-8, merge_genes=True)
    mp.full_plot(save=str(out), save_res=120)
    plt.close("all")
    print(f"  {out.name}")


def make_qq(path=GWAS, out=IMG / "qq_plot.png"):
    plt.close("all")
    mp = ManhattanPlot(str(path), title="Chr22 GWAS")
    mp.prepare()
    mp.qq_plot(save=str(out), save_res=120)
    plt.close("all")
    print(f"  {out.name}")


def make_qq_multi(path=GWAS, out=IMG / "qq_multiseries.png"):
    plt.close("all")
    mp = ManhattanPlot(str(path))
    mp.prepare()
    pos_p = mp.df.loc[mp.df["BETA"] > 0, "P"].dropna()
    neg_p = mp.df.loc[mp.df["BETA"] < 0, "P"].dropna()
    mp.qq_plot(
        save=str(out), save_res=120,
        additional_series={"Positive effect (β > 0)": pos_p,
                           "Negative effect (β < 0)": neg_p},
    )
    plt.close("all")
    print(f"  {out.name}")


def make_boroughs(out=IMG / "boroughs_twas.png"):
    plt.close("all")
    rng = np.random.default_rng(seed=999)
    _, lengths = BasePlot._build_chr_offset_map()
    tissues = ["Brain", "Liver", "Kidney", "Muscle"]
    rows = []
    for c in range(1, 4):
        for t in tissues:
            for i in range(150):
                pos = int(rng.integers(1_000_000, lengths[c] - 1_000_000))
                p   = float(10 ** rng.uniform(-12 if i == 0 else -4, 0))
                rows.append({"#CHROM": c, "POS": pos, "ID": f"GENE_{c}_{i}",
                             "P": p, "WRAP": t, "BETA": float(rng.normal(0, 0.3))})
    df_b = pd.DataFrame(rows)
    tf = tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w")
    df_b.to_csv(tf, sep="\t", index=False)
    tf.close()

    bp = BoroughsPlot(tf.name, title="TWAS — Multi-tissue Boroughs")
    bp.load_data(); bp.clean_data(); bp.get_thinned_data()
    bp.update_plotting_parameters(sig=5e-8, signal_color_col="BETA", merge_genes=True)
    bp.facets = sorted(bp.thinned["WRAP"].unique())
    bp.full_plot(save=str(out), save_res=120, legend_loc="top")
    plt.close("all")
    os.unlink(tf.name)
    print(f"  {out.name}")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    print("Generating gallery figures...")
    make_horizontal()
    make_vertical()
    make_qq()
    make_qq_multi()
    make_boroughs()
    print("Done.")
