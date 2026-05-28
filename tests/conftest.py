"""
Shared pytest fixtures for NEAT-Plots tests.

All fixtures use fully synthetic data so the test suite runs offline without
any large files.  A small real-data integration test lives in
``tests/integration/`` (Task 13).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make the backward-compatibility shim importable as ``manhattan_plot``
# ---------------------------------------------------------------------------
_repo_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "manhattan-plot"))

# ---------------------------------------------------------------------------
# Deterministic RNG
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=12345)


# ---------------------------------------------------------------------------
# Raw DataFrame builders
# ---------------------------------------------------------------------------


def _make_gwas_df(
    n_per_chr: int = 300,
    chrs: list = None,
) -> pd.DataFrame:
    """
    Build a clean synthetic GWAS summary-statistics DataFrame.

    Columns: #CHROM, POS, ID, P
    One genome-wide-significant variant (P < 5e-8) is planted per chromosome
    at index 1 of the per-chromosome block (avoids the every-50th-row slots
    used in NaN-handling tests).
    """
    if chrs is None:
        chrs = list(range(1, 7))  # chromosomes 1–6

    rng = np.random.default_rng(seed=42)
    rows = []
    for c in chrs:
        positions = sorted(rng.integers(10_000, 200_000_000, n_per_chr).tolist())
        p_vals = rng.uniform(1e-4, 1.0, n_per_chr).tolist()
        # Plant one strong signal at index 1 (safe from every-50th-row injection)
        p_vals[1] = float(10 ** rng.uniform(-15, -8))
        for i, (pos, p) in enumerate(zip(positions, p_vals)):
            rows.append({"#CHROM": c, "POS": int(pos), "ID": f"rs{c}_{i:04d}", "P": float(p)})

    return pd.DataFrame(rows)


def _make_boroughs_df(n_per_chr: int = 200, chrs: list = None) -> pd.DataFrame:
    """Like _make_gwas_df but with a WRAP column required by BoroughsPlot."""
    if chrs is None:
        chrs = list(range(1, 5))
    wrap_labels = ["Tissue_A", "Tissue_B", "Tissue_A", "Tissue_B"]
    df = _make_gwas_df(n_per_chr=n_per_chr, chrs=chrs)
    df["WRAP"] = [wrap_labels[i % len(wrap_labels)] for i in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# Fixtures: TSV files on disk
# ---------------------------------------------------------------------------


@pytest.fixture
def gwas_tsv(tmp_path) -> str:
    """Write synthetic GWAS data to a temp TSV and return the path."""
    path = str(tmp_path / "gwas.tsv")
    _make_gwas_df().to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def gwas_tsv_alt_cols(tmp_path) -> str:
    """
    GWAS TSV with non-standard column names (CHR, BP, SNP, PVAL).
    Tests that col_map renaming works correctly.
    """
    df = _make_gwas_df().rename(
        columns={"#CHROM": "CHR", "POS": "BP", "ID": "SNP", "P": "PVAL"}
    )
    path = str(tmp_path / "gwas_altcols.tsv")
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def gwas_tsv_logp(tmp_path) -> str:
    """GWAS TSV with LOG10_P column instead of raw P."""
    df = _make_gwas_df()
    df["LOG10_P"] = -np.log10(df["P"])
    df = df.drop(columns="P")
    path = str(tmp_path / "gwas_logp.tsv")
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def gwas_tsv_with_nan(tmp_path) -> str:
    """GWAS TSV that contains NaN and zero P-values for edge-case tests."""
    df = _make_gwas_df()
    df.loc[df.index[::50], "P"] = float("nan")   # every 50th row → NaN
    df.loc[df.index[::75], "P"] = 0.0            # every 75th row → zero
    path = str(tmp_path / "gwas_nan.tsv")
    df.to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def boroughs_tsv(tmp_path) -> str:
    """Write synthetic boroughs data (with WRAP column) to a temp TSV."""
    path = str(tmp_path / "boroughs.tsv")
    _make_boroughs_df().to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture
def annotation_df(gwas_tsv, tmp_path) -> pd.DataFrame:
    """
    Gene-annotation table whose positions are guaranteed to exist in gwas_tsv.

    We load the first two rows of the prepared GWAS data and use their exact
    (CHR, POS) values so that add_annotations() will actually find matches.
    """
    import matplotlib
    matplotlib.use("Agg")
    from neat_plots import ManhattanPlot
    mp = ManhattanPlot(gwas_tsv)
    mp.load_data()
    mp.clean_data()
    first_rows = mp.df.head(3)[["#CHROM", "POS"]].copy()
    first_rows["ID"] = ["GENE_A", "GENE_B", "GENE_C"]
    return first_rows.reset_index(drop=True)
