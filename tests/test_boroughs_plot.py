"""
Tests for BoroughsPlot-specific functionality:
  WRAP validation, per-facet thinning, load_and_thin override.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neat_plots import BoroughsPlot


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# clean_data: WRAP column validation
# ---------------------------------------------------------------------------


class TestBoroughsCleanData:
    def test_missing_wrap_raises(self, gwas_tsv):
        """clean_data should raise ValueError when WRAP column is absent."""
        bp = BoroughsPlot(gwas_tsv)
        bp.load_data()
        with pytest.raises(ValueError, match="WRAP"):
            bp.clean_data()

    def test_with_wrap_column_passes(self, boroughs_tsv):
        bp = BoroughsPlot(boroughs_tsv)
        bp.load_data()
        bp.clean_data()   # should not raise
        assert "WRAP" in bp.df.columns

    def test_chr_prefix_stripped(self, tmp_path):
        """BoroughsPlot.clean_data should inherit the chr-prefix strip."""
        df = pd.DataFrame({
            "#CHROM": ["chr1", "chr2"],
            "POS":    [1000, 2000],
            "ID":     ["g1", "g2"],
            "P":      [0.01, 0.02],
            "WRAP":   ["T1", "T2"],
        })
        path = str(tmp_path / "bp_chr.tsv")
        df.to_csv(path, sep="\t", index=False)
        bp = BoroughsPlot(path)
        bp.load_data()
        bp.clean_data()
        assert set(bp.df["#CHROM"]) == {1, 2}


# ---------------------------------------------------------------------------
# get_thinned_data: WRAP in uniqueness key
# ---------------------------------------------------------------------------


class TestBoroughsThinning:
    def test_thinned_has_wrap_column(self, boroughs_tsv):
        bp = BoroughsPlot(boroughs_tsv)
        bp.prepare()
        assert "WRAP" in bp.thinned.columns

    def test_same_position_different_wrap_both_kept(self, tmp_path):
        """
        Two variants at the same genomic position but different WRAP values
        should both survive thinning.
        """
        df = pd.DataFrame({
            "#CHROM": [1, 1],
            "POS":    [100_000, 100_000],
            "ID":     ["rs1", "rs1"],
            "P":      [1e-5, 1e-5],
            "WRAP":   ["TissueA", "TissueB"],
        })
        path = str(tmp_path / "same_pos.tsv")
        df.to_csv(path, sep="\t", index=False)
        bp = BoroughsPlot(path)
        bp.load_data()
        bp.clean_data()
        bp.get_thinned_data()
        # Both rows should survive (different WRAP)
        assert len(bp.thinned) == 2

    def test_dedup_within_same_wrap(self, tmp_path):
        """
        Two variants at the same pixel (same ROUNDED_X *and* ROUNDED_Y) with the
        same WRAP — only the lowest-P representative should survive.

        We force identical ROUNDED_Y by choosing P-values whose -log10 both
        round to 5.00 (e.g. 1e-5 and 9.9e-6).
        """
        from neat_plots._constants import CHR_POS_ROUND
        pos = int(CHR_POS_ROUND)   # both positions fall in the same 50 kb bucket
        # -log10(1e-5) = 5.0  and  -log10(9.9e-6) ≈ 5.004 → both round to 5.0
        df = pd.DataFrame({
            "#CHROM": [1, 1],
            "POS":    [pos, pos + 1],
            "ID":     ["rs1", "rs2"],
            "P":      [9.9e-6, 1e-5],   # rs1 has lower P (9.9e-6 < 1e-5) → should be kept
            "WRAP":   ["TissueA", "TissueA"],
        })
        path = str(tmp_path / "dedup_within.tsv")
        df.to_csv(path, sep="\t", index=False)
        bp = BoroughsPlot(path)
        bp.load_data()
        bp.clean_data()
        bp.get_thinned_data()
        assert len(bp.thinned) == 1
        # sort_values(by="P") puts the lowest-P row first; drop_duplicates keeps it
        assert bp.thinned["ID"].iloc[0] == "rs1"


# ---------------------------------------------------------------------------
# load_and_thin override
# ---------------------------------------------------------------------------


class TestBoroughsLoadAndThin:
    def test_load_and_thin_validates_wrap(self, gwas_tsv):
        """
        BoroughsPlot.load_and_thin should raise if WRAP is absent.
        """
        bp = BoroughsPlot(gwas_tsv)
        with pytest.raises((ValueError, KeyError)):
            bp.load_and_thin()

    def test_load_and_thin_with_wrap(self, boroughs_tsv):
        bp = BoroughsPlot(boroughs_tsv)
        bp.load_and_thin()
        assert "WRAP" in bp.thinned.columns

    def test_prepare_chunked_boroughs(self, boroughs_tsv):
        bp = BoroughsPlot(boroughs_tsv)
        bp.prepare(chunked=True, chunksize=100)
        assert "WRAP" in bp.thinned.columns


# ---------------------------------------------------------------------------
# update_plotting_parameters (boroughs-specific constraints)
# ---------------------------------------------------------------------------


class TestBoroughsPlottingParameters:
    def test_always_horizontal(self, boroughs_tsv):
        """BoroughsPlot does not support vertical orientation."""
        bp = BoroughsPlot(boroughs_tsv)
        bp.prepare()
        bp.update_plotting_parameters(sig=5e-8)
        assert bp.plot_x_col == "ROUNDED_X"
        assert bp.plot_y_col == "ROUNDED_Y"

    def test_sig_threshold_updated(self, boroughs_tsv):
        bp = BoroughsPlot(boroughs_tsv)
        bp.prepare()
        bp.update_plotting_parameters(sig=1e-6)
        assert bp.sig == 1e-6
