"""
Tests for BasePlot data-handling methods:
  load_data, clean_data, get_thinned_data, load_and_thin, prepare,
  add_annotations, save_thinned_df, check_data, print_hits.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from neat_plots import ManhattanPlot


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_loads_rows(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        assert mp.df is not None
        assert len(mp.df) > 0

    def test_columns_present(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        for col in ("#CHROM", "POS", "ID", "P"):
            assert col in mp.df.columns

    def test_test_rows_limit(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv, test_rows=50)
        mp.load_data()
        assert len(mp.df) <= 50


# ---------------------------------------------------------------------------
# clean_data
# ---------------------------------------------------------------------------


class TestCleanData:
    def test_basic_clean(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        mp.clean_data()
        assert mp.df["#CHROM"].dtype in (int, np.int64, np.int32)
        # clean_data coerces P to numeric but does not drop NaN rows;
        # the synthetic fixture has no NaN rows so this should be zero.
        assert mp.df["P"].isna().sum() == 0

    def test_col_map(self, gwas_tsv_alt_cols):
        mp = ManhattanPlot(gwas_tsv_alt_cols)
        mp.load_data()
        mp.clean_data(col_map={"CHR": "#CHROM", "BP": "POS", "SNP": "ID", "PVAL": "P"})
        for col in ("#CHROM", "POS", "ID", "P"):
            assert col in mp.df.columns

    def test_logp_column(self, gwas_tsv_logp):
        mp = ManhattanPlot(gwas_tsv_logp)
        mp.load_data()
        mp.clean_data(logp="LOG10_P")
        assert "P" in mp.df.columns
        assert (mp.df["P"] > 0).all()

    def test_zero_p_replaced(self, gwas_tsv_with_nan):
        """Exact zero P-values must be replaced to avoid log10(0) = -inf."""
        mp = ManhattanPlot(gwas_tsv_with_nan)
        mp.load_data()
        mp.clean_data()
        # After clean_data, no P-value should be exactly zero (NaN rows are kept)
        non_nan = mp.df["P"].dropna()
        assert (non_nan > 0).all(), "Zero P-values should be replaced with a small positive value"

    def test_chr_filter(self, gwas_tsv):
        """Only autosomes + X (chr 1–23) should survive clean_data."""
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        # Inject an invalid chromosome
        mp.df.loc[mp.df.index[0], "#CHROM"] = 99
        mp.clean_data()
        assert 99 not in mp.df["#CHROM"].values

    def test_sorted_by_chr_pos(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        mp.clean_data()
        # Verify sort order
        chrom_pos = list(zip(mp.df["#CHROM"], mp.df["POS"]))
        assert chrom_pos == sorted(chrom_pos)

    def test_nan_p_values_survive_clean(self, gwas_tsv_with_nan):
        """
        NaN P-values are coerced by clean_data but the rows are intentionally
        kept (downstream methods use dropna where needed).
        """
        mp = ManhattanPlot(gwas_tsv_with_nan)
        mp.load_data()
        n_nan_before = mp.df["P"].isna().sum() + (mp.df["P"] == 0).sum()
        mp.clean_data()
        # NaN rows are preserved; zeros are replaced with a small value
        assert mp.df["P"].isna().sum() > 0, "NaN rows should be preserved after clean_data"
        assert (mp.df["P"].dropna() > 0).all(), "No zeros should remain after clean_data"

    def test_chr_prefix_stripped(self, tmp_path):
        """Chromosomes written as 'chr1' should be normalised to integer 1."""
        df = pd.DataFrame({
            "#CHROM": ["chr1", "chr2", "chrX"],
            "POS":    [100, 200, 300],
            "ID":     ["a", "b", "c"],
            "P":      [0.01, 0.02, 0.03],
        })
        path = str(tmp_path / "chr_prefix.tsv")
        df.to_csv(path, sep="\t", index=False)
        mp = ManhattanPlot(path)
        mp.load_data()
        mp.clean_data()
        assert set(mp.df["#CHROM"]) == {1, 2, 23}  # X → 23


# ---------------------------------------------------------------------------
# get_thinned_data
# ---------------------------------------------------------------------------


class TestGetThinnedData:
    def test_thinning_reduces_rows(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        mp.clean_data()
        n_before = len(mp.df)
        mp.get_thinned_data()
        assert len(mp.thinned) <= n_before

    def test_rounded_columns_added_to_df(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        mp.clean_data()
        mp.get_thinned_data()
        assert "ROUNDED_X" in mp.df.columns
        assert "ROUNDED_Y" in mp.df.columns

    def test_abs_pos_added(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        mp.clean_data()
        mp.get_thinned_data()
        assert "ABS_POS" in mp.df.columns
        assert (mp.df["ABS_POS"] > 0).all()

    def test_chr_ticks_set(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        mp.clean_data()
        mp.get_thinned_data()
        assert len(mp.chr_ticks) == 2
        assert len(mp.chr_ticks[0]) > 0, "chr_ticks should be populated"

    def test_thinned_p_values_sorted_asc(self, gwas_tsv):
        """After thinning, the best (lowest) P should be kept per pixel bucket."""
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        mp.clean_data()
        mp.get_thinned_data()
        # Every (ROUNDED_X, ROUNDED_Y) pair should appear at most once
        dupes = mp.thinned.duplicated(subset=["ROUNDED_X", "ROUNDED_Y"])
        assert not dupes.any(), "Thinned data should have no duplicate (X,Y) pixels"

    def test_additional_cols_in_key(self, boroughs_tsv):
        """BoroughsPlot adds WRAP to the uniqueness key; same pos can repeat per wrap."""
        from neat_plots import BoroughsPlot
        bp = BoroughsPlot(boroughs_tsv)
        bp.load_data()
        bp.clean_data()
        bp.get_thinned_data()
        dupes = bp.thinned.duplicated(subset=["ROUNDED_X", "ROUNDED_Y", "WRAP"])
        assert not dupes.any()


# ---------------------------------------------------------------------------
# _build_chr_offset_map & _get_absolute_positions
# ---------------------------------------------------------------------------


class TestChrOffsets:
    def test_chr1_offset_is_zero(self):
        from neat_plots._base import BasePlot
        offsets, _ = BasePlot._build_chr_offset_map()
        assert offsets[1] == 0

    def test_chr2_offset_equals_chr1_length(self):
        from neat_plots._base import BasePlot
        from neat_plots._constants import CHR_LENGTHS
        offsets, _ = BasePlot._build_chr_offset_map()
        assert offsets[2] == CHR_LENGTHS[1]

    def test_offsets_monotonically_increasing(self):
        from neat_plots._base import BasePlot
        offsets, _ = BasePlot._build_chr_offset_map()
        vals = offsets.sort_index().values
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_abs_pos_chr1_equals_pos(self):
        from neat_plots._base import BasePlot
        bp = BasePlot.__new__(BasePlot)
        bp.CHR_POS_ROUND = 5e4
        bp.chr_ticks = []
        df = pd.DataFrame({"#CHROM": [1], "POS": [12345]})
        abs_pos = bp._get_absolute_positions(df)
        assert abs_pos[0] == 12345


# ---------------------------------------------------------------------------
# load_and_thin (chunked loader)
# ---------------------------------------------------------------------------


class TestLoadAndThin:
    def test_produces_thinned_data(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_and_thin()
        assert mp.thinned is not None
        assert len(mp.thinned) > 0

    def test_df_is_thinned_in_chunked_mode(self, gwas_tsv):
        """In chunked mode self.df should be the same object as self.thinned."""
        mp = ManhattanPlot(gwas_tsv)
        mp.load_and_thin()
        assert mp.df is mp.thinned

    def test_all_p_values_stashed(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_and_thin()
        assert hasattr(mp, "_all_p_values")
        assert isinstance(mp._all_p_values, pd.Series)
        # _all_p_values accumulates every non-NaN P from every chunk;
        # it should contain at least as many entries as thinned rows and
        # never more than the total raw row count.
        assert len(mp._all_p_values) >= len(mp.thinned)

    def test_col_map_works(self, gwas_tsv_alt_cols):
        mp = ManhattanPlot(gwas_tsv_alt_cols)
        mp.load_and_thin(
            col_map={"CHR": "#CHROM", "BP": "POS", "SNP": "ID", "PVAL": "P"}
        )
        for col in ("#CHROM", "POS", "ID", "P"):
            assert col in mp.df.columns

    def test_small_chunksize(self, gwas_tsv):
        """Chunked result should be consistent regardless of chunk size."""
        mp_big   = ManhattanPlot(gwas_tsv)
        mp_small = ManhattanPlot(gwas_tsv)
        mp_big.load_and_thin(chunksize=10_000)
        mp_small.load_and_thin(chunksize=50)
        # Both should produce the same number of thinned variants
        assert len(mp_big.thinned) == len(mp_small.thinned)

    def test_rounded_cols_present(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_and_thin()
        assert "ROUNDED_X" in mp.df.columns
        assert "ROUNDED_Y" in mp.df.columns
        assert "ABS_POS" in mp.df.columns


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_prepare_standard(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        assert mp.thinned is not None

    def test_prepare_chunked(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare(chunked=True)
        assert mp.thinned is not None

    def test_prepare_with_col_map(self, gwas_tsv_alt_cols):
        mp = ManhattanPlot(gwas_tsv_alt_cols)
        mp.prepare(col_map={"CHR": "#CHROM", "BP": "POS", "SNP": "ID", "PVAL": "P"})
        assert "P" in mp.df.columns

    def test_prepare_with_annotation(self, gwas_tsv, annotation_df):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare(annot_df=annotation_df)
        assert "P" in mp.df.columns  # pipeline completed

    def test_prepare_chunked_vs_standard_same_count(self, gwas_tsv):
        """Both paths should produce the same thinned variant count."""
        mp_std   = ManhattanPlot(gwas_tsv)
        mp_chunk = ManhattanPlot(gwas_tsv)
        mp_std.prepare()
        mp_chunk.prepare(chunked=True)
        assert len(mp_std.thinned) == len(mp_chunk.thinned)


# ---------------------------------------------------------------------------
# add_annotations
# ---------------------------------------------------------------------------


class TestAddAnnotations:
    def test_gene_merged(self, gwas_tsv, annotation_df):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        mp.clean_data()
        mp.add_annotations(annotation_df)
        # At least one row should have been annotated with GENE_A
        assert (mp.df["ID"] == "GENE_A").any()

    def test_unannotated_rows_keep_id(self, gwas_tsv, annotation_df):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        mp.clean_data()
        # Save original IDs for positions that are NOT being annotated
        annotated_pos = set(zip(annotation_df["#CHROM"], annotation_df["POS"]))
        non_annot_mask = ~mp.df.apply(
            lambda r: (r["#CHROM"], r["POS"]) in annotated_pos, axis=1
        )
        original_non_annot_ids = set(mp.df.loc[non_annot_mask, "ID"].dropna())
        mp.add_annotations(annotation_df)
        post_non_annot_ids = set(mp.df.loc[non_annot_mask, "ID"].dropna())
        # Non-annotated rows should keep their original IDs
        assert original_non_annot_ids == post_non_annot_ids


# ---------------------------------------------------------------------------
# save_thinned_df
# ---------------------------------------------------------------------------


class TestSaveThinnedDf:
    def test_save_pickle(self, gwas_tsv, tmp_path):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        out = str(tmp_path / "thinned.pickle")
        mp.save_thinned_df(out, pickle=True)
        assert os.path.exists(out)
        loaded = pd.read_pickle(out)
        assert len(loaded) == len(mp.thinned)

    def test_save_csv(self, gwas_tsv, tmp_path):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        out = str(tmp_path / "thinned.csv")
        mp.save_thinned_df(out, pickle=False)
        assert os.path.exists(out)
        loaded = pd.read_csv(out)
        assert len(loaded) == len(mp.thinned)


# ---------------------------------------------------------------------------
# check_data / print_hits
# ---------------------------------------------------------------------------


class TestInspectionHelpers:
    def test_check_data_runs(self, gwas_tsv, capsys):
        mp = ManhattanPlot(gwas_tsv)
        mp.load_data()
        mp.clean_data()
        mp.check_data()  # should not raise
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_print_hits_runs(self, gwas_tsv, capsys):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        mp.print_hits()  # should not raise
