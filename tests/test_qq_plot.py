"""
Tests for BasePlot.qq_plot, including multi-series support.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neat_plots import ManhattanPlot


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to prevent resource leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Single-series (backward-compatibility)
# ---------------------------------------------------------------------------


class TestQqPlotSingleSeries:
    def test_runs_without_error(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        mp.qq_plot()  # no save, no error

    def test_saves_png(self, gwas_tsv, tmp_path):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        out_png = str(tmp_path / "qq.png")
        mp.qq_plot(save=out_png)
        assert os.path.exists(out_png)

    def test_saves_csv_alongside_png(self, gwas_tsv, tmp_path):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        out_png = str(tmp_path / "qq.png")
        mp.qq_plot(save=out_png)
        assert os.path.exists(out_png.replace(".png", ".csv"))

    def test_csv_has_expected_columns(self, gwas_tsv, tmp_path):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        out_png = str(tmp_path / "qq.png")
        mp.qq_plot(save=out_png)
        csv_df = pd.read_csv(out_png.replace(".png", ".csv"), index_col=0)
        assert "Log P" in csv_df.columns
        assert "Log Exp P" in csv_df.columns

    def test_lambda_gc_printed(self, gwas_tsv, capsys):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        mp.qq_plot()
        out = capsys.readouterr().out
        assert "Lambda GC" in out

    def test_uses_all_p_values_in_chunked_mode(self, gwas_tsv, capsys):
        """
        In chunked mode qq_plot should use self._all_p_values (all raw P-values),
        not just the thinned rows stored in self.df.
        """
        mp = ManhattanPlot(gwas_tsv)
        mp.load_and_thin()
        # _all_p_values is a Series and holds at least as many entries as thinned
        assert isinstance(mp._all_p_values, pd.Series)
        assert len(mp._all_p_values) >= len(mp.thinned)
        # qq_plot should run without error and report Lambda GC
        mp.qq_plot()
        out = capsys.readouterr().out
        assert "Lambda GC" in out


# ---------------------------------------------------------------------------
# Multi-series
# ---------------------------------------------------------------------------


class TestQqPlotMultiSeries:
    def _make_extra_series(self):
        rng = np.random.default_rng(999)
        return {
            "Males":   pd.Series(rng.uniform(1e-7, 1.0, 500)),
            "Females": pd.Series(rng.uniform(1e-9, 1.0, 400)),
        }

    def test_runs_without_error(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        mp.qq_plot(additional_series=self._make_extra_series())

    def test_lambda_gc_per_series(self, gwas_tsv, capsys):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        mp.qq_plot(additional_series=self._make_extra_series())
        out = capsys.readouterr().out
        assert "Lambda GC [Primary]" in out
        assert "Lambda GC [Males]" in out
        assert "Lambda GC [Females]" in out

    def test_saves_primary_csv(self, gwas_tsv, tmp_path):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        out_png = str(tmp_path / "qq_multi.png")
        mp.qq_plot(save=out_png, additional_series=self._make_extra_series())
        assert os.path.exists(out_png.replace(".png", ".csv"))

    def test_empty_additional_series(self, gwas_tsv):
        """Passing an empty dict should behave like single-series."""
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        mp.qq_plot(additional_series={})  # no error, no extra scatter

    def test_single_additional_series(self, gwas_tsv, capsys):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        extra = {"Cohort B": pd.Series(np.random.uniform(1e-8, 1.0, 300))}
        mp.qq_plot(additional_series=extra)
        out = capsys.readouterr().out
        assert "Lambda GC [Cohort B]" in out
