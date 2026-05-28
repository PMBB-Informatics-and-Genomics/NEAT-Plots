"""
Tests for ManhattanPlot-specific functionality:
  update_plotting_parameters, full_plot (headless), signal/color helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neat_plots import ManhattanPlot


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_title(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        assert mp.title == "Manhattan Plot"

    def test_custom_title(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv, title="My GWAS")
        assert mp.title == "My GWAS"

    def test_test_rows(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv, test_rows=20)
        mp.load_data()
        assert len(mp.df) <= 20


# ---------------------------------------------------------------------------
# update_plotting_parameters
# ---------------------------------------------------------------------------


class TestUpdatePlottingParameters:
    def _prepared_mp(self, gwas_tsv):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        return mp

    def test_sig_updated(self, gwas_tsv):
        mp = self._prepared_mp(gwas_tsv)
        mp.update_plotting_parameters(sig=1e-5)
        assert mp.sig == 1e-5

    def test_sug_updated(self, gwas_tsv):
        mp = self._prepared_mp(gwas_tsv)
        mp.update_plotting_parameters(sug=1e-4)
        assert mp.sug == 1e-4

    def test_vertical_sets_axis_cols(self, gwas_tsv):
        mp = self._prepared_mp(gwas_tsv)
        mp.update_plotting_parameters(vertical=True)
        assert mp.plot_x_col == "ROUNDED_Y"
        assert mp.plot_y_col == "ROUNDED_X"

    def test_horizontal_sets_axis_cols(self, gwas_tsv):
        mp = self._prepared_mp(gwas_tsv)
        mp.update_plotting_parameters(vertical=False)
        assert mp.plot_x_col == "ROUNDED_X"
        assert mp.plot_y_col == "ROUNDED_Y"

    def test_max_log_p(self, gwas_tsv):
        mp = self._prepared_mp(gwas_tsv)
        mp.update_plotting_parameters(max_log_p=50.0)
        assert mp.max_log_p == 50.0

    def test_merge_genes_flag(self, gwas_tsv):
        mp = self._prepared_mp(gwas_tsv)
        mp.update_plotting_parameters(merge_genes=True)
        assert mp.merge_genes is True

    def test_title_update(self, gwas_tsv):
        mp = self._prepared_mp(gwas_tsv)
        mp.update_plotting_parameters(title="Updated Title")
        assert mp.title == "Updated Title"


# ---------------------------------------------------------------------------
# Color configuration
# ---------------------------------------------------------------------------


class TestColorConfig:
    def test_reset_colors(self, gwas_tsv):
        from neat_plots._constants import DEFAULT_COLORS
        mp = ManhattanPlot(gwas_tsv)
        mp.DARK_CHR_COLOR = "#000000"
        mp.reset_colors()
        assert mp.DARK_CHR_COLOR == DEFAULT_COLORS["DARK_CHR_COLOR"]

    def test_config_colors_from_json(self, gwas_tsv, tmp_path):
        import json
        colors = {"DARK_CHR_COLOR": "#AABBCC", "LIGHT_CHR_COLOR": "#DDEEFF"}
        json_path = str(tmp_path / "colors.json")
        with open(json_path, "w") as fh:
            json.dump(colors, fh)
        mp = ManhattanPlot(gwas_tsv)
        mp.config_colors(json_path)
        assert mp.DARK_CHR_COLOR  == "#AABBCC"
        assert mp.LIGHT_CHR_COLOR == "#DDEEFF"


# ---------------------------------------------------------------------------
# full_plot (smoke test — headless, no save)
# ---------------------------------------------------------------------------


class TestFullPlot:
    def _mp(self, gwas_tsv, sig=5e-8):
        """Helper: prepared ManhattanPlot with threshold guaranteed to find signals."""
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        # The synthetic fixture plants P ~ 1e-15 .. 1e-8 signals per chromosome;
        # use sig=5e-8 which they all satisfy.
        mp.update_plotting_parameters(sig=sig, merge_genes=False)
        return mp

    def test_full_plot_runs(self, gwas_tsv):
        mp = self._mp(gwas_tsv)
        mp.full_plot()  # should not raise

    def test_full_plot_saves_png(self, gwas_tsv, tmp_path):
        import os
        mp = self._mp(gwas_tsv)
        out = str(tmp_path / "manhattan.png")
        mp.full_plot(save=out)
        assert os.path.exists(out)

    def test_full_plot_no_table(self, gwas_tsv):
        """with_table=False should work for both vertical and horizontal orientations."""
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        mp.update_plotting_parameters(vertical=True)
        mp.full_plot(with_table=False)   # vertical + no table (new branch)
        plt.close("all")
        mp.update_plotting_parameters(vertical=False)
        mp.full_plot(with_table=False)   # horizontal + no table

    def test_check_plotting_parameters(self, gwas_tsv, capsys):
        mp = ManhattanPlot(gwas_tsv)
        mp.prepare()
        mp.check_plotting_parameters()
        out = capsys.readouterr().out
        assert "Significance Threshold" in out


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_shim_import_same_class(self):
        """The manhattan-plot shim must export the exact same class."""
        from manhattan_plot import ManhattanPlot as ManhattanPlotShim
        assert ManhattanPlotShim is ManhattanPlot

    def test_shim_boroughs_same_class(self):
        from neat_plots import BoroughsPlot
        from manhattan_plot import BoroughsPlot as BoroughsPlotShim
        assert BoroughsPlotShim is BoroughsPlot
