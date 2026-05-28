"""
BoroughsPlot — faceted (multi-panel) Manhattan plots.

Each "borough" is one panel driven by a ``WRAP`` column in the input data.
Inherits all data-handling logic from BasePlot; overrides thinning,
parameter handling, and the entire rendering pipeline for multi-axis layouts.

Typical usage
-------------
::

    from neat_plots import BoroughsPlot

    bp = BoroughsPlot("twas_sumstats.txt.gz", title="TWAS by Tissue")
    bp.load_data()
    bp.clean_data(col_map={"CHR": "#CHROM", "BP": "POS", "P_VALUE": "P"})
    bp.get_thinned_data()
    bp.update_plotting_parameters(
        sig=5e-8, signal_color_col="TISSUE", merge_genes=True
    )
    bp.full_plot(save="boroughs.png", legend_loc="top")
"""

from __future__ import annotations

from typing import Optional

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch

from ._base import BasePlot
from ._constants import DEFAULT_TABLE_FONTSIZE

# noinspection SpellCheckingInspection
class BoroughsPlot(BasePlot):
    """
    Faceted Manhattan plot.

    Requires a ``WRAP`` column in the input data that identifies which
    panel (borough) each variant belongs to.

    Parameters
    ----------
    file_path : str
        Path to the summary statistics file.
    test_rows : int, optional
        Limit the number of rows loaded.
    title : str
        Plot title.
    """

    # Boroughs-specific state
    facets:      list = []
    facet_count: Optional[int] = None

    # BoroughsPlot is always horizontal (no vertical option)
    # plot_x_col / plot_y_col are fixed in update_plotting_parameters
    plot_x_col: str = "ROUNDED_X"
    plot_y_col: str = "ROUNDED_Y"

    def __init__(
        self,
        file_path: str,
        test_rows: Optional[int] = None,
        title: str = "Manhattan Plot",
    ) -> None:
        super().__init__(file_path, test_rows=test_rows, title=title)

    # ------------------------------------------------------------------
    # Overrides: data handling
    # ------------------------------------------------------------------

    def clean_data(
        self,
        col_map: Optional[dict] = None,
        logp: Optional[str] = None,
        has_chr_prefix: bool = False,
    ) -> None:
        """
        Normalise the loaded DataFrame for boroughs plotting.

        Identical to :meth:`BasePlot.clean_data` but additionally validates
        that a ``WRAP`` column is present (required for faceting).

        Parameters
        ----------
        col_map:
            Column rename map (``#CHROM``, ``POS``, ``ID``, ``P``).
        logp:
            Column in ``-log10(p)`` scale; derives ``P`` when supplied.
        has_chr_prefix:
            Set ``True`` if the chromosome column contains a ``chr`` prefix
            that should be stripped *before* the standard normalisation.
            (The base class always strips the prefix automatically; this
            parameter is kept for API compatibility.)
        """
        # The base class already strips 'chr' unconditionally, so
        # has_chr_prefix is effectively a no-op here — preserved for
        # call-site compatibility.
        super().clean_data(col_map=col_map, logp=logp)

        if "WRAP" not in self.df.columns:
            raise ValueError("WRAP column must be present for BoroughsPlot")

    def get_thinned_data(
        self,
        log_p_round: int = 2,
        additional_cols: list = [],
    ) -> None:
        """
        Thin data for display; includes ``WRAP`` in the de-duplication key
        so that the same genomic position can appear in multiple boroughs.
        """
        super().get_thinned_data(
            log_p_round=log_p_round,
            additional_cols=["WRAP"] + list(additional_cols),
        )

    def load_and_thin(
        self,
        col_map=None,
        logp=None,
        log_p_round: int = 2,
        additional_cols: list = [],
        chunksize: int = 500_000,
        delim: str = r"\s+",
    ) -> None:
        """
        Chunked load + thin for boroughs data.

        Delegates to :meth:`BasePlot.load_and_thin` with ``WRAP`` added to
        the de-duplication key, then validates that the assembled result
        contains a ``WRAP`` column.

        Parameters mirror :meth:`BasePlot.load_and_thin`.
        """
        super().load_and_thin(
            col_map=col_map,
            logp=logp,
            log_p_round=log_p_round,
            additional_cols=["WRAP"] + list(additional_cols),
            chunksize=chunksize,
            delim=delim,
        )
        if "WRAP" not in self.df.columns:
            raise ValueError("WRAP column must be present for BoroughsPlot")

    # ------------------------------------------------------------------
    # Overrides: parameter management
    # ------------------------------------------------------------------

    def update_plotting_parameters(
        self,
        annotate="",
        signal_color_col="",
        twas_color_col="",
        twas_updown_col="",
        sig="",
        sug="",
        annot_thresh="",
        ld_block="",
        max_log_p="",
        invert="",
        merge_genes="",
        title="",
    ) -> None:
        """
        Update plotting parameters.

        BoroughsPlot does not support ``vertical``, ``log_p_axis_midpoint``,
        or the ``phewas_*`` columns — it is always horizontal.

        Parameters
        ----------
        sig, sug, annot_thresh : float
            P-value thresholds.
        ld_block : float
            LD window half-width (bp).
        merge_genes : bool
        invert : bool
        max_log_p : float or None
        signal_color_col, twas_color_col, twas_updown_col : str or None
        annotate : bool
        title : str
        """
        super().update_plotting_parameters(
            annotate=annotate,
            signal_color_col=signal_color_col,
            twas_color_col=twas_color_col,
            twas_updown_col=twas_updown_col,
            sig=sig, sug=sug, annot_thresh=annot_thresh,
            ld_block=ld_block, max_log_p=max_log_p,
            invert=invert, merge_genes=merge_genes, title=title,
        )
        # Boroughs is always horizontal
        self.plot_x_col = "ROUNDED_X"
        self.plot_y_col = "ROUNDED_Y"

    # ------------------------------------------------------------------
    # High-level plot entry points
    # ------------------------------------------------------------------

    def plot_data(self, with_table: bool = True, legend_loc=None) -> None:
        """Scatter all thinned data onto each borough axis."""
        self._config_axes(with_table=with_table, legend_loc=legend_loc)
        odds_list, evens_list = self._get_odds_evens()

        for i, b in enumerate(self.base_ax):
            odds  = odds_list[i]
            evens = evens_list[i]

            b.set_xticks(self.chr_ticks[0])
            b.set_xticklabels(self.chr_ticks[1])
            if self.invert:
                b.xaxis.set_label_position("top")
                b.xaxis.tick_top()

            if self.signal_color_col is None and self.twas_color_col is None:
                b.scatter(odds[self.plot_x_col],  odds[self.plot_y_col],  c=self.LIGHT_CHR_COLOR, s=2)
                b.scatter(evens[self.plot_x_col], evens[self.plot_y_col], c=self.DARK_CHR_COLOR,  s=2)
            else:
                b.scatter(odds[self.plot_x_col],  odds[self.plot_y_col],  edgecolors="silver",  s=2)
                b.scatter(evens[self.plot_x_col], evens[self.plot_y_col], edgecolors="dimgray", s=2)

        self._add_threshold_ticks()
        self._cosmetic_axis_edits()

    def plot_specific_signals(self, signal_bed_df: pd.DataFrame) -> None:
        """Highlight specific loci across all boroughs."""
        odds_df, evens_df = self._find_signals_specific(signal_bed_df)
        if self.signal_color_col is None:
            self._plot_signals(odds_df, evens_df)
        else:
            self._plot_color_signals(odds_df, evens_df)

    def plot_sig_signals(
        self,
        rep_genes: list = [],
        rep_boost: bool = False,
        legend_loc=None,
    ) -> None:
        """Highlight genome-wide significant signals across all boroughs."""
        odds_df, evens_df = self._find_signals_sig(rep_genes, rep_boost)
        if self.signal_color_col is None:
            self._plot_signals(odds_df, evens_df)
        else:
            self._plot_color_signals(odds_df, evens_df, legend_loc=legend_loc)

    def plot_annotations(
        self,
        plot_sig: bool = True,
        rep_genes: list = [],
        rep_boost: bool = False,
    ) -> None:
        """Draw pointer lines from signal peaks to the annotation label margin."""
        half_ld = self.ld_block / 2
        self.annot_list = []

        for i, b in enumerate(self.base_ax):
            already_pos   = []
            already_genes = []
            this_annot    = []

            annot_mask = self.thinned["P"] < self.annot_thresh
            annot_df   = self.thinned[annot_mask]
            annot_df   = annot_df[annot_df["WRAP"] == self.facets[i]]

            sig_mask  = annot_df["P"] < self.sig  if plot_sig    else False
            rep_mask  = annot_df["ID"].isin(rep_genes) if rep_boost else False
            sug_mask  = annot_df["P"] < self.sug
            spec_mask = annot_df["ID"].isin(self.spec_genes)
            full_mask = sig_mask | (sug_mask & rep_mask) | spec_mask
            annot_df  = annot_df[full_mask].set_index("ID")

            for signal_id, row in annot_df.iterrows():
                signal_gene = signal_id
                if rep_boost and signal_id in self.signal_rep_map:
                    new_gene = self.signal_rep_map[signal_id]
                    signal_id = new_gene
                    row.name  = new_gene

                plot = True
                if signal_id in already_genes:
                    plot = False
                elif self.merge_genes:
                    for x in already_pos:
                        if not plot:
                            break
                        if x - half_ld < row["ROUNDED_X"] < x + half_ld:
                            plot = False
                if row["P"] > self.annot_thresh:
                    plot = False

                if plot:
                    signal_df = annot_df.loc[annot_df.index == signal_gene]
                    pointer_y = (
                        signal_df[signal_df[self.plot_y_col] <= self.max_log_p][self.plot_y_col].max()
                        if self.max_log_p is not None
                        else signal_df[self.plot_y_col].max()
                    )
                    max_ax_y = b.get_ylim()[1]
                    b.plot(
                        [row[self.plot_x_col], row[self.plot_x_col]],
                        [pointer_y, max_ax_y],
                        c="silver", linewidth=1.5,
                    )
                    already_pos.append(row["ROUNDED_X"])
                    already_genes.append(signal_id)
                    this_annot.append(row)

            self.annot_list.append(this_annot)

    def plot_table(
        self,
        extra_cols: dict = {},
        number_cols: list = [],
        rep_genes: list = [],
        keep_chr_pos: bool = True,
        with_table_bg: bool = True,
        with_table_grid: bool = True,
        text_rep_colors: bool = False,
    ) -> None:
        """Render the annotation table below each borough panel."""
        self._plot_table_horizontal(
            rep_genes=rep_genes,
            with_table_bg=with_table_bg,
            with_table_grid=with_table_grid,
            text_rep_colors=text_rep_colors,
        )

    def full_plot(
        self,
        rep_genes: list = [],
        extra_cols: dict = {},
        number_cols: list = [],
        rep_boost: bool = False,
        save: Optional[str] = None,
        with_table: bool = True,
        save_res: int = 150,
        with_title: bool = True,
        keep_chr_pos: bool = True,
        with_table_bg: bool = True,
        with_table_grid: bool = True,
        legend_loc=None,
        text_rep_colors: bool = False,
    ) -> None:
        """
        One-call convenience: scatter, highlight signals, annotate, table, save.

        Parameters
        ----------
        rep_genes:
            Gene names considered replicated.
        rep_boost:
            Promote suggestive signals matching *rep_genes*.
        save:
            Output file path.
        save_res:
            DPI for the saved figure.
        legend_loc:
            ``'top'`` places the color legend above each borough.
        """
        self.facets = sorted(self.thinned["WRAP"].unique())
        self.plot_data(with_table=with_table, legend_loc=legend_loc)
        self.plot_sig_signals(rep_genes=rep_genes, rep_boost=rep_boost, legend_loc=legend_loc)
        if with_table:
            self.plot_annotations(rep_genes=rep_genes, rep_boost=rep_boost)
            self.plot_table(
                extra_cols=extra_cols,
                number_cols=number_cols,
                rep_genes=rep_genes,
                keep_chr_pos=keep_chr_pos,
                with_table_bg=with_table_bg,
                with_table_grid=with_table_grid,
                text_rep_colors=text_rep_colors,
            )
        if with_title:
            plt.suptitle(self.title)
        if save is not None:
            plt.savefig(save, dpi=save_res)

    def signal_plot(
        self,
        rep_genes: list = [],
        extra_cols: dict = {},
        number_cols: list = [],
        rep_boost: bool = False,
        save: Optional[str] = None,
        with_table: bool = True,
        save_res: int = 150,
        with_title: bool = True,
        keep_chr_pos: bool = True,
    ) -> None:
        """
        Plot only the significant signal regions (single-panel, no faceting).

        .. note::
            This method creates a single-axis layout and does not facet by
            ``WRAP``.  Use :meth:`full_plot` for the faceted view.
        """
        self._config_axes(with_table=with_table)

        odds_dfs, evens_dfs = self._find_signals_sig(rep_genes, rep_boost)
        # Collapse facet lists to single DataFrames for the single-axis view
        odds_df  = pd.concat(odds_dfs)
        evens_df = pd.concat(evens_dfs)
        signal_df = pd.concat([odds_df, evens_df]).sort_values(by=["#CHROM", "POS"])

        signal_order = signal_df["ID"].unique()
        signal_min   = signal_df.groupby("ID")["POS"].min().loc[signal_order]
        signal_max   = signal_df.groupby("ID")["POS"].max().loc[signal_order]
        signal_size  = signal_max - signal_min

        start_vals   = signal_size.cumsum().values[:-1]
        signal_start = pd.Series(data=start_vals, index=signal_size.index[1:])
        signal_start.loc[signal_size.index[0]] = 0
        signal_start = signal_start.loc[signal_size.index]
        signal_mid   = signal_start + signal_size / 2

        # NOTE: base_ax may not be a single Axes here if _config_axes wasn't
        # fully implemented for this config.  Guard against that.
        if hasattr(self.base_ax, "set_xticks"):
            self.base_ax.set_xticks(signal_mid.values)
            self.base_ax.set_xticklabels(signal_mid.index, rotation=30, ha="right")

        odd_signals  = signal_size.index[::2]
        even_signals = signal_size.index[1::2]
        pos_adjust   = -signal_min.loc[signal_df["ID"]] + signal_start.loc[signal_df["ID"]]
        signal_df["SIGNAL_X"]    = signal_df["POS"] + pos_adjust.values
        signal_df["SIGNAL_TEST"] = signal_df["POS"] - signal_min.loc[signal_df["ID"]].values
        self.df["SIGNAL_POS"]    = signal_df["SIGNAL_X"]
        self.plot_x_col = "SIGNAL_X"

        odds_df  = signal_df[signal_df["ID"].isin(odd_signals)]
        evens_df = signal_df[signal_df["ID"].isin(even_signals)]

        if self.signal_color_col is None:
            self.base_ax.scatter(odds_df["SIGNAL_X"],  odds_df[self.plot_y_col],  c=self.LIGHT_CHR_COLOR, s=25)
            self.base_ax.scatter(evens_df["SIGNAL_X"], evens_df[self.plot_y_col], c=self.DARK_CHR_COLOR,  s=25)
        else:
            self.base_ax.scatter(odds_df["SIGNAL_X"],  odds_df[self.plot_y_col],  c="silver",  s=25)
            self.base_ax.scatter(evens_df["SIGNAL_X"], evens_df[self.plot_y_col], c="dimgrey", s=25)

            color_min = min(odds_df[self.signal_color_col].quantile(0.05),  evens_df[self.signal_color_col].quantile(0.05))
            color_max = max(odds_df[self.signal_color_col].quantile(0.95),  evens_df[self.signal_color_col].quantile(0.95))
            print(color_min, color_max)

            filt_odds  = odds_df[odds_df["P"]   < 1e-3]
            filt_evens = evens_df[evens_df["P"] < 1e-3]
            kw = dict(s=25, cmap=self.COLOR_MAP, vmin=color_min, vmax=color_max)
            self.base_ax.scatter(filt_odds["SIGNAL_X"],  filt_odds[self.plot_y_col],  c=filt_odds[self.signal_color_col],  **kw)
            scat = self.base_ax.scatter(filt_evens["SIGNAL_X"], filt_evens[self.plot_y_col], c=filt_evens[self.signal_color_col], **kw)
            self.fig.colorbar(scat, cax=self.cbar_ax, orientation="horizontal")

        peak_idx  = signal_df.groupby("ID")["ROUNDED_Y"].idxmax()
        signal_df = signal_df.rename(columns=extra_cols)
        annot_df  = signal_df.loc[peak_idx.values].set_index("ID")
        self.annot_list = [r for _, r in annot_df.iterrows()]

        self._cosmetic_axis_edits(signals_only=True)
        self.base_ax.set_xlabel("Signal Label")

        if with_table:
            for _, row in annot_df.iterrows():
                self.base_ax.plot([row["SIGNAL_X"], row["SIGNAL_X"]], [row[self.plot_y_col], self.max_y], c="silver", linewidth=1.5)
            self.plot_table(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos)

        if with_title:
            plt.suptitle("Signals Only:\n" + self.title)
        if save is not None:
            plt.savefig(save, dpi=save_res)

    # PheWAS stubs (mirror ManhattanPlot API for consistency)
    def abacus_phewas_plot(self, save=None, save_res=150, with_title=True):
        """PheWAS abacus plot (boroughs layout)."""
        self.signal_color_col = "TRAIT"
        self._config_axes(with_table=False)
        self.plot_phewas_signals()
        if with_title:
            plt.suptitle(self.title)
        if save is not None:
            plt.savefig(save, dpi=save_res)

    def plot_phewas_signals(self):
        """Render PheWAS signals across boroughs."""
        self.df = self.df[self.df["P"] < 5e-8]
        unique_snps = self.df["ID"].unique()
        x_map = pd.Series(index=unique_snps, data=np.arange(len(unique_snps)) + 1)

        for x in x_map.values:
            self.base_ax.axvline(x, c="silver", zorder=0)

        unique_traits = list(self.df["TRAIT"].dropna().unique())
        categories    = sorted(unique_traits)
        cat_to_num    = dict(zip(categories, np.arange(len(categories))))
        cat_num_list  = [cat_to_num[t] for t in self.df["TRAIT"].dropna()]
        cmap          = plt.cm.get_cmap(self.COLOR_MAP, len(categories))

        self.fig.set_facecolor("w")
        scat = self.base_ax.scatter(
            x=x_map.loc[self.df.dropna(subset="TRAIT")["ID"]],
            y=-np.log10(self.df.dropna(subset="TRAIT")["P"]),
            c=cat_num_list, cmap=cmap, s=60, zorder=10,
        )
        self.base_ax.set_xticks(x_map.values)
        self.base_ax.set_xticklabels(x_map.index, rotation=30, ha="right")
        self.base_ax.set_xlabel("Search Identifiers")
        self.base_ax.set_ylabel("-Log10 P Value (Reported)")
        self.base_ax.set_ylim(0, self.max_log_p)

        print(categories, cat_to_num, unique_traits)
        self._add_color_bar(scat, categories)

    # ------------------------------------------------------------------
    # Private: axis configuration
    # ------------------------------------------------------------------

    def _config_axes(self, with_table: bool = True, legend_loc=None) -> None:
        """
        Set up the matplotlib figure / axes layout for the boroughs view.

        Currently only the ``need_cbar and with_table and legend_loc == 'top'``
        configuration is fully implemented.  Other branches are stubs that
        will raise or silently return until they are needed.
        """
        need_cbar   = (self.signal_color_col is not None) or (self.twas_color_col is not None)
        facet_count = len(self.facets)
        print(f"Configuring axes for {facet_count} facets/boroughs", flush=True)

        if not need_cbar and with_table:
            # TODO: implement no-colorbar layout for boroughs
            return

        elif need_cbar and with_table and legend_loc is None:
            # TODO: implement default color-bar layout for boroughs
            return

        elif need_cbar and with_table and legend_loc == "top":
            ratios = [0.15, 0.45, 1] if not self.invert else [1, 0.45, 0.15]
            ratios = np.tile(ratios, facet_count)

            self.fig, axes = plt.subplots(
                nrows=3 * facet_count, ncols=1,
                gridspec_kw={"height_ratios": ratios},
            )
            self.fig.set_size_inches(14.4, 5 * facet_count)
            self.table_ax = axes[1::3]

            if not self.invert:
                self.cbar_ax  = axes[0::3]
                self.base_ax  = axes[2::3]
            else:
                self.cbar_ax  = axes[2::3]
                self.base_ax  = axes[0::3]

        elif need_cbar and with_table and legend_loc == "side":
            # TODO: implement side-legend layout for boroughs
            return

        elif need_cbar and not with_table:
            # TODO: implement no-table layout for boroughs
            return

        elif not need_cbar and not with_table:
            # TODO: implement minimal layout for boroughs
            return

        else:
            raise ValueError("No support for your configuration.")

        if self.invert and self.base_ax is not None:
            for b in self.base_ax:
                b.invert_yaxis()

    # ------------------------------------------------------------------
    # Private: rendering helpers
    # ------------------------------------------------------------------

    def _get_odds_evens(self):
        """Return per-facet lists of odd-chromosome and even-chromosome DataFrames."""
        odds_nums  = np.arange(1, 24, 2)
        evens_nums = np.arange(2, 23, 2)

        odds_all  = self.thinned[self.thinned["#CHROM"].isin(odds_nums)].copy()
        evens_all = self.thinned[self.thinned["#CHROM"].isin(evens_nums)].copy()

        odds_list  = [odds_all[odds_all["WRAP"]  == f] for f in self.facets]
        evens_list = [evens_all[evens_all["WRAP"] == f] for f in self.facets]

        return odds_list, evens_list

    def _add_threshold_ticks(self) -> None:
        ticks = []
        if self.annot_thresh <= self.sug:
            ticks.append(-np.log10(self.annot_thresh))
        if self.sug < self.annot_thresh:
            ticks.append(-np.log10(self.sug))
        if self.sig < self.annot_thresh:
            ticks.append(-np.log10(self.sig))

        end1 = self.df["ABS_POS"].max()
        end2 = end1 * 0.99
        for t in ticks:
            for b in self.base_ax:
                b.plot([end1, end2], [t, t], c=self.FIFTH_COLOR)

    def _cosmetic_axis_edits(self, signals_only: bool = False) -> None:
        pos_col = "ABS_POS" if not signals_only else "SIGNAL_POS"
        for i, b in enumerate(self.base_ax):
            b.set_xlim(self.df[pos_col].min(), self.df[pos_col].max())
            b.set_ylabel("- Log10 P")
            b.set_xlabel("Chromosomal Position")
            b.axhline(-np.log10(self.sig_line), c=self.FIFTH_COLOR)
            invisi = "top" if not self.invert else "bottom"
            b.spines[invisi].set_visible(False)

            if not self.invert:
                b.set_ylim(bottom=np.floor(-np.log10(self.df["P"].max())))
                if self.max_log_p is not None:
                    b.set_ylim(top=self.max_log_p)
                self.max_y = b.get_ylim()[1]
            else:
                b.set_ylim(top=np.floor(-np.log10(self.df["P"].max())))
                if self.max_log_p is not None:
                    b.set_ylim(bottom=self.max_log_p)
                self.max_y = b.get_ylim()[0]

        self.fig.patch.set_facecolor("white")

    def _find_signals_sig(self, rep_genes=[], rep_boost=False):
        """Identify significant signals, independently per facet."""
        odds_list, evens_list = self._get_odds_evens()
        half_ld = self.ld_block / 2

        new_odds  = []
        new_evens = []
        self.signal_rep_map = {}

        for i in range(len(self.facets)):
            odds  = odds_list[i].copy()
            evens = evens_list[i].copy()

            odds["SIGNAL"]      = False
            evens["SIGNAL"]     = False
            odds["Replication"] = False
            evens["Replication"] = False

            annot_mask = self.thinned["P"] < self.annot_thresh
            test_df    = self.thinned[annot_mask]
            test_df    = test_df[test_df["WRAP"] == self.facets[i]]
            p_mask     = test_df["P"] < (self.sug if rep_boost else self.sig)
            test_df    = test_df[p_mask].sort_values(by="P")

            signal_genes = []

            for row_id, row in test_df.iterrows():
                if rep_boost and row["ID"] not in rep_genes and row["P"] > self.sig:
                    continue

                chr_df = odds if row["#CHROM"] % 2 == 1 else evens
                if (self.merge_genes or row["ID"] in signal_genes) and chr_df.loc[row_id, "SIGNAL"]:
                    continue

                x, gene  = row["ROUNDED_X"], row["ID"]
                pos_mask = chr_df["ROUNDED_X"].between(x - half_ld, x + half_ld)
                pos_idx  = chr_df.index[pos_mask]

                if rep_boost and self.merge_genes and np.any(chr_df.loc[pos_idx, "ID"].isin(rep_genes)):
                    window_genes = chr_df.loc[pos_idx, ["ID", "P"]].set_index("ID")
                    window_genes = window_genes[window_genes.index.isin(rep_genes)]
                    new_gene     = window_genes.idxmin().values[0]
                    target       = odds if row["#CHROM"] % 2 == 1 else evens
                    target.loc[pos_idx, "ID"] = new_gene
                    self.signal_rep_map[gene] = new_gene
                    gene = new_gene

                target    = odds if row["#CHROM"] % 2 == 1 else evens
                cur_rep   = target.loc[pos_idx, "Replication"]
                target.loc[pos_idx, "Replication"] = np.logical_or(cur_rep, gene in rep_genes)
                target.loc[pos_idx, "SIGNAL"] = True
                target.loc[pos_idx, "ID"]     = gene
                signal_genes.append(gene)

            new_odds.append(odds[odds["SIGNAL"]])
            new_evens.append(evens[evens["SIGNAL"]])

        print("Due to signal merging and replication prioritization, the following genes were renamed:")
        print("\n".join(k + ": " + v for k, v in self.signal_rep_map.items()))

        return new_odds, new_evens

    def _find_signals_specific(self, signal_bed_df: pd.DataFrame):
        """Mark variants overlapping user-supplied BED regions as signals."""
        odds_list, evens_list = self._get_odds_evens()
        self.spec_genes = []

        signal_bed_df = (
            signal_bed_df[signal_bed_df["#CHROM"].isin(self.df["#CHROM"])]
            .copy()
            .sort_values(by=["#CHROM", "POS"])
        )
        signal_bed_df["ABS_POS"]   = self._get_absolute_positions(signal_bed_df)
        signal_bed_df["ROUNDED_X"] = signal_bed_df["ABS_POS"] // self.CHR_POS_ROUND * self.CHR_POS_ROUND

        test_df = (
            self.thinned[self.thinned["P"] < self.annot_thresh]
            .reset_index(drop=False)
            .set_index("ROUNDED_X")
        )
        n = int(self.ld_block // self.CHR_POS_ROUND // 2 + 1)
        keep_locs = []
        for _, row in signal_bed_df.iterrows():
            x = row["ROUNDED_X"]
            keep_locs.extend(x + self.CHR_POS_ROUND * np.arange(-n, n + 1))
        test_df = test_df.loc[test_df.index.intersection(keep_locs)]

        # Operate on all facets combined; replication flags are applied globally
        for signal_df_list in (odds_list, evens_list):
            for fi, data_df in enumerate(signal_df_list):
                data_df["SIGNAL"]      = False
                data_df["Replication"] = False

                for chrom, sub_df in data_df.groupby("#CHROM"):
                    if chrom not in signal_bed_df["#CHROM"].values:
                        continue
                    print("chr" + str(chrom), end=" ")
                    rep_sub = signal_bed_df[signal_bed_df["#CHROM"] == chrom]

                    for _, bed_row in rep_sub.iterrows():
                        x     = bed_row["ROUNDED_X"]
                        start = bed_row["START"]
                        end   = bed_row["END"]

                        rounded_locs = x + self.CHR_POS_ROUND * np.arange(-n, n + 1)
                        locs_in_test = test_df.index.intersection(rounded_locs)
                        if len(locs_in_test) == 0:
                            continue

                        gene_df = test_df.loc[locs_in_test].copy().reset_index(drop=False).set_index("index")
                        gene    = bed_row["ID"] if "ID" in bed_row.index else gene_df["ID"].mode().iloc[0]

                        self.thinned.loc[gene_df.index, "ID"] = gene
                        if self.signal_color_col is not None and False in pd.isnull(gene_df[self.signal_color_col]):
                            self.thinned.loc[gene_df.index, self.signal_color_col] = gene_df[self.signal_color_col].mode().iloc[0]

                        if self.thinned.loc[self.thinned["ID"] == gene, "P"].min() > self.sug:
                            continue

                        self.spec_genes.append(gene)
                        pos_mask    = data_df["POS"].between(start, end)
                        signal_idx  = sub_df.index.intersection(data_df.index[pos_mask])
                        data_df.loc[signal_idx, "SIGNAL"]      = True
                        data_df.loc[signal_idx, "Replication"] = True

        odds_out  = [df[df["SIGNAL"]]  for df in odds_list]
        evens_out = [df[df["SIGNAL"]] for df in evens_list]
        print()
        return odds_out, evens_out

    def _plot_signals(self, odds_list, evens_list) -> None:
        for i, b in enumerate(self.base_ax):
            odds  = odds_list[i]
            evens = evens_list[i]
            colors_odd  = odds["Replication"].replace({True: self.REP_HIT_COLOR,  False: self.NOVEL_HIT_COLOR})
            colors_even = evens["Replication"].replace({True: self.REP_HIT_COLOR, False: self.NOVEL_HIT_COLOR})
            b.scatter(odds[self.plot_x_col],  odds[self.plot_y_col],  c=colors_odd,  s=10)
            b.scatter(evens[self.plot_x_col], evens[self.plot_y_col], c=colors_even, s=10)

    def _plot_color_signals(self, odds_list, evens_list, legend_loc=None) -> None:
        unique_vals = sorted(set(self.thinned[self.signal_color_col].dropna().unique()))
        discrete    = not pd.api.types.is_numeric_dtype(self.thinned[self.signal_color_col])

        for i, b in enumerate(self.base_ax):
            odds  = odds_list[i]
            evens = evens_list[i]

            if not discrete:
                color_min = min(odds[self.signal_color_col].quantile(0.05),  evens[self.signal_color_col].quantile(0.05))
                color_max = max(odds[self.signal_color_col].quantile(0.95),  evens[self.signal_color_col].quantile(0.95))
                kw = dict(cmap=plt.cm.get_cmap(self.COLOR_MAP), s=10, vmin=color_min, vmax=color_max)
                b.scatter(odds[self.plot_x_col],  odds[self.plot_y_col],  c=odds[self.signal_color_col],  **kw)
                scat = b.scatter(evens[self.plot_x_col], evens[self.plot_y_col], c=evens[self.signal_color_col], **kw)
                self.fig.colorbar(scat, cax=self.cbar_ax[i], orientation="horizontal")
            else:
                categories  = sorted(unique_vals)
                cat_to_num  = dict(zip(categories, np.arange(len(categories))))
                odds_copy   = odds.copy()
                evens_copy  = evens.copy()
                odds_copy["Cat_Num"]  = odds_copy[self.signal_color_col].replace(cat_to_num)
                evens_copy["Cat_Num"] = evens_copy[self.signal_color_col].replace(cat_to_num)
                odds_copy["pt_sz"]    = 10
                evens_copy["pt_sz"]   = 10

                use_cm  = plt.cm.get_cmap(self.COLOR_MAP, len(categories))
                cmap_kw = dict(cmap=use_cm, vmin=0, vmax=len(categories) - 1)
                b.scatter(odds_copy[self.plot_x_col],  odds_copy[self.plot_y_col],  c=odds_copy["Cat_Num"],  s=odds_copy["pt_sz"],  **cmap_kw)
                scat = b.scatter(evens_copy[self.plot_x_col], evens_copy[self.plot_y_col], c=evens_copy["Cat_Num"], s=evens_copy["pt_sz"], **cmap_kw)

                self._add_color_bar(scat, categories, legend_loc=legend_loc, axis_index=i)

    def _add_color_bar(self, mappable, categories, legend_loc=None, axis_index: int = 0) -> None:
        cb = self.cbar_ax[axis_index]

        if legend_loc is None:
            cbar   = self.fig.colorbar(mappable, cax=cb, orientation="horizontal")
            xmin, xmax = cb.get_xlim()
            factor = (xmax - xmin) / len(categories)
            cats   = [c if len(c) < 20 else c[:17] + "..." for c in categories]
            cbar.set_ticks((np.arange(len(categories)) + 0.5) * factor + xmin)
            if not self.invert:
                cbar.ax.set_xticklabels(cats, rotation=30, ha="left")
                cb.xaxis.tick_top()
            else:
                cbar.ax.set_xticklabels(cats, rotation=30, ha="right")
        else:
            plt.rc("legend", fontsize=12)
            cmap_obj = plt.cm.get_cmap(self.COLOR_MAP, len(categories))
            handles  = [mpatches.Patch(color=cmap_obj(mappable.norm(i)), label=cat) for i, cat in enumerate(categories)]
            if legend_loc == "side":
                nrows = 14
                cb.legend(handles=handles, loc="lower left", ncols=max(len(categories) // nrows, 1))
            elif legend_loc == "top":
                ncols = self.TOP_LEGEND_COLS
                cb.legend(handles=handles, loc="lower center", ncols=ncols)
            cb.xaxis.set_visible(False)
            cb.yaxis.set_visible(False)
            cb.spines[["right", "top", "left", "bottom"]].set_visible(False)

    # ------------------------------------------------------------------
    # Private: table rendering
    # ------------------------------------------------------------------

    def _plot_table_horizontal(
        self,
        rep_genes: list = [],
        with_table_bg: bool = True,
        with_table_grid: bool = True,
        text_rep_colors: bool = False,
    ) -> None:
        if not self.annot_list:
            raise ValueError("No signals to annotate. Try making P-value thresholds less stringent")

        for axi, ta in enumerate(self.table_ax):
            try:
                annot_table = pd.concat(self.annot_list[axi], axis=1).transpose()
            except ValueError:
                ta.set_visible(False)
                continue

            annot_table = annot_table.sort_values(by=["#CHROM", "POS"])
            genes    = [list(annot_table.index)]
            num_cols = len(annot_table)

            table = ta.table(
                cellText=genes,
                loc="lower center",
                colWidths=[1 / (num_cols + 2) for _ in genes[0]],
                cellLoc="center",
            )
            table.AXESPAD = 0
            cell_width  = 1 / (num_cols + 2)
            cell_height = table[(0, 0)].get_height()
            ta.set_axis_off()

            # TWAS color bar
            color_map = {}
            if self.twas_color_col is not None:
                unique_vals = sorted(annot_table[self.twas_color_col].unique())
                cmap        = plt.cm.get_cmap(self.COLOR_MAP, len(unique_vals))
                fractions   = (np.arange(len(unique_vals)) / len(unique_vals)) + 0.5 / len(unique_vals)
                colors_list = [cmap(f) for f in fractions]
                color_map   = dict(zip(unique_vals, colors_list))

                fractions   = list(fractions) + [1.0]
                new_norm    = mpl.colors.BoundaryNorm(boundaries=np.arange(len(unique_vals) + 1), ncolors=len(unique_vals))
                new_mappable = plt.cm.ScalarMappable(norm=new_norm, cmap=plt.cm.get_cmap(self.COLOR_MAP, len(unique_vals)))
                self._add_color_bar(new_mappable, list(color_map.keys()), axis_index=axi)

            for _, cell in table.get_celld().items():
                if with_table_grid:
                    cell.get_text().set_rotation(90)
                    cell.PAD = 0
                    cell.set_height(1)
                else:
                    cell.set_linewidth(0)
                    cell.get_text().set_visible(False)
                    cell.set_height(1)
                cell.get_text().set_fontsize(cell.get_text().get_fontsize() + 5)

            for i in range(num_cols):
                connection_row = annot_table.iloc[i]
                cell_text      = table[(0, i)].get_text().get_text()
                is_rep         = cell_text in rep_genes

                if with_table_bg:
                    table[(0, i)].set_facecolor(self.REP_TABLE_COLOR if is_rep else self.NOVEL_TABLE_COLOR)
                if text_rep_colors:
                    table[(0, i)].get_text().set_color(self.DARK_CHR_COLOR if is_rep else self.NOVEL_TABLE_COLOR)

                connect_y = 0 if not self.invert else 1
                connect_x = (cell_width * (i + 1)) - cell_width * 0.5
                max_ax_y  = self.base_ax[axi].get_ylim()[1]

                cp = ConnectionPatch(
                    xyA=(connection_row[self.plot_x_col], max_ax_y),
                    axesA=self.base_ax[axi], coordsA="data",
                    xyB=(connect_x, connect_y),
                    axesB=ta, coordsB="axes fraction",
                    arrowstyle="-", color="silver",
                )

                if not with_table_grid:
                    row_text_color = "dimgrey" if is_rep and text_rep_colors else (self.NOVEL_HIT_COLOR if text_rep_colors else "k")
                    if not self.invert:
                        ta.text(connect_x - 0.005, connect_y, cell_text, ha="left", va="bottom", rotation=45, transform=ta.transAxes, color=row_text_color, style="italic")
                    else:
                        ta.text(connect_x + 0.005, connect_y, cell_text, ha="right", va="top",   rotation=45, transform=ta.transAxes, color=row_text_color, style="italic")

                if self.twas_updown_col is not None:
                    shape = "v" if connection_row[self.twas_updown_col] < 0 else "^"
                    color = color_map.get(connection_row[self.twas_color_col], self.NOVEL_HIT_COLOR) if self.twas_color_col else (self.REP_HIT_COLOR if is_rep else self.NOVEL_HIT_COLOR)
                    self.base_ax[axi].scatter(connection_row[self.plot_x_col], connection_row[self.plot_y_col], color=color, marker=shape, s=60)

                self.fig.add_artist(cp)

            # Facet label
            cba = self.cbar_ax[axi]
            cba.text(cba.get_xlim()[0], cba.get_ylim()[0] - 0.1, self.facets[axi], ha="left", va="top")

        self.fig.tight_layout()
