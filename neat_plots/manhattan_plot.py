"""
ManhattanPlot — single-axis Manhattan (and QQ) plots for GWAS/TWAS/PheWAS.

Inherits all data-handling, thinning, and utility logic from BasePlot.
This module owns only the rendering pipeline.

Typical usage
-------------
::

    from neat_plots import ManhattanPlot

    mp = ManhattanPlot("gwas_sumstats.txt.gz", title="My GWAS")
    mp.load_data()
    mp.clean_data(col_map={"CHR": "#CHROM", "BP": "POS", "P_VALUE": "P"})
    mp.get_thinned_data()
    mp.update_plotting_parameters(vertical=True, sig=5e-8, merge_genes=True)
    mp.full_plot(save="manhattan.png", rep_boost=True)
    mp.qq_plot(save="qq.png")
"""

from __future__ import annotations

import sys
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
class ManhattanPlot(BasePlot):
    """
    Single-axis Manhattan plot supporting vertical and horizontal
    orientations, axis breaks, GWAS / TWAS / PheWAS coloring schemes,
    and an optional annotation table.

    Parameters
    ----------
    file_path : str
        Path to the summary statistics file.
    test_rows : int, optional
        Limit the number of rows loaded (useful during development).
    title : str
        Plot title.
    """

    # PheWAS-specific columns (not present in BoroughsPlot)
    phewas_updown_col:   Optional[str] = None
    phewas_rep_color_col: Optional[str] = None
    phewas_size_col:     Optional[str] = None
    phewas_annotate_col: Optional[str] = None
    phewas_fill_col:     Optional[str] = None

    # Axis-break midpoint for split-axis PheWAS plots
    log_p_axis_midpoint: Optional[float] = None

    vertical: bool = True

    def __init__(
        self,
        file_path: str,
        test_rows: Optional[int] = None,
        title: str = "Manhattan Plot",
    ) -> None:
        super().__init__(file_path, test_rows=test_rows, title=title)

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def update_plotting_parameters(
        self,
        log_p_axis_midpoint="",
        annotate="",
        signal_color_col="",
        phewas_rep_color_col="",
        phewas_updown_col="",
        phewas_size_col="",
        phewas_annotate_col="",
        phewas_fill_col="",
        twas_color_col="",
        twas_updown_col="",
        sig="",
        sug="",
        annot_thresh="",
        ld_block="",
        vertical="",
        max_log_p="",
        invert="",
        merge_genes="",
        title="",
    ) -> None:
        """
        Update plotting parameters.  Only non-empty arguments change state.

        Parameters
        ----------
        sig, sug, annot_thresh : float
            P-value thresholds for significance, suggestiveness, and annotation.
        ld_block : float
            LD window half-width (bp) used for signal merging.
        vertical : bool
            ``True`` → chromosomes on Y-axis (default). ``False`` → horizontal.
        merge_genes : bool
            Collapse nearby signals within *ld_block* into a single label.
        invert : bool
            Invert the p-value axis direction.
        max_log_p : float or None
            Hard cap on the displayed ``-log10(P)`` axis.
        log_p_axis_midpoint : float or None
            If set, a split (broken) y-axis is drawn at this value (PheWAS).
        signal_color_col : str or None
            Column name used to colour significant signals.
        twas_color_col, twas_updown_col : str or None
            TWAS tissue-coloring columns.
        phewas_annotate_col, phewas_rep_color_col, phewas_size_col, phewas_updown_col, phewas_fill_col : str or None
            PheWAS-specific annotation columns.
        annotate : bool
            Whether to add gene-name annotations.
        title : str
            Override the plot title.
        """
        # Delegate shared params to base
        super().update_plotting_parameters(
            annotate=annotate,
            signal_color_col=signal_color_col,
            twas_color_col=twas_color_col,
            twas_updown_col=twas_updown_col,
            sig=sig,
            sug=sug,
            annot_thresh=annot_thresh,
            ld_block=ld_block,
            max_log_p=max_log_p,
            invert=invert,
            merge_genes=merge_genes,
            title=title,
        )

        _set = self._update_param
        self.log_p_axis_midpoint = _set(self.log_p_axis_midpoint, log_p_axis_midpoint)

        self.phewas_rep_color_col  = _set(self.phewas_rep_color_col,  phewas_rep_color_col)
        self.phewas_updown_col     = _set(self.phewas_updown_col,     phewas_updown_col)
        self.phewas_size_col       = _set(self.phewas_size_col,       phewas_size_col)
        self.phewas_annotate_col   = _set(self.phewas_annotate_col,   phewas_annotate_col)
        self.phewas_fill_col       = _set(self.phewas_fill_col,       phewas_fill_col)

        self.vertical = _set(self.vertical, vertical)
        # Keep the axis-column aliases consistent with orientation
        self.plot_x_col = "ROUNDED_Y" if self.vertical else "ROUNDED_X"
        self.plot_y_col = "ROUNDED_X" if self.vertical else "ROUNDED_Y"

    def check_plotting_parameters(self) -> None:
        """Print current plotting parameters including orientation."""
        super().check_plotting_parameters()
        print({"Orientation": "Vertical" if self.vertical else "Horizontal"})

    # ------------------------------------------------------------------
    # High-level plot entry points
    # ------------------------------------------------------------------

    def plot_data(self, with_table: bool = True, legend_loc=None) -> None:
        """
        Scatter all thinned data onto ``self.base_ax``.

        Colours points by chromosome (alternating light/dark) unless a
        coloring column is active, in which case background points are grey.
        """
        if self.base_ax is None:
            self._config_axes(with_table=with_table, legend_loc=legend_loc)

        if self.vertical:
            self.base_ax.set_yticks(self.chr_ticks[0])
            self.base_ax.set_yticklabels(self.chr_ticks[1])
            if self.invert:
                self.base_ax.yaxis.set_label_position("right")
                self.base_ax.yaxis.tick_right()
        else:
            self.base_ax.set_xticks(self.chr_ticks[0])
            self.base_ax.set_xticklabels(self.chr_ticks[1])
            if self.invert:
                self.base_ax.xaxis.set_label_position("top")
                self.base_ax.xaxis.tick_top()

        odds_df, evens_df = self._get_odds_evens()

        if self.signal_color_col is None and self.twas_color_col is None:
            self.base_ax.scatter(odds_df[self.plot_x_col],  odds_df[self.plot_y_col],  c=self.LIGHT_CHR_COLOR, s=2)
            self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=self.DARK_CHR_COLOR,  s=2)
        else:
            if self.phewas_size_col is None:
                odds_df["pt_sz"]  = 2
                evens_df["pt_sz"] = 2
            else:
                odds_df["pt_sz"]  = self._convert_linear_scale(odds_df[self.phewas_size_col].abs(),  self.MIN_PT_SZ, self.MAX_PT_SZ)
                evens_df["pt_sz"] = self._convert_linear_scale(evens_df[self.phewas_size_col].abs(), self.MIN_PT_SZ, self.MAX_PT_SZ)

            if self.phewas_updown_col is None:
                self.base_ax.scatter(odds_df[self.plot_x_col],  odds_df[self.plot_y_col],  c="silver",  s=odds_df["pt_sz"])
                self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c="dimgray", s=evens_df["pt_sz"])
            else:
                self.base_ax.scatter(odds_df[self.plot_x_col],  odds_df[self.plot_y_col],  edgecolors="silver",  facecolors="none", s=odds_df["pt_sz"],  alpha=1, linewidth=0.2)
                self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], edgecolors="dimgray", facecolors="none", s=evens_df["pt_sz"], alpha=1, linewidth=0.2)

        self._add_threshold_ticks()
        self._cosmetic_axis_edits()

    def plot_specific_signals(
        self,
        signal_bed_df: pd.DataFrame,
        rep_genes: list = [],
        legend_loc=None,
    ) -> None:
        """Highlight signals from a user-supplied BED-like DataFrame."""
        odds_df, evens_df = self._find_signals_specific(signal_bed_df, rep_genes=rep_genes)
        if self.signal_color_col is None:
            self._plot_signals(odds_df, evens_df)
        else:
            self._plot_color_signals(odds_df, evens_df, legend_loc=legend_loc)

    def plot_sig_signals(
        self,
        rep_genes: list = [],
        rep_boost: bool = False,
        legend_loc=None,
    ) -> None:
        """Highlight genome-wide significant (or suggestive + known) signals."""
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
        specific_sig_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Draw pointer lines from signal peaks to the annotation table margin.

        Populates ``self.annot_list`` with the annotated signal rows.
        """
        half_ld = self.ld_block / 2
        already_plotted_pos   = []
        already_plotted_genes = []
        self.annot_list = []
        specific_loci = None if specific_sig_df is None else specific_sig_df.set_index("ID")

        annot_mask = self.thinned["P"] < self.annot_thresh
        annot_df = self.thinned[annot_mask]

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
            if signal_id in already_plotted_genes:
                plot = False
            elif self.merge_genes:
                for x in already_plotted_pos:
                    if not plot:
                        break
                    if x - half_ld < row["ROUNDED_X"] < x + half_ld:
                        plot = False

            if row["P"] > self.annot_thresh:
                plot = False

            if plot:
                signal_df = annot_df.loc[annot_df.index == signal_gene]
                skip_pointer = (
                    specific_loci is not None
                    and signal_id in specific_loci.index
                    and "SKIP_POINTER" in specific_loci.columns
                    and specific_loci.loc[signal_id, "SKIP_POINTER"]
                )
                if skip_pointer:
                    print(f"Skipping pointer for {signal_id}")
                else:
                    if self.vertical:
                        pointer_x = (
                            signal_df[signal_df[self.plot_x_col] <= self.max_log_p][self.plot_x_col].max()
                            if self.max_log_p is not None
                            else signal_df[self.plot_x_col].max()
                        )
                        self.base_ax.plot([pointer_x, self.max_x], [row[self.plot_y_col], row[self.plot_y_col]], c="silver", linewidth=1.5)
                    else:
                        pointer_y = (
                            signal_df[signal_df[self.plot_y_col] <= self.max_log_p][self.plot_y_col].max()
                            if self.max_log_p is not None
                            else signal_df[self.plot_y_col].max()
                        )
                        self.base_ax.plot([row[self.plot_x_col], row[self.plot_x_col]], [pointer_y, self.max_y], c="silver", linewidth=1.5)

                already_plotted_pos.append(row["ROUNDED_X"])
                already_plotted_genes.append(signal_id)
                self.annot_list.append(row)

    def plot_table(
        self,
        extra_cols: dict = {},
        number_cols: list = [],
        rep_genes: list = [],
        keep_chr_pos: bool = True,
        with_table_bg: bool = True,
        with_table_grid: bool = True,
        text_rep_colors: bool = False,
        table_fontsize: int = DEFAULT_TABLE_FONTSIZE,
        legend_loc=None,
        specific_sig_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Dispatch to the vertical or horizontal table renderer."""
        if self.vertical:
            self._plot_table_vertical(
                extra_cols=extra_cols,
                number_cols=number_cols,
                rep_genes=rep_genes,
                keep_chr_pos=keep_chr_pos,
                table_fontsize=table_fontsize,
                legend_loc=legend_loc,
                specific_sig_df=specific_sig_df,
            )
        else:
            self._plot_table_horizontal(
                rep_genes=rep_genes,
                with_table_bg=with_table_bg,
                with_table_grid=with_table_grid,
                text_rep_colors=text_rep_colors,
                legend_loc=legend_loc,
                specific_sig_df=specific_sig_df,
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
        table_fontsize: int = DEFAULT_TABLE_FONTSIZE,
    ) -> None:
        """
        One-call convenience: scatter all data, highlight signals, annotate,
        add table, save.

        Parameters
        ----------
        rep_genes:
            Gene names considered replicated (shown in gold).
        rep_boost:
            Promote suggestive signals that match *rep_genes* to significant.
        extra_cols:
            ``{source_col: display_label}`` extra columns to show in the table.
        number_cols:
            Column names in the table that should be formatted as numbers.
        with_table:
            Include the annotation table panel.
        save:
            Output file path.
        save_res:
            DPI for the saved figure.
        """
        self.plot_data(with_table=with_table, legend_loc=legend_loc)
        self.plot_sig_signals(rep_genes=rep_genes, rep_boost=rep_boost, legend_loc=legend_loc)
        if with_table:
            if self.phewas_annotate_col is None:
                self.plot_annotations(rep_genes=rep_genes, rep_boost=rep_boost)
            else:
                self._plot_pointers_only()
            self.plot_table(
                extra_cols=extra_cols,
                number_cols=number_cols,
                rep_genes=rep_genes,
                keep_chr_pos=keep_chr_pos,
                with_table_bg=with_table_bg,
                with_table_grid=with_table_grid,
                text_rep_colors=text_rep_colors,
                table_fontsize=table_fontsize,
                legend_loc=legend_loc,
            )
        if with_title:
            plt.suptitle(self.title)
            plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res, bbox_inches="tight")

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
        table_fontsize: int = DEFAULT_TABLE_FONTSIZE,
    ) -> None:
        """Plot only the significant signal regions (zoomed locus view)."""
        self._config_axes(with_table=with_table)
        odds_df, evens_df = self._find_signals_sig(rep_genes, rep_boost)
        signal_df = pd.concat([odds_df, evens_df]).sort_values(by=["#CHROM", "POS"])
        signal_df, signal_mid = self._build_signal_coords(signal_df)

        if self.vertical:
            self.base_ax.set_yticks(signal_mid.values)
            self.base_ax.set_yticklabels(signal_mid.index)
        else:
            self.base_ax.set_xticks(signal_mid.values)
            self.base_ax.set_xticklabels(signal_mid.index, rotation=30, ha="right")

        odds_df, evens_df = self._split_signal_odds_evens(signal_df)
        self._scatter_signal_plot(odds_df, evens_df)

        peak_idx = signal_df.groupby("ID")["ROUNDED_Y"].idxmax()
        signal_df = signal_df.rename(columns=extra_cols)
        annot_df = signal_df.loc[peak_idx.values].set_index("ID")
        self.annot_list = [r for _, r in annot_df.iterrows()]

        self._cosmetic_axis_edits(signals_only=True)
        if self.vertical:
            self.base_ax.set_ylabel("Signal Label")
            self.base_ax.grid(visible=False, which="both", axis="x")
        else:
            self.base_ax.set_xlabel("Signal Label")
            self.base_ax.grid(visible=False, which="both", axis="y")

        if with_table:
            self._draw_signal_pointers(annot_df)
            self.plot_table(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos, table_fontsize=table_fontsize)

        if with_title:
            plt.suptitle("Signals Only:\n" + self.title)
            plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res, bbox_inches="tight")

    def full_plot_with_specific(
        self,
        signal_bed_df: pd.DataFrame,
        plot_sig: bool = True,
        rep_boost: bool = False,
        rep_genes: list = [],
        extra_cols: dict = {},
        number_cols: list = [],
        verbose: bool = False,
        save: Optional[str] = None,
        save_res: int = 150,
        keep_chr_pos: bool = True,
        with_table_bg: bool = True,
        with_table_grid: bool = True,
        legend_loc=None,
        with_table: bool = True,
        table_fontsize: int = DEFAULT_TABLE_FONTSIZE,
    ) -> None:
        """Full plot combining genome-wide signals with user-supplied loci."""
        if verbose:
            print("Plotting All Data...", flush=True)
        self.plot_data(with_table=with_table)
        if plot_sig:
            if verbose:
                print("Plotting Significant Signals...", flush=True)
            self.plot_sig_signals()
        if verbose:
            print("Plotting Specific Signals...", flush=True)
        self.plot_specific_signals(signal_bed_df, rep_genes=rep_genes, legend_loc=legend_loc)
        if with_table:
            if verbose:
                print("Finding Annotations...", flush=True)
            self.plot_annotations(plot_sig=plot_sig, rep_genes=rep_genes, rep_boost=rep_boost, specific_sig_df=signal_bed_df)
            if verbose:
                print("Adding Table...", flush=True)
            self.plot_table(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos, with_table_grid=with_table_grid, with_table_bg=with_table_bg, table_fontsize=table_fontsize)
        if save is not None:
            if verbose:
                print("Writing Figure to File...", flush=True)
            plt.savefig(save, dpi=save_res, bbox_inches="tight")

    def signal_plot_with_specific(
        self,
        signal_bed_df: pd.DataFrame,
        rep_genes: list = [],
        extra_cols: dict = {},
        number_cols: list = [],
        rep_boost: bool = False,
        save: Optional[str] = None,
        with_table: bool = True,
        save_res: int = 150,
        with_title: bool = True,
        keep_chr_pos: bool = True,
        table_fontsize: int = DEFAULT_TABLE_FONTSIZE,
    ) -> None:
        """Signals-only plot using user-supplied BED regions."""
        self._config_axes(with_table=with_table)
        odds_df, evens_df = self._find_signals_specific(signal_bed_df, rep_genes=rep_genes)
        print(len(odds_df), len(evens_df))
        signal_df = pd.concat([odds_df, evens_df]).sort_values(by=["#CHROM", "POS"])
        signal_df, signal_mid = self._build_signal_coords(signal_df)

        if self.vertical:
            self.base_ax.set_yticks(signal_mid.values)
            self.base_ax.set_yticklabels(signal_mid.index)
        else:
            self.base_ax.set_xticks(signal_mid.values)
            self.base_ax.set_xticklabels(signal_mid.index, rotation=30, ha="right")

        odds_df, evens_df = self._split_signal_odds_evens(signal_df)
        self._scatter_signal_plot(odds_df, evens_df)

        peak_idx = signal_df.groupby("ID")["ROUNDED_Y"].idxmax()
        print(peak_idx)
        signal_df = signal_df.rename(columns=extra_cols)
        annot_df = signal_df.loc[peak_idx.values].set_index("ID")
        self.annot_list = [r for _, r in annot_df.iterrows()]

        if len(annot_df) > 45:
            sys.exit("Too many signals to annotate...Exiting")

        self._cosmetic_axis_edits(signals_only=True)
        if self.vertical:
            self.base_ax.set_ylabel("Signal Label")
            self.base_ax.grid(visible=False, which="both", axis="x")
        else:
            self.base_ax.set_xlabel("Signal Label")
            self.base_ax.grid(visible=False, which="both", axis="y")

        if with_table:
            self._draw_signal_pointers(annot_df)
            self.plot_table(extra_cols=extra_cols, number_cols=number_cols, rep_genes=rep_genes, keep_chr_pos=keep_chr_pos, table_fontsize=table_fontsize)

        if with_title:
            plt.suptitle("Signals Only:\n" + self.title)
            plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res, bbox_inches="tight")

    # ------------------------------------------------------------------
    # PheWAS
    # ------------------------------------------------------------------

    def abacus_phewas_plot(
        self,
        save: Optional[str] = None,
        save_res: int = 150,
        with_title: bool = True,
    ) -> None:
        """Abacus-style PheWAS plot (horizontal, colored by trait category)."""
        self.vertical = False
        self.signal_color_col = "TRAIT"
        self._config_axes(with_table=False)
        self.plot_phewas_signals()
        if with_title:
            plt.suptitle(self.title)
            plt.tight_layout()
        if save is not None:
            plt.savefig(save, dpi=save_res, bbox_inches="tight")

    def plot_phewas_signals(self) -> None:
        """Render significant PheWAS associations colored by trait."""
        self.df = self.df[self.df["P"] < 5e-8]
        unique_snps = self.df["ID"].unique()
        x_map = pd.Series(index=unique_snps, data=np.arange(len(unique_snps)) + 1)

        for x in x_map.values:
            ax = self.base_ax if self.log_p_axis_midpoint is None else self.lower_base_ax
            ax.axvline(x, c="silver", zorder=0)
            if self.log_p_axis_midpoint is not None:
                self.upper_base_ax.axvline(x, c="silver", zorder=0)

        unique_traits = list(self.df["TRAIT"].dropna().unique())
        categories = sorted(unique_traits)
        cat_to_num  = dict(zip(categories, np.arange(len(categories))))
        cat_num_list = [cat_to_num[t] for t in self.df["TRAIT"].dropna()]

        self.fig.set_facecolor("w")
        cmap = plt.cm.get_cmap(self.COLOR_MAP, len(categories))

        if self.log_p_axis_midpoint is None:
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
        else:
            kwargs = dict(
                x=x_map.loc[self.df.dropna(subset="TRAIT")["ID"]],
                y=-np.log10(self.df.dropna(subset="TRAIT")["P"]),
                c=cat_num_list, cmap=cmap, s=60, zorder=10,
            )
            self.upper_base_ax.scatter(**kwargs)
            scat = self.lower_base_ax.scatter(**kwargs)
            self.lower_base_ax.set_xticks(x_map.values)
            self.lower_base_ax.set_xticklabels(x_map.index, rotation=30, ha="right")
            self.lower_base_ax.set_xlabel("Search Identifiers")
            self.lower_base_ax.set_ylabel("-Log10 P Value (Reported)")
            self.lower_base_ax.set_ylim(0, self.log_p_axis_midpoint)
            self.upper_base_ax.set_ylim(self.log_p_axis_midpoint, self.max_log_p)
            if not self.invert:
                self.lower_base_ax.spines["top"].set_visible(False)
                self.upper_base_ax.spines["bottom"].set_visible(False)
            else:
                self.upper_base_ax.spines["top"].set_visible(False)
                self.lower_base_ax.spines["bottom"].set_visible(False)

        print(categories, cat_to_num, unique_traits)
        self._add_color_bar(scat, categories)

    # ------------------------------------------------------------------
    # Private: axis configuration
    # ------------------------------------------------------------------

    def _config_axes(self, with_table: bool = True, legend_loc=None) -> None:
        """Set up the matplotlib figure / axes layout for the chosen configuration."""
        need_cbar = (self.signal_color_col is not None) or (self.twas_color_col is not None)

        if self.log_p_axis_midpoint is None:
            if self.vertical and not need_cbar and with_table:
                self.fig, axes = plt.subplots(nrows=1, ncols=2)
                self.fig.set_size_inches(12, 12)
                self.base_ax, self.table_ax = (axes[0], axes[1]) if not self.invert else (axes[1], axes[0])

            elif not self.vertical and not need_cbar and with_table:
                print("Horizontal, no color bar")
                ratios = [0.4, 1] if not self.invert else [1, 0.4]
                self.fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={"height_ratios": ratios})
                self.fig.set_size_inches(14.4, 6)
                self.table_ax, self.base_ax = (axes[0], axes[1]) if not self.invert else (axes[1], axes[0])

            elif not self.vertical and need_cbar and with_table and legend_loc is None:
                ratios = [0.08, 0.45, 1] if not self.invert else [1, 0.45, 0.08]
                self.fig, axes = plt.subplots(nrows=3, ncols=1, gridspec_kw={"height_ratios": ratios})
                self.fig.set_size_inches(14.4, 6)
                self.table_ax = axes[1]
                if not self.invert:
                    self.cbar_ax, self.base_ax = axes[0], axes[2]
                else:
                    self.cbar_ax, self.base_ax = axes[2], axes[0]

            elif not self.vertical and need_cbar and with_table and legend_loc == "top":
                ratios = [0.15, 0.45, 1] if not self.invert else [1, 0.45, 0.15]
                self.fig, axes = plt.subplots(nrows=3, ncols=1, gridspec_kw={"height_ratios": ratios})
                self.fig.set_size_inches(14.4, 6)
                self.table_ax = axes[1]
                if not self.invert:
                    self.cbar_ax, self.base_ax = axes[0], axes[2]
                else:
                    self.cbar_ax, self.base_ax = axes[2], axes[0]

            elif not self.vertical and need_cbar and with_table and legend_loc == "side":
                print("Horizontal, table, side legend")
                ratios = [0.4, 1] if not self.invert else [1, 0.4]
                self.fig = plt.figure()
                gs0 = self.fig.add_gridspec(1, 2, width_ratios=[1, 0.2])
                gs1 = gs0[0].subgridspec(2, 1, height_ratios=ratios)
                if not self.invert:
                    self.table_ax = self.fig.add_subplot(gs1[0])
                    self.base_ax  = self.fig.add_subplot(gs1[1])
                else:
                    self.table_ax = self.fig.add_subplot(gs1[1])
                    self.base_ax  = self.fig.add_subplot(gs1[0])
                self.cbar_ax = self.fig.add_subplot(gs0[1])
                self.fig.set_size_inches(14.4, 6)

            elif not self.vertical and need_cbar and not with_table:
                ratios = [0.08, 1] if not self.invert else [1, 0.08]
                self.fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={"height_ratios": ratios})
                self.fig.set_size_inches(14.4, 4)
                if not self.invert:
                    self.cbar_ax, self.base_ax = axes[0], axes[1]
                else:
                    self.cbar_ax, self.base_ax = axes[1], axes[0]

            elif not self.vertical and not need_cbar and not with_table:
                self.fig, self.base_ax = plt.subplots()
                self.fig.set_size_inches(13, 3)

            elif self.vertical and not need_cbar and not with_table:
                self.fig, self.base_ax = plt.subplots()
                self.fig.set_size_inches(6, 12)

            elif self.vertical and need_cbar and not with_table:
                ratios = [0.08, 1] if not self.invert else [1, 0.08]
                self.fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios": ratios})
                self.fig.set_size_inches(6, 12)
                if not self.invert:
                    self.cbar_ax, self.base_ax = axes[0], axes[1]
                else:
                    self.cbar_ax, self.base_ax = axes[1], axes[0]

            elif self.vertical and need_cbar and with_table:
                ratios = [0.45, 0.08, 1] if not self.invert else [1, 0.08, 0.45]
                self.fig, axes = plt.subplots(nrows=1, ncols=3, gridspec_kw={"width_ratios": ratios})
                self.fig.set_size_inches(14, 12)
                if not self.invert:
                    self.table_ax, self.cbar_ax, self.base_ax = axes[0], axes[1], axes[2]
                else:
                    self.base_ax, self.cbar_ax, self.table_ax = axes[0], axes[1], axes[2]

            else:
                raise ValueError("No support for your configuration.")

        else:  # split-axis (log_p_axis_midpoint set)
            if not self.vertical and need_cbar and with_table:
                ratios = [0.08, 0.45, 0.5, 0.5] if not self.invert else [0.5, 0.5, 0.45, 0.08]
                self.fig, axes = plt.subplots(nrows=4, ncols=1, gridspec_kw={"height_ratios": ratios, "hspace": 0.05})
                self.fig.set_size_inches(14.4, 6)
                if not self.invert:
                    self.cbar_ax      = axes[0]
                    self.table_ax     = axes[1]
                    self.upper_base_ax = axes[2]
                    self.lower_base_ax = axes[3]
                else:
                    self.cbar_ax      = axes[3]
                    self.table_ax     = axes[2]
                    self.upper_base_ax = axes[1]
                    self.lower_base_ax = axes[0]

            elif not self.vertical and need_cbar and not with_table:
                ratios = [0.08, 0.5, 0.5] if not self.invert else [0.5, 0.5, 0.08]
                self.fig, axes = plt.subplots(nrows=3, ncols=1, gridspec_kw={"height_ratios": ratios, "hspace": 0})
                self.fig.set_size_inches(14.4, 4)
                if not self.invert:
                    self.cbar_ax      = axes[0]
                    self.upper_base_ax = axes[1]
                    self.lower_base_ax = axes[2]
                else:
                    self.cbar_ax      = axes[2]
                    self.upper_base_ax = axes[1]
                    self.lower_base_ax = axes[0]
            else:
                print("No support for your configuration.")

        if self.vertical and self.invert:
            self.base_ax.invert_xaxis()
        elif not self.vertical and self.invert:
            self.base_ax.invert_yaxis()

    # ------------------------------------------------------------------
    # Private: rendering helpers
    # ------------------------------------------------------------------

    def _get_odds_evens(self):
        odds  = np.arange(1, 24, 2)
        evens = np.arange(2, 23, 2)
        odds_df  = self.thinned[self.thinned["#CHROM"].isin(odds)].copy()
        evens_df = self.thinned[self.thinned["#CHROM"].isin(evens)].copy()
        return odds_df, evens_df

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
            if self.vertical:
                self.base_ax.plot([t, t], [end1, end2], c=self.FIFTH_COLOR)
            else:
                self.base_ax.plot([end1, end2], [t, t], c=self.FIFTH_COLOR)

    def _cosmetic_axis_edits(self, signals_only: bool = False) -> None:
        pos_col = "ABS_POS" if not signals_only else "SIGNAL_POS"
        if self.vertical:
            self.base_ax.set_ylim(self.df[pos_col].min(), self.df[pos_col].max())
            self.base_ax.set_xlabel("- Log10 P")
            self.base_ax.set_ylabel("Chromosomal Position")
            self.base_ax.axvline(-np.log10(self.sig_line), c=self.FIFTH_COLOR)
            self.base_ax.invert_yaxis()
            invisi = "right" if not self.invert else "left"
            self.base_ax.spines[invisi].set_visible(False)
            print(self.thinned[self.plot_x_col].min())
            print(self.plot_x_col)
            if not self.invert:
                self.base_ax.set_xlim(left=self.thinned[self.plot_x_col].min())
                if self.max_log_p is not None:
                    self.base_ax.set_xlim(right=self.max_log_p)
                self.max_x = self.base_ax.get_xlim()[1]
            else:
                self.base_ax.set_xlim(right=self.thinned[self.plot_x_col].min())
                if self.max_log_p is not None:
                    self.base_ax.set_xlim(left=self.max_log_p)
                self.max_x = self.base_ax.get_xlim()[0]
        else:
            self.base_ax.set_xlim(self.df[pos_col].min(), self.df[pos_col].max())
            self.base_ax.set_ylabel("- Log10 P")
            self.base_ax.set_xlabel("Chromosomal Position")
            self.base_ax.axhline(-np.log10(self.sig_line), c=self.FIFTH_COLOR)
            invisi = "top" if not self.invert else "bottom"
            self.base_ax.spines[invisi].set_visible(False)
            if not self.invert:
                self.base_ax.set_ylim(bottom=np.floor(-np.log10(self.df["P"].max())))
                if self.max_log_p is not None:
                    self.base_ax.set_ylim(top=self.max_log_p)
                self.max_y = self.base_ax.get_ylim()[1]
            else:
                self.base_ax.set_ylim(top=np.floor(-np.log10(self.df["P"].max())))
                if self.max_log_p is not None:
                    self.base_ax.set_ylim(bottom=self.max_log_p)
                self.max_y = self.base_ax.get_ylim()[0]
        self.fig.patch.set_facecolor("white")

    def _find_signals_sig(self, rep_genes=[], rep_boost=False):
        """Identify significant (and optionally rep-boosted) signal windows."""
        odds_df, evens_df = self._get_odds_evens()
        half_ld = self.ld_block / 2

        for df in (odds_df, evens_df):
            df["SIGNAL"]      = False
            df["Replication"] = False

        annot_mask = self.thinned["P"] < self.annot_thresh
        test_df    = self.thinned[annot_mask]
        p_mask     = test_df["P"] < (self.sug if rep_boost else self.sig)
        test_df    = test_df[p_mask].sort_values(by="P")

        signal_genes = []
        self.signal_rep_map = {}

        for row_id, row in test_df.iterrows():
            if rep_boost and row["ID"] not in rep_genes and row["P"] > self.sig:
                continue

            chr_df = odds_df if row["#CHROM"] % 2 == 1 else evens_df
            if (self.merge_genes or row["ID"] in signal_genes) and chr_df.loc[row_id, "SIGNAL"]:
                continue

            x, gene = row["ROUNDED_X"], row["ID"]
            pos_mask    = chr_df["ROUNDED_X"].between(x - half_ld, x + half_ld)
            pos_idx     = chr_df.index[pos_mask]

            if rep_boost and self.merge_genes and np.any(chr_df.loc[pos_idx, "ID"].isin(rep_genes)):
                window_genes = chr_df.loc[pos_idx, ["ID", "P"]].set_index("ID")
                window_genes = window_genes[window_genes.index.isin(rep_genes)]
                new_gene     = window_genes.idxmin().values[0]
                target_df    = odds_df if row["#CHROM"] % 2 == 1 else evens_df
                target_df.loc[pos_idx, "ID"] = new_gene
                self.signal_rep_map[gene] = new_gene
                gene = new_gene

            target_df = odds_df if row["#CHROM"] % 2 == 1 else evens_df
            cur_rep   = target_df.loc[pos_idx, "Replication"]
            target_df.loc[pos_idx, "Replication"] = np.logical_or(cur_rep, gene in rep_genes)
            target_df.loc[pos_idx, "SIGNAL"] = True
            target_df.loc[pos_idx, "ID"]     = gene
            signal_genes.append(gene)

        odds_df  = odds_df[odds_df["SIGNAL"]]
        evens_df = evens_df[evens_df["SIGNAL"]]

        print("Due to signal merging and replication prioritization, the following genes were renamed:")
        print("\n".join(k + ": " + v for k, v in self.signal_rep_map.items()))

        if len(odds_df) == 0 and len(evens_df) == 0:
            raise ValueError("No signals to annotate. Try making P-value thresholds less stringent")

        return odds_df, evens_df

    def _find_signals_specific(self, signal_bed_df, rep_genes=[]):
        """Mark variants that overlap user-supplied BED regions as signals."""
        odds_df, evens_df = self._get_odds_evens()
        self.spec_genes = []

        for df in (odds_df, evens_df):
            df["SIGNAL"]      = False
            df["Replication"] = False

        signal_bed_df = (
            signal_bed_df[signal_bed_df["#CHROM"].isin(self.df["#CHROM"])]
            .copy()
            .sort_values(by=["#CHROM", "POS"])
        )

        for data_df in (odds_df, evens_df):
            n        = len(signal_bed_df)
            m        = len(data_df)
            shape_2d = (n, m)
            shape_t  = (m, n)

            search_starts = np.broadcast_to(signal_bed_df["START"], shape_t).T
            search_stops  = np.broadcast_to(signal_bed_df["END"],   shape_t).T
            search_chr    = np.broadcast_to(signal_bed_df["#CHROM"], shape_t).T
            data_pos      = np.broadcast_to(data_df["POS"],    shape_2d)
            data_chr      = np.broadcast_to(data_df["#CHROM"], shape_2d)

            overlap = (search_chr == data_chr) & (search_starts < data_pos) & (data_pos < search_stops)

            for i, (_, bed_row) in enumerate(signal_bed_df.iterrows()):
                keep_locs = data_df.index[overlap[i]]
                if len(keep_locs) == 0:
                    continue
                gene_df = data_df.loc[keep_locs].copy().reset_index(drop=False).set_index("index")
                gene = bed_row["ID"] if "ID" in bed_row.index else gene_df.sort_values(by="P")["ID"].iloc[0]

                print(gene, len(gene_df.index))
                self.thinned.loc[gene_df.index, "ID"] = gene
                data_df.loc[gene_df.index, "ID"] = gene
                if self.signal_color_col is not None and False in pd.isnull(gene_df[self.signal_color_col]):
                    self.thinned.loc[gene_df.index, self.signal_color_col] = gene_df[self.signal_color_col].mode().iloc[0]

                if self.thinned.loc[self.thinned["ID"] == gene, "P"].min() > self.sug:
                    continue

                self.spec_genes.append(gene)
                signal_idx = data_df.index.intersection(data_df.index[overlap[i]])
                data_df.loc[signal_idx, "SIGNAL"]      = True
                data_df.loc[signal_idx, "Replication"] = gene in rep_genes

        odds_df  = odds_df[odds_df["SIGNAL"]]
        evens_df = evens_df[evens_df["SIGNAL"]]
        print()
        return odds_df, evens_df

    def _plot_signals(self, odds_df, evens_df) -> None:
        colors_odd  = odds_df["Replication"].replace({True: self.REP_HIT_COLOR,  False: self.NOVEL_HIT_COLOR})
        colors_even = evens_df["Replication"].replace({True: self.REP_HIT_COLOR, False: self.NOVEL_HIT_COLOR})
        self.base_ax.scatter(odds_df[self.plot_x_col],  odds_df[self.plot_y_col],  c=colors_odd,  s=10)
        self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=colors_even, s=10)

    def _plot_color_signals(self, odds_df, evens_df, legend_loc=None) -> None:
        if self.phewas_rep_color_col is not None:
            odds_df  = odds_df[~odds_df[self.phewas_rep_color_col]].copy()
            evens_df = evens_df[~evens_df[self.phewas_rep_color_col]].copy()
            self.fig.set_size_inches(14.4, 8)

        unique_vals = sorted(set(self.thinned[self.signal_color_col].dropna().unique()))
        discrete    = not pd.api.types.is_numeric_dtype(odds_df[self.signal_color_col])

        if not discrete:
            color_min = min(odds_df[self.signal_color_col].quantile(0.05),  evens_df[self.signal_color_col].quantile(0.05))
            color_max = max(odds_df[self.signal_color_col].quantile(0.95),  evens_df[self.signal_color_col].quantile(0.95))
            kw = dict(cmap=plt.cm.get_cmap(self.COLOR_MAP), s=10, vmin=color_min, vmax=color_max)
            self.base_ax.scatter(odds_df[self.plot_x_col],  odds_df[self.plot_y_col],  c=odds_df[self.signal_color_col],  **kw)
            scat = self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=evens_df[self.signal_color_col], **kw)
            self.fig.colorbar(scat, cax=self.cbar_ax, orientation="horizontal")
        else:
            categories  = sorted(unique_vals)
            cat_to_num  = dict(zip(categories, np.arange(len(categories))))
            odds_df["Cat_Num"]  = odds_df[self.signal_color_col].replace(cat_to_num)
            evens_df["Cat_Num"] = evens_df[self.signal_color_col].replace(cat_to_num)

            if self.phewas_updown_col is None:
                if self.phewas_size_col is None:
                    odds_df["pt_sz"]  = 10
                    evens_df["pt_sz"] = 10
                else:
                    odds_df["pt_sz"]  = self._convert_linear_scale(odds_df[self.phewas_size_col].abs(),  self.MIN_PT_SZ, self.MAX_PT_SZ)
                    evens_df["pt_sz"] = self._convert_linear_scale(evens_df[self.phewas_size_col].abs(), self.MIN_PT_SZ, self.MAX_PT_SZ)

                use_cm = plt.cm.get_cmap(self.COLOR_MAP, len(categories))
                cmap_kw = dict(cmap=use_cm, vmin=0, vmax=len(categories) - 1)
                self.base_ax.scatter(odds_df[self.plot_x_col],  odds_df[self.plot_y_col],  c=odds_df["Cat_Num"],  s=odds_df["pt_sz"],  **cmap_kw)
                scat = self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=evens_df["Cat_Num"], s=evens_df["pt_sz"], **cmap_kw)

            else:  # directional triangle markers
                if self.phewas_size_col is None:
                    odds_df["pt_sz"]  = 60
                    evens_df["pt_sz"] = 60
                else:
                    odds_df["pt_sz"]  = self._convert_linear_scale(odds_df[self.phewas_size_col].abs(),  self.MIN_TRI_SZ, self.MAX_TRI_SZ)
                    evens_df["pt_sz"] = self._convert_linear_scale(evens_df[self.phewas_size_col].abs(), self.MIN_TRI_SZ, self.MAX_TRI_SZ)

                odds_df["up"]  = odds_df[self.phewas_updown_col]  > 0
                evens_df["up"] = evens_df[self.phewas_updown_col] > 0

                for src_df in (odds_df, evens_df):
                    for updown, sub_df in src_df.groupby("up"):
                        shape    = "^" if updown else "v"
                        cmap_obj = plt.cm.get_cmap(self.COLOR_MAP, len(categories))
                        if self.phewas_fill_col is None:
                            scat = self.base_ax.scatter(
                                sub_df[self.plot_x_col], sub_df[self.plot_y_col],
                                c=sub_df["Cat_Num"], cmap=cmap_obj,
                                s=sub_df["pt_sz"], marker=shape,
                                edgecolors="k", linewidth=0.3,
                            )
                        else:
                            print("Updown and Fill", flush=True)
                            edge_colors = sub_df["Cat_Num"].apply(lambda x: cmap_obj(x))
                            face_colors = sub_df[["Cat_Num", self.phewas_fill_col]].apply(
                                lambda r: cmap_obj(r["Cat_Num"]) if r[self.phewas_fill_col] else "none", axis=1
                            )
                            self.base_ax.scatter(
                                sub_df[self.plot_x_col], sub_df[self.plot_y_col],
                                s=sub_df["pt_sz"], marker=shape,
                                edgecolors=edge_colors.values, linewidth=1,
                                facecolors=face_colors.values,
                            )
                            fractions  = (np.arange(len(categories)) / len(categories)) + (0.5 / len(categories))
                            new_norm   = mpl.colors.BoundaryNorm(boundaries=np.arange(len(categories) + 1), ncolors=len(categories))
                            scat       = plt.cm.ScalarMappable(norm=new_norm, cmap=plt.cm.get_cmap(self.COLOR_MAP, len(categories)))

            self._add_color_bar(scat, categories, legend_loc=legend_loc)

    def _add_color_bar(self, mappable, categories, legend_loc=None) -> None:
        if legend_loc is None:
            cbar   = self.fig.colorbar(mappable, cax=self.cbar_ax, orientation="horizontal")
            xmin, xmax = self.cbar_ax.get_xlim()
            factor = (xmax - xmin) / len(categories)
            cats   = [c if len(c) < 20 else c[:17] + "..." for c in categories]
            cbar.set_ticks((np.arange(len(categories)) + 0.5) * factor + xmin)
            if not self.invert:
                cbar.ax.set_xticklabels(cats, rotation=30, ha="left")
                self.cbar_ax.xaxis.tick_top()
            else:
                cbar.ax.set_xticklabels(cats, rotation=30, ha="right")
            self.fig.tight_layout()
        else:
            plt.rc("legend", fontsize=12)
            cmap_obj = mappable.get_cmap()
            handles  = [mpatches.Patch(color=cmap_obj(i), label=cat) for i, cat in enumerate(categories)]
            if legend_loc == "side":
                nrows = 14
                self.cbar_ax.legend(handles=handles, loc="lower left", ncols=max(len(categories) // nrows, 1))
            elif legend_loc == "top":
                self.cbar_ax.legend(handles=handles, loc="lower center", ncols=self.TOP_LEGEND_COLS)
            self.cbar_ax.xaxis.set_visible(False)
            self.cbar_ax.yaxis.set_visible(False)
            self.cbar_ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
            self.fig.tight_layout()

    def _plot_pointers_only(self) -> None:
        signal_df = self.thinned[self.thinned[self.phewas_annotate_col]]
        for _, row in signal_df.iterrows():
            if self.vertical:
                pointer_x = (
                    signal_df[signal_df[self.plot_x_col] <= self.max_log_p][self.plot_x_col].max()
                    if self.max_log_p is not None
                    else signal_df[self.plot_x_col].max()
                )
                self.base_ax.plot([pointer_x, self.max_x], [row[self.plot_y_col], row[self.plot_y_col]], c="silver", linewidth=1.5)
            else:
                self.base_ax.plot([row[self.plot_x_col], row[self.plot_x_col]], [row[self.plot_y_col], self.max_y], c="silver", linewidth=1.5)
            self.annot_list.append(row)

    # ------------------------------------------------------------------
    # Private: signal-plot coordinate helpers (shared by signal_plot /
    #          signal_plot_with_specific to eliminate duplicate code)
    # ------------------------------------------------------------------

    def _build_signal_coords(self, signal_df: pd.DataFrame):
        """Compute relative per-signal x-coordinates and mid-points."""
        signal_order = signal_df["ID"].unique()
        signal_min   = signal_df.groupby("ID")["POS"].min().loc[signal_order]
        signal_max   = signal_df.groupby("ID")["POS"].max().loc[signal_order]
        signal_size  = signal_max - signal_min

        start_vals   = signal_size.cumsum().values[:-1]
        signal_start = pd.Series(data=start_vals, index=signal_size.index[1:])
        signal_start.loc[signal_size.index[0]] = 0
        signal_start = signal_start.loc[signal_size.index]
        signal_mid   = signal_start + signal_size / 2

        pos_adjust = -signal_min.loc[signal_df["ID"]] + signal_start.loc[signal_df["ID"]]
        signal_df["SIGNAL_X"]    = signal_df["POS"] + pos_adjust.values
        signal_df["SIGNAL_TEST"] = signal_df["POS"] - signal_min.loc[signal_df["ID"]].values
        self.df["SIGNAL_POS"]    = signal_df["SIGNAL_X"]

        if not self.vertical:
            self.plot_x_col = "SIGNAL_X"
        else:
            self.plot_y_col = "SIGNAL_X"

        return signal_df, signal_mid

    def _split_signal_odds_evens(self, signal_df: pd.DataFrame):
        signal_min   = signal_df.groupby("ID")["POS"].min()
        signal_max   = signal_df.groupby("ID")["POS"].max()
        signal_size  = signal_max - signal_min
        odd_signals  = signal_size.index[::2]
        even_signals = signal_size.index[1::2]
        return (
            signal_df[signal_df["ID"].isin(odd_signals)],
            signal_df[signal_df["ID"].isin(even_signals)],
        )

    def _scatter_signal_plot(self, odds_df, evens_df) -> None:
        if self.signal_color_col is None:
            self.base_ax.scatter(odds_df[self.plot_x_col],  odds_df[self.plot_y_col],  c=self.LIGHT_CHR_COLOR, s=25)
            self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c=self.DARK_CHR_COLOR,  s=25)
        else:
            self.base_ax.scatter(odds_df[self.plot_x_col],  odds_df[self.plot_y_col],  c="silver",  s=25)
            self.base_ax.scatter(evens_df[self.plot_x_col], evens_df[self.plot_y_col], c="dimgrey", s=25)
            for src, filtered in (
                (odds_df,  odds_df[odds_df["P"]   < 1e-3]),
                (evens_df, evens_df[evens_df["P"] < 1e-3]),
            ):
                pass  # color rendering happens in the last scatter call
            color_min = min(odds_df[self.signal_color_col].quantile(0.05),  evens_df[self.signal_color_col].quantile(0.05))
            color_max = max(odds_df[self.signal_color_col].quantile(0.95),  evens_df[self.signal_color_col].quantile(0.95))
            print(color_min, color_max)
            filt_odds  = odds_df[odds_df["P"]   < 1e-3]
            filt_evens = evens_df[evens_df["P"] < 1e-3]
            kw = dict(s=25, cmap=self.COLOR_MAP, vmin=color_min, vmax=color_max)
            self.base_ax.scatter(filt_odds[self.plot_x_col],  filt_odds[self.plot_y_col],  c=filt_odds[self.signal_color_col],  **kw)
            scat = self.base_ax.scatter(filt_evens[self.plot_x_col], filt_evens[self.plot_y_col], c=filt_evens[self.signal_color_col], **kw)
            self.fig.colorbar(scat, cax=self.cbar_ax, orientation="horizontal")

    def _draw_signal_pointers(self, annot_df: pd.DataFrame) -> None:
        for _, row in annot_df.iterrows():
            if self.vertical:
                self.base_ax.plot([row[self.plot_x_col], self.max_x], [row[self.plot_y_col], row[self.plot_y_col]], c="silver", linewidth=1.5)
            else:
                self.base_ax.plot([row[self.plot_x_col], row[self.plot_x_col]], [row[self.plot_y_col], self.max_y], c="silver", linewidth=1.5)

    # ------------------------------------------------------------------
    # Private: table rendering
    # ------------------------------------------------------------------

    def _plot_table_vertical(
        self,
        extra_cols: dict = {},
        number_cols: list = [],
        rep_genes: list = [],
        keep_chr_pos: bool = True,
        table_fontsize: int = DEFAULT_TABLE_FONTSIZE,
        legend_loc=None,
        specific_sig_df: Optional[pd.DataFrame] = None,
    ) -> None:
        if not self.annot_list:
            raise ValueError("No signals to annotate. Try making P-value thresholds less stringent")

        columns = (["ID", "CHR", "POS", "P"] if keep_chr_pos else ["ID", "P"])
        columns.extend(extra_cols.values())

        annot_table = pd.concat(self.annot_list, axis=1).transpose()
        annot_table = (
            annot_table
            .sort_values(by=["#CHROM", "POS"])
            .reset_index()
            .rename(columns={"#CHROM": "CHR", "index": "ID"})
            .rename(columns=extra_cols)
        )
        annot_table["P"]  = annot_table["P"].apply(lambda x: "{:.2e}".format(x))
        annot_table["ID"] = annot_table["ID"].apply(lambda x: r"$\it{" + x + r"}$")

        try:
            annot_table[number_cols] = annot_table[number_cols].map(lambda x: "{:.3}".format(x))
        except AttributeError:
            annot_table[number_cols] = annot_table[number_cols].applymap(lambda x: "{:.3}".format(x))

        # Escape underscores in non-numeric text columns for LaTeX rendering
        for col in annot_table.columns:
            try:
                pd.to_numeric(annot_table[col])
            except (ValueError, TypeError):
                annot_table[col] = annot_table[col].astype(str).str.replace("_", r"\_")

        location = "center left" if not self.invert else "center right"
        table = mpl.table.table(
            ax=self.table_ax,
            cellText=annot_table[columns].infer_objects(copy=False).fillna("").values,
            colLabels=columns,
            loc=location,
            colColours=[self.TABLE_HEAD_COLOR] * len(columns),
        )
        table.AXESPAD = 0
        table.auto_set_font_size(False)
        table.set_fontsize(table_fontsize)
        table.auto_set_column_width(col=list(range(len(annot_table.columns))))
        self.fig.tight_layout()

        self.table_ax.set_axis_off()
        self.table_ax.invert_yaxis()
        if self.invert:
            self.table_ax.invert_xaxis()

        h_factor    = table_fontsize / DEFAULT_TABLE_FONTSIZE
        cell_height = table[(0, 0)].get_height() * h_factor
        table.scale(1, h_factor)

        new_table_height = (len(annot_table) + 1) * cell_height
        table_min_y = 0.5 - 0.5 * new_table_height

        specific_loci = None if specific_sig_df is None else specific_sig_df.set_index("ID")

        for i in range(len(annot_table)):
            connection_row = annot_table.iloc[i]
            raw_id = connection_row["ID"]
            clean_id = raw_id.replace(r"$\it{", "").replace(r"}$", "")

            # Check SKIP_POINTER from specific_sig_df
            if (
                specific_loci is not None
                and clean_id in specific_loci.index
                and "SKIP_POINTER" in specific_loci.columns
                and specific_loci.loc[clean_id, "SKIP_POINTER"]
            ):
                print(f"Skipping connector for: {clean_id}")
                continue

            cell = table[(i + 1, 0)]
            cell.set_facecolor(
                self.REP_TABLE_COLOR if clean_id in rep_genes else self.NOVEL_TABLE_COLOR
            )
            connect_x = 0
            connect_y = table_min_y + (1.5 + i) * cell_height

            cp = ConnectionPatch(
                xyA=(self.max_x, connection_row[self.plot_y_col]),
                axesA=self.base_ax, coordsA="data",
                xyB=(connect_x, connect_y),
                axesB=self.table_ax, coordsB="data",
                arrowstyle="-", color="silver",
            )
            self.fig.add_artist(cp)

    def _plot_table_horizontal(
        self,
        rep_genes: list = [],
        with_table_bg: bool = True,
        with_table_grid: bool = True,
        text_rep_colors: bool = False,
        legend_loc=None,
        specific_sig_df: Optional[pd.DataFrame] = None,
    ) -> None:
        specific_loci = None if specific_sig_df is None else specific_sig_df.set_index("ID")

        if not self.annot_list and self.phewas_annotate_col is None:
            raise ValueError("No signals to annotate. Try making P-value thresholds less stringent")

        if self.phewas_annotate_col is None:
            annot_table = pd.concat(self.annot_list, axis=1).transpose()
        else:
            annot_table = self.thinned[self.thinned[self.phewas_annotate_col]].set_index("ID")

        annot_table = annot_table.sort_values(by=["#CHROM", "POS"])
        annot_table.index = [r"$\it{" + i + r"}$" for i in annot_table.index]
        genes    = [list(annot_table.index)]
        num_cols = len(annot_table)

        table = self.table_ax.table(
            cellText=genes,
            loc="lower center",
            colWidths=[1 / (num_cols + 2) for _ in genes[0]],
            cellLoc="center",
        )
        table.AXESPAD = 0
        self.table_ax.set_axis_off()
        self.fig.tight_layout()

        # TWAS color bar (if applicable)
        color_map = {}
        if self.twas_color_col is not None:
            if self.signal_color_col is None:
                unique_vals = sorted(annot_table[self.twas_color_col].unique())
            else:
                unique_vals = sorted(set(list(annot_table[self.twas_color_col].unique()) + list(self.thinned[self.signal_color_col].unique())))
            cmap         = plt.cm.get_cmap(self.COLOR_MAP, len(unique_vals))
            fractions    = (np.arange(len(unique_vals)) / len(unique_vals)) + 0.5 / len(unique_vals)
            colors_list  = [cmap(f) for f in fractions]
            color_map    = dict(zip(unique_vals, colors_list))
            fractions    = list(fractions) + [1.0]
            new_norm     = mpl.colors.BoundaryNorm(boundaries=np.arange(len(unique_vals) + 1), ncolors=len(unique_vals))
            new_mappable = plt.cm.ScalarMappable(norm=new_norm, cmap=plt.cm.get_cmap(self.COLOR_MAP, len(unique_vals)))
            self._add_color_bar(new_mappable, color_map.keys(), legend_loc=legend_loc)

        cell_width  = table[(0, 0)].get_width()
        cell_height = table[(0, 0)].get_height()

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
            cell_text_raw  = table[(0, i)].get_text().get_text()
            cell_text      = cell_text_raw[5:-2]  # strip $\it{...}$

            # SKIP_POINTER check
            skip = (
                specific_loci is not None
                and cell_text in specific_loci.index
                and "SKIP_POINTER" in specific_loci.columns
                and specific_loci.loc[cell_text, "SKIP_POINTER"]
            )
            if skip:
                print(f"Skipping connector for {cell_text}")
                continue

            is_rep = (cell_text in rep_genes) or (
                self.phewas_rep_color_col is not None and connection_row[self.phewas_rep_color_col]
            )

            if with_table_bg:
                table[(0, i)].set_facecolor(self.REP_TABLE_COLOR if is_rep else self.NOVEL_TABLE_COLOR)
            if text_rep_colors:
                table[(0, i)].get_text().set_color(self.DARK_CHR_COLOR if is_rep else self.NOVEL_TABLE_COLOR)

            connect_y = 0 if not self.invert else 1
            connect_x = (
                table[(0, i)].get_x() + 0.5 * cell_width if with_table_grid
                else table[(0, i)].get_x()
            )
            cp = ConnectionPatch(
                xyA=(connection_row[self.plot_x_col], self.max_y),
                axesA=self.base_ax, coordsA="data",
                xyB=(connect_x, connect_y),
                axesB=self.table_ax, coordsB="axes fraction",
                arrowstyle="-", color="silver",
            )

            if not with_table_grid:
                row_text_color = "dimgrey" if is_rep and text_rep_colors else (self.NOVEL_HIT_COLOR if text_rep_colors else "k")
                if not self.invert:
                    self.table_ax.text(connect_x - 0.005, connect_y, cell_text, ha="left", va="bottom", rotation=45, transform=self.table_ax.transAxes, color=row_text_color)
                else:
                    self.table_ax.text(connect_x + 0.005, connect_y, cell_text, ha="right", va="top",    rotation=45, transform=self.table_ax.transAxes, color=row_text_color)

            if self.twas_updown_col is not None:
                shape = "v" if connection_row[self.twas_updown_col] < 0 else "^"
                color = color_map.get(connection_row[self.twas_color_col], self.REP_HIT_COLOR if cell_text in rep_genes else self.NOVEL_HIT_COLOR)
                self.base_ax.scatter(connection_row[self.plot_x_col], connection_row[self.plot_y_col], color=color, marker=shape, s=60)

            self.fig.add_artist(cp)
