"""
BasePlot — shared data-handling and utility logic for NEAT-Plots.

Both ManhattanPlot and BoroughsPlot inherit from this class.
Rendering methods (plot_data, plot_annotations, full_plot, …) live in the
subclasses because they differ meaningfully between single-axis and
multi-facet (boroughs) layouts.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from ._constants import (
    CHR_LENGTHS,
    CHR_POS_ROUND,
    DEFAULT_COLORS,
    DEFAULT_TABLE_FONTSIZE,
    MIN_PT_SZ,
    MAX_PT_SZ,
    MIN_TRI_SZ,
    MAX_TRI_SZ,
    TOP_LEGEND_COLS,
)

logger = logging.getLogger(__name__)


class BasePlot:
    """
    Abstract base class providing data loading, cleaning, thinning,
    annotation, QQ plotting, and shared utilities.

    Subclasses must implement the rendering pipeline:
        plot_data, plot_sig_signals, plot_annotations, plot_table, full_plot.
    """

    # ------------------------------------------------------------------
    # Class-level defaults (overridden per-instance via update_plotting_parameters)
    # ------------------------------------------------------------------

    df: Optional[pd.DataFrame] = None
    thinned: Optional[pd.DataFrame] = None

    sig: float = 5e-8
    sig_line: float = 5e-8
    sug: float = 1e-5
    annot_thresh: float = 5e-8

    annotate: bool = True
    signal_color_col: Optional[str] = None
    twas_color_col: Optional[str] = None
    twas_updown_col: Optional[str] = None

    ld_block: float = 4e5
    plot_x_col: str = "ROUNDED_Y"
    plot_y_col: str = "ROUNDED_X"
    chr_ticks: list = []
    max_x: float = 10
    max_y: float = 10

    invert: bool = False
    merge_genes: bool = False
    max_log_p: Optional[float] = None
    signal_rep_map: dict = {}

    fig = None
    base_ax = None
    table_ax = None
    cbar_ax = None
    lower_base_ax = None
    upper_base_ax = None
    annot_list: list = []
    spec_genes: list = []

    # Color palette — mirrored from DEFAULT_COLORS for direct attribute access
    DARK_CHR_COLOR:    str = DEFAULT_COLORS["DARK_CHR_COLOR"]
    LIGHT_CHR_COLOR:   str = DEFAULT_COLORS["LIGHT_CHR_COLOR"]
    NOVEL_HIT_COLOR:   str = DEFAULT_COLORS["NOVEL_HIT_COLOR"]
    NOVEL_TABLE_COLOR: str = DEFAULT_COLORS["NOVEL_TABLE_COLOR"]
    REP_HIT_COLOR:     str = DEFAULT_COLORS["REP_HIT_COLOR"]
    REP_TABLE_COLOR:   str = DEFAULT_COLORS["REP_TABLE_COLOR"]
    FIFTH_COLOR:       str = DEFAULT_COLORS["FIFTH_COLOR"]
    TABLE_HEAD_COLOR:  str = DEFAULT_COLORS["TABLE_HEAD_COLOR"]
    COLOR_MAP:         str = DEFAULT_COLORS["COLOR_MAP"]

    CHR_POS_ROUND: float = CHR_POS_ROUND
    MIN_PT_SZ:     int   = MIN_PT_SZ
    MAX_PT_SZ:     int   = MAX_PT_SZ
    MIN_TRI_SZ:    int   = MIN_TRI_SZ
    MAX_TRI_SZ:    int   = MAX_TRI_SZ
    TOP_LEGEND_COLS: int = TOP_LEGEND_COLS

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        file_path: str,
        test_rows: Optional[int] = None,
        title: str = "Manhattan Plot",
    ) -> None:
        """
        Parameters
        ----------
        file_path:
            Path to the summary statistics file (tab/space/comma delimited,
            or a ``.pickle`` file).
        test_rows:
            If set, only the first *N* rows are loaded (useful for
            development/testing).
        title:
            Plot title used by all high-level plot methods.
        """
        self.path = file_path
        self.title = title
        self.test_rows = test_rows
        mpl.rcParams.update({"font.size": 14})

    # ------------------------------------------------------------------
    # Color configuration
    # ------------------------------------------------------------------

    def config_colors(self, color_file_json: str) -> None:
        """
        Load a custom color palette from a JSON file.

        The JSON should be an object whose keys match the color attribute
        names (e.g. ``DARK_CHR_COLOR``, ``COLOR_MAP``).

        Parameters
        ----------
        color_file_json:
            Path to a JSON file mapping color-attribute names to values.
        """
        with open(color_file_json) as fh:
            color_config_dict: dict = json.load(fh)
        for k, v in color_config_dict.items():
            setattr(self, k, v)

    def reset_colors(self) -> None:
        """Reset all color attributes to the package defaults."""
        for k, v in DEFAULT_COLORS.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, delim: str = r"\s+") -> None:
        """
        Read the summary statistics file into ``self.df``.

        Parameters
        ----------
        delim:
            Column delimiter passed to :func:`pandas.read_table`.
            Defaults to any whitespace (handles both space and tab).
        """
        if ".pickle" in self.path:
            self.df = pd.read_pickle(self.path).reset_index()
            if self.test_rows is not None:
                self.df = self.df.sort_values(by=["#CHROM"]).iloc[: int(self.test_rows)]
        else:
            self.df = pd.read_table(
                self.path, sep=delim, nrows=self.test_rows, low_memory=False
            )

        self.df.index = np.arange(len(self.df))
        logger.info("Loaded %d rows", len(self.df))
        logger.debug("Columns: %s", list(self.df.columns))
        print("Loaded", len(self.df), "Rows")
        print(self.df.columns)

    # ------------------------------------------------------------------
    # Data cleaning
    # ------------------------------------------------------------------

    def clean_data(
        self,
        col_map: Optional[dict] = None,
        logp: Optional[str] = None,
    ) -> None:
        """
        Normalise the loaded DataFrame for plotting.

        Renames columns, filters to autosomes + X, converts types, and
        replaces zero p-values to avoid ``log10(0)`` errors.

        Parameters
        ----------
        col_map:
            Mapping from existing column names to the required names
            (``#CHROM``, ``POS``, ``ID``, ``P``).
        logp:
            Name of a column already in ``-log10(p)`` scale.  If given,
            the ``P`` column is derived from it.
        """
        if col_map is not None:
            # If the mapping would create a duplicate P column, drop the old one
            if "P" in col_map.values() and "P" in self.df.columns:
                self.df = self.df.drop(columns="P")
            col_map = {k: v for k, v in col_map.items() if k in self.df.columns}
            self.df = self.df.rename(columns=col_map)

        if logp is not None:
            self.df["P"] = 10 ** -self.df[logp]

        # Normalise chromosome column: strip "chr" prefix, recode X → 23
        self.df["#CHROM"] = self.df["#CHROM"].astype(str).str.replace("chr", "", regex=False)
        valid_chrs = list(range(1, 23)) + [str(i) for i in range(1, 23)] + ["X"]
        self.df = self.df[self.df["#CHROM"].isin(valid_chrs)]
        self.df["#CHROM"] = self.df["#CHROM"].replace("X", 23).astype(int)

        self.df["POS"] = self.df["POS"].astype(float).astype(int)
        self.df = self.df.sort_values(by=["#CHROM", "POS"])
        self.df["ID"] = self.df["ID"].fillna("")

        self.df["P"] = pd.to_numeric(self.df["P"], errors="coerce")
        # Replace exact zeros to avoid -log10(0) = inf
        p_min = self.df["P"].replace(0, np.nan).min()
        self.df["P"] = self.df["P"].replace(0, p_min / 100)

    def check_data(self) -> None:
        """Print the first/last rows and row count for a quick sanity check."""
        print(self.df.head()[["#CHROM", "POS", "P", "ID"]])
        print(self.df.tail()[["#CHROM", "POS", "P", "ID"]])
        print(len(self.df))

    # ------------------------------------------------------------------
    # Annotation merging
    # ------------------------------------------------------------------

    def add_annotations(
        self,
        annot_df: pd.DataFrame,
        extra_cols: list = [],
    ) -> None:
        """
        Merge a gene-annotation table into ``self.df``.

        The annotation DataFrame must contain ``#CHROM``, ``POS``, and
        ``ID`` columns (plus any ``extra_cols``).

        Parameters
        ----------
        annot_df:
            Annotation table to merge.
        extra_cols:
            Additional columns from *annot_df* to carry over.
        """
        annot_cols = ["#CHROM", "POS", "ID"] + list(extra_cols)
        self.df = self.df.drop(columns="ID_y", errors="ignore")
        annot_df = annot_df.copy()
        annot_df["#CHROM"] = annot_df["#CHROM"].replace("X", 23).astype(int)
        self.df = self.df.merge(annot_df[annot_cols], on=["#CHROM", "POS"], how="left")
        # Fill ID_x with ID_y where ID_y is not NaN (pandas 3.0-safe pattern)
        self.df["ID_x"] = self.df["ID_x"].where(self.df["ID_y"].isna(), self.df["ID_y"])
        self.df = self.df.drop(columns="ID_y").rename(columns={"ID_x": "ID"})

    # ------------------------------------------------------------------
    # Thinning
    # ------------------------------------------------------------------

    def get_thinned_data(
        self,
        log_p_round: int = 2,
        additional_cols: list = [],
    ) -> None:
        """
        Thin the data for display by de-duplicating on rounded position and
        ``-log10(P)``.

        Results are stored in ``self.thinned``.  ROUNDED_X and ROUNDED_Y are
        added directly to ``self.df`` (no full-frame copy), keeping peak RAM
        at roughly 2× the loaded DataFrame size rather than 3×.

        Parameters
        ----------
        log_p_round:
            Decimal places to round ``-log10(P)`` before de-duplication.
        additional_cols:
            Extra columns to include in the uniqueness key.
        """
        if "ABS_POS" not in self.df.columns:
            self.df["ABS_POS"] = self._get_absolute_positions(self.df)

        # Mutate self.df in place — no copy needed before thinning
        self.df["ROUNDED_X"] = (
            self.df["ABS_POS"] // self.CHR_POS_ROUND * self.CHR_POS_ROUND
        )
        self.df["ROUNDED_Y"] = (
            pd.Series(-np.log10(self.df["P"]), index=self.df.index).round(log_p_round)
        )
        subset_cols = ["ROUNDED_X", "ROUNDED_Y"] + list(additional_cols)
        # sort_values creates a single sorted frame; drop_duplicates then
        # reduces it — peak is ~2× df rather than the previous ~3×
        self.thinned = self.df.sort_values(by="P").drop_duplicates(subset=subset_cols)
        logger.info("%d variants after thinning", len(self.thinned))
        print(len(self.thinned), "Variants After Thinning")

    def load_and_thin(
        self,
        col_map: Optional[dict] = None,
        logp: Optional[str] = None,
        log_p_round: int = 2,
        additional_cols: list = [],
        chunksize: int = 500_000,
        delim: str = r"\s+",
    ) -> None:
        """
        Memory-efficient single-pass load + thin for large summary-stat files.

        Instead of reading the entire file into ``self.df`` and then calling
        :meth:`get_thinned_data` (which sorts the full DataFrame, peaking at
        ~2–3× RAM), this method reads *chunksize* rows at a time, applies
        cleaning and thinning to each chunk, and accumulates only the thinned
        rows.  A final global de-duplication step is applied to the concatenated
        per-chunk results.

        After the call:

        * ``self.df`` and ``self.thinned`` both point to the globally-thinned
          result — the full raw DataFrame is **never** assembled in memory.
        * ``self._all_p_values`` holds every raw P-value (one column) for use
          by :meth:`qq_plot`.

        Parameters
        ----------
        col_map:
            Column rename map (same semantics as :meth:`clean_data`).
        logp:
            Name of a column already in ``-log10(p)`` scale; derives ``P``
            when supplied.
        log_p_round:
            Decimal places for ``ROUNDED_Y``.
        additional_cols:
            Extra columns to include in the thinning uniqueness key.
        chunksize:
            Rows to read per chunk.  500 000 works well for most systems;
            reduce for very tight memory budgets.
        delim:
            Column delimiter passed to :func:`pandas.read_table`.
        """
        offsets, _ = self._build_chr_offset_map()
        valid_chrs  = list(range(1, 23)) + [str(i) for i in range(1, 23)] + ["X"]
        subset_cols = ["ROUNDED_X", "ROUNDED_Y"] + list(additional_cols)

        thinned_chunks: list = []
        all_p_vals:     list = []

        reader = pd.read_table(self.path, sep=delim, chunksize=chunksize, low_memory=False)

        for chunk_idx, chunk in enumerate(reader):
            # --- column rename ---
            if col_map is not None:
                if "P" in col_map.values() and "P" in chunk.columns:
                    chunk = chunk.drop(columns="P")
                _map = {k: v for k, v in col_map.items() if k in chunk.columns}
                chunk = chunk.rename(columns=_map)

            # --- logp → P ---
            if logp is not None:
                chunk["P"] = 10 ** -chunk[logp]

            # --- chromosome normalisation ---
            chunk["#CHROM"] = chunk["#CHROM"].astype(str).str.replace("chr", "", regex=False)
            chunk = chunk[chunk["#CHROM"].isin(valid_chrs)]
            if chunk.empty:
                continue
            chunk["#CHROM"] = chunk["#CHROM"].replace("X", 23).astype(int)

            # --- type coercion ---
            chunk["POS"] = chunk["POS"].astype(float).astype(int)
            if "ID" in chunk.columns:
                chunk["ID"] = chunk["ID"].fillna("")
            chunk["P"] = pd.to_numeric(chunk["P"], errors="coerce")
            p_min = chunk["P"].replace(0, np.nan).min()
            if pd.notna(p_min):
                chunk["P"] = chunk["P"].replace(0, p_min / 100)

            all_p_vals.append(chunk["P"].dropna())

            # --- absolute positions + rounding ---
            chr_offset         = chunk["#CHROM"].map(offsets)
            chunk["ABS_POS"]   = (chunk["POS"].values + chr_offset.values).astype(np.int64)
            chunk["ROUNDED_X"] = chunk["ABS_POS"] // self.CHR_POS_ROUND * self.CHR_POS_ROUND
            chunk["ROUNDED_Y"] = (
                pd.Series(-np.log10(chunk["P"]), index=chunk.index).round(log_p_round)
            )

            # --- within-chunk thin ---
            chunk_thinned = chunk.sort_values(by="P").drop_duplicates(subset=subset_cols)
            thinned_chunks.append(chunk_thinned)
            logger.debug(
                "Chunk %d: %d rows → %d thinned", chunk_idx, len(chunk), len(chunk_thinned)
            )

        if not thinned_chunks:
            raise ValueError("No valid rows found; check file path and col_map.")

        # --- global de-duplication across chunks ---
        combined = pd.concat(thinned_chunks, ignore_index=True)
        combined = (
            combined.sort_values(by="P")
            .drop_duplicates(subset=subset_cols)
            .sort_values(by=["#CHROM", "POS"])
            .reset_index(drop=True)
        )

        # In chunked mode self.df == self.thinned (full raw DF never held)
        self.df = combined
        # Recompute chr_ticks from assembled data (updates self.chr_ticks)
        self.df["ABS_POS"] = self._get_absolute_positions(self.df)
        self.thinned = self.df  # same object

        # Stash all raw P-values for qq_plot (one float column — small)
        self._all_p_values: Optional[pd.Series] = pd.concat(all_p_vals, ignore_index=True)

        n_chunks = chunk_idx + 1
        logger.info(
            "load_and_thin: %d thinned variants from %d chunks", len(self.thinned), n_chunks
        )
        print(len(self.thinned), "Variants After Chunked Thinning", f"({n_chunks} chunks)")

    def prepare(
        self,
        col_map: Optional[dict] = None,
        logp: Optional[str] = None,
        log_p_round: int = 2,
        additional_cols: list = [],
        annot_df: Optional[pd.DataFrame] = None,
        annot_extra_cols: list = [],
        chunked: bool = False,
        chunksize: int = 500_000,
        delim: str = r"\s+",
    ) -> None:
        """
        Single-call preprocessing pipeline: load → clean → annotate → thin.

        Replaces the four-step boilerplate::

            mp.load_data()
            mp.clean_data(col_map={"CHR": "#CHROM", "BP": "POS", "PVAL": "P"})
            mp.add_annotations(annot_df)   # optional
            mp.get_thinned_data()

        with a single call::

            mp.prepare(col_map={"CHR": "#CHROM", "BP": "POS", "PVAL": "P"},
                       annot_df=annot_df)

        Parameters
        ----------
        col_map:
            Column rename map (passed to :meth:`clean_data` or
            :meth:`load_and_thin`).
        logp:
            Column already in ``-log10(p)`` scale.
        log_p_round:
            Decimal places for ``ROUNDED_Y``.
        additional_cols:
            Extra columns in the thinning uniqueness key.
        annot_df:
            Optional gene-annotation DataFrame passed to
            :meth:`add_annotations`.
        annot_extra_cols:
            Extra columns to carry over from *annot_df*.
        chunked:
            ``True`` → use :meth:`load_and_thin` (memory-efficient, suitable
            for large files in Nextflow pipelines).  ``False`` → standard
            :meth:`load_data` → :meth:`clean_data` → :meth:`get_thinned_data`
            pipeline.
        chunksize:
            Rows per chunk when ``chunked=True``.
        delim:
            Column delimiter.
        """
        if chunked:
            self.load_and_thin(
                col_map=col_map,
                logp=logp,
                log_p_round=log_p_round,
                additional_cols=additional_cols,
                chunksize=chunksize,
                delim=delim,
            )
            if annot_df is not None:
                self.add_annotations(annot_df, extra_cols=annot_extra_cols)
        else:
            self.load_data(delim=delim)
            self.clean_data(col_map=col_map, logp=logp)
            if annot_df is not None:
                self.add_annotations(annot_df, extra_cols=annot_extra_cols)
            self.get_thinned_data(
                log_p_round=log_p_round,
                additional_cols=additional_cols,
            )

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def print_hits(self) -> None:
        """Print significant and suggestive hits to stdout."""
        df = self.df.set_index("ID")
        sorted_df = (
            df[df["P"] < self.sug]
            .sort_values(by="P", ascending=True)
        )
        sorted_df = sorted_df[~sorted_df.index.duplicated(keep="first")]

        keep_cols = ["#CHROM", "POS", "P", "ABS_POS"]
        print_cols = ["#CHROM", "POS", "P"]

        for col_attr in ("signal_color_col", "twas_color_col", "twas_updown_col"):
            col = getattr(self, col_attr)
            if col is not None:
                keep_cols.append(col)
                print_cols.append(col)

        sig_df = sorted_df.loc[sorted_df["P"] <= self.sig, keep_cols]
        print("Significant:")
        print("\n".join(self._fmt_print_rows(sig_df[print_cols])))
        print()
        print("Suggestive:")
        print(
            "\n".join(
                self._fmt_print_rows(sorted_df.loc[sorted_df["P"] > self.sig, print_cols])
            )
        )
        print()

    def check_plotting_parameters(self) -> None:
        """Print current plotting parameter values."""
        params = {
            "Significance Threshold": self.sig,
            "Suggestive Threshold": self.sug,
            "Annotation Threshold": self.annot_thresh,
            "LD Block Width": self.ld_block,
            "Annotating?": self.annotate,
            "Maximum Neg. Log P-Val": self.max_log_p,
        }
        if self.signal_color_col is not None:
            params["Edge Color Column"] = self.signal_color_col
        print(params)

    def update_plotting_parameters(self, **kwargs) -> None:
        """
        Update one or more plotting parameters.

        Accepts the same keyword arguments as the subclass implementations;
        unknown keys are silently ignored by the base class so that
        subclass-specific parameters (e.g. ``vertical``, ``phewas_*``) can
        be passed through ``super()``.

        Common parameters
        -----------------
        sig, sug, annot_thresh : float
            P-value thresholds.
        ld_block : float
            LD block half-width for signal merging (bp).
        annotate : bool
        merge_genes : bool
        invert : bool
        max_log_p : float or None
        title : str
        signal_color_col, twas_color_col, twas_updown_col : str or None
        """
        _set = self._update_param
        self.annotate       = _set(self.annotate,       kwargs.get("annotate",       ""))
        self.ld_block       = _set(self.ld_block,       kwargs.get("ld_block",       ""))
        self.title          = _set(self.title,          kwargs.get("title",          ""))
        self.signal_color_col = _set(self.signal_color_col, kwargs.get("signal_color_col", ""))
        self.twas_updown_col  = _set(self.twas_updown_col,  kwargs.get("twas_updown_col",  ""))
        self.twas_color_col   = _set(self.twas_color_col,   kwargs.get("twas_color_col",   ""))
        self.sig            = _set(self.sig,            kwargs.get("sig",            ""))
        self.sug            = _set(self.sug,            kwargs.get("sug",            ""))
        self.annot_thresh   = _set(self.annot_thresh,   kwargs.get("annot_thresh",   ""))
        self.max_log_p      = _set(self.max_log_p,      kwargs.get("max_log_p",      ""))
        self.invert         = _set(self.invert,         kwargs.get("invert",         ""))
        self.merge_genes    = _set(self.merge_genes,    kwargs.get("merge_genes",    ""))

    # ------------------------------------------------------------------
    # QQ plot
    # ------------------------------------------------------------------

    def qq_plot(
        self,
        save: Optional[str] = None,
        save_res: int = 150,
        with_title: bool = True,
        steps: int = 30,
        additional_series: Optional[dict] = None,
    ) -> None:
        """
        Generate a quantile-quantile (QQ) plot and optionally save it.

        Lambda GC (genomic inflation factor) is shown in each series' legend
        entry.  Pass *additional_series* to overlay multiple P-value
        distributions on the same axes — useful for comparing subgroups,
        conditions, or analysis methods.

        Parameters
        ----------
        save:
            Output file path.  If provided the figure *and* the primary
            series' underlying data (as ``.csv``) are written to disk.
        save_res:
            DPI for the saved figure.
        with_title:
            Whether to add ``self.title`` as the plot title.
        steps:
            Unused; kept for API compatibility.
        additional_series:
            Optional mapping of ``{label: pd.Series}`` where each Series
            contains raw P-values to overlay.  Example::

                mp.qq_plot(
                    additional_series={
                        "Males":   males_pval_series,
                        "Females": females_pval_series,
                    }
                )

            Lambda GC for each series is appended to its legend label.
            The primary series (``self.df["P"]`` or the stashed chunked
            P-values) is always plotted first under the label
            ``"Primary"``.
        """
        # In chunked (load_and_thin) mode self._all_p_values holds every raw
        # P-value; self.df only contains thinned rows.  Fall back gracefully.
        _p_source = getattr(self, "_all_p_values", None)
        primary_p = (_p_source.dropna().copy() if _p_source is not None
                     else self.df["P"].dropna().copy())

        logger.info("%d SNPs dropped for missing P", len(self.df) - len(primary_p))
        print(len(self.df) - len(primary_p), "SNPs Dropped for Missing P")

        # Build ordered dict of all series to plot
        primary_label = "Primary" if additional_series else "GWAS Results"
        all_series: dict = {primary_label: primary_p}
        if additional_series:
            all_series.update(additional_series)

        # ------------------------------------------------------------------
        # Figure setup
        # ------------------------------------------------------------------
        fig = plt.gcf()
        fig.set_size_inches(6, 6)
        fig.set_facecolor("w")

        max_obs_log = 0.0
        max_exp_log = 0.0
        plot_dfs: dict = {}

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for idx, (label, p_series) in enumerate(all_series.items()):
            p_series = p_series.dropna()
            M = len(p_series)

            chi2_med  = chi2.ppf((1 - p_series).median(), df=1)
            lambda_gc = chi2_med / chi2.ppf(0.5, 1)
            print(f"Lambda GC [{label}]: {lambda_gc:.5f}")

            max_exp  = -np.log10(1 / M)
            frac_seq = (np.arange(M) + 1) / M
            p_vec    = p_series.sort_values()

            plot_df = p_vec.to_frame(name="P")
            plot_df["Exp P"] = frac_seq
            plot_df[["Log P", "Log Exp P"]] = (
                -np.log10(plot_df[["P", "Exp P"]]).round(3)
            )
            plot_df = plot_df.drop_duplicates(subset=["Log P", "Log Exp P"])
            plot_dfs[label] = plot_df

            scatter_label = f"{label}  (λ={lambda_gc:.4f})"
            plt.scatter(
                plot_df["Log Exp P"], plot_df["Log P"],
                c=colors[idx % len(colors)],
                label=scatter_label,
                s=16,
            )

            max_obs_log = max(max_obs_log, float(plot_df["Log P"].max()))
            max_exp_log = max(max_exp_log, max_exp)

        # Null-model diagonal
        diag_max = np.ceil(max_exp_log)
        plt.plot(
            [0, diag_max], [0, diag_max],
            c="r", label="Null Model", linestyle="dashed",
        )

        plt.xlabel("Expected −log₁₀(P) Quantiles")
        plt.ylabel("Observed −log₁₀(P) Quantiles")
        plt.legend(loc="upper left")
        if with_title:
            plt.title("QQ Plot:\n" + self.title)

        plt.xlim(0, diag_max)
        plt.ylim(0, np.ceil(max_obs_log))
        plt.tight_layout()

        if save is not None:
            plt.savefig(save, dpi=save_res, bbox_inches="tight")
            # Save the primary series CSV (backward compat)
            primary_df = plot_dfs[primary_label]
            primary_df.to_csv(save.replace(".png", ".csv"))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_thinned_df(self, path: str, pickle: bool = True) -> None:
        """
        Write ``self.thinned`` to disk.

        Parameters
        ----------
        path:
            Output file path.
        pickle:
            If ``True`` (default) write a ``.pickle`` file; otherwise write CSV.
        """
        if pickle:
            self.thinned.to_pickle(path)
        else:
            self.thinned.to_csv(path)

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chr_offset_map() -> tuple:
        """
        Precompute per-chromosome start offsets and lengths from ``CHR_LENGTHS``.

        Uses only the static ``CHR_LENGTHS`` constant — no DataFrame dependency —
        so the result can be computed before (or during) chunked loading.

        Returns
        -------
        offsets : pd.Series
            Cumulative genome start position for each chromosome (index 1–23).
        lengths : pd.Series
            Canonical chromosome length for each chromosome (index 1–23).
        """
        chrs = sorted(CHR_LENGTHS.keys())          # [1, 2, ..., 23]
        lengths = pd.Series(CHR_LENGTHS, index=chrs, dtype=np.int64)
        # offset[c] = sum of lengths for chromosomes 1 .. c-1
        cumlen = lengths.cumsum()
        offsets = pd.Series(np.int64(0), index=chrs, dtype=np.int64)
        offsets.iloc[1:] = cumlen.values[:-1]
        return offsets, lengths

    def _get_absolute_positions(self, active_df: pd.DataFrame) -> pd.Series:
        """
        Compute genome-wide absolute positions for *active_df* and update
        ``self.chr_ticks`` for axis labelling.

        Offsets are derived entirely from the static ``CHR_LENGTHS`` table, so
        this method no longer reads ``self.df`` and can safely be called on
        individual chunks during streaming / chunked loading.

        Parameters
        ----------
        active_df:
            DataFrame containing ``#CHROM`` and ``POS`` columns.

        Returns
        -------
        pd.Series
            Absolute genomic position for every row of *active_df*,
            aligned to *active_df*'s index.
        """
        offsets, lengths = self._build_chr_offset_map()

        # Chromosome tick marks at midpoints of each chromosome present in
        # active_df (or the full genome if called before filtering)
        present = sorted(active_df["#CHROM"].unique())
        tick_locs = (offsets.loc[present] + lengths.loc[present] / 2).values
        self.chr_ticks = [tick_locs, pd.Index(present)]

        # Map each variant to its chromosome's start offset, then add POS
        chr_offset = active_df["#CHROM"].map(offsets)
        return (active_df["POS"].values + chr_offset.values).astype(np.int64)

    def _update_param(self, old, new):
        """Return *new* if it is not the sentinel empty string, else *old*."""
        if new != "" and new != old:
            return new
        return old

    def _fmt_print_rows(self, print_df: pd.DataFrame) -> pd.Series:
        """Format a DataFrame as tab-separated strings, one row per line."""
        return print_df.apply(
            lambda x: x.name + "\t" + "\t".join(x.astype(str)), axis=1
        )

    def _convert_linear_scale(
        self,
        data: pd.Series,
        new_min: float,
        new_max: float,
    ) -> pd.Series:
        """
        Linearly rescale *data* to the range [*new_min*, *new_max*].

        Parameters
        ----------
        data:
            Input numeric series.
        new_min, new_max:
            Target range boundaries.
        """
        a_range = data.max() - data.min()
        a_min = data.min()
        new_range = new_max - new_min
        return (((data - a_min) / a_range) * new_range) + new_min
