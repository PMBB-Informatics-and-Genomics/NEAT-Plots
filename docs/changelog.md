# Changelog

All notable changes to NEAT-Plots are documented here.
This project follows [Semantic Versioning](https://semver.org/).

---

## [0.2.0] — *in preparation*

### Added

- **`prepare()`** — single-call preprocessing pipeline replacing the four-step
  `load_data → clean_data → add_annotations → get_thinned_data` boilerplate.
  Accepts `col_map`, `logp`, `annot_df`, `chunked`, and `chunksize` arguments.

- **`load_and_thin()`** — memory-efficient chunked loader that reads the input
  file in `chunksize`-row batches, thins each batch, and accumulates only the
  thinned rows. Peak RAM drops from ~3× to ~1–2× the thinned result size —
  suitable for 50M+ variant files in Nextflow / HPC pipelines.

- **Multi-series QQ plot** — `qq_plot(additional_series={"Males": p_series, ...})`
  overlays any number of P-value distributions on a single QQ plot, each with
  its own colour and lambda GC annotation in the legend.

- **`_build_chr_offset_map()`** — static method that precomputes chromosome
  start offsets from `CHR_LENGTHS` alone (no DataFrame dependency), enabling
  the chunked loader to compute absolute positions chunk-by-chunk.

- **Test suite** — 78 unit tests (`tests/`) and 20 integration tests
  (`tests/integration/`) against a synthetic chr22 GWAS fixture.

- **Documentation** — full MkDocs + Material site with usage guide, gallery,
  and auto-generated API reference.

### Changed

- `get_thinned_data()` now adds `ROUNDED_X` and `ROUNDED_Y` directly to
  `self.df` in-place (no `.copy()`), reducing peak RAM from ~3× to ~2× the
  loaded DataFrame.

- `_get_absolute_positions()` refactored to use static `CHR_LENGTHS` offsets
  only (no `self.df` dependency), enabling use on individual chunks.

- `add_annotations()` updated to the pandas 3.0-safe `.where()` pattern
  (eliminates `FutureWarning` about chained assignment).

- `qq_plot()` automatically uses `self._all_p_values` when available (set by
  the chunked path) so QQ plots remain accurate even when `self.df` only holds
  thinned rows.

- Missing `vertical=True, with_table=False` axis configuration now handled in
  `ManhattanPlot._config_axes()` (previously raised `ValueError`).

### Fixed

- `config_colors()` and `reset_colors()` previously iterated over dict keys
  only (missing `.items()`), causing `TypeError` on unpacking — fixed.

---

## [0.1.0] — initial release

- `ManhattanPlot` and `BoroughsPlot` extracted from the legacy
  `manhattan-plot/manhattan_plot.py` monolith into the `neat_plots` package.
- Shared data-handling logic moved to `neat_plots/_base.py` (`BasePlot`).
- Shared constants moved to `neat_plots/_constants.py`.
- Backward-compatibility shim added so `from manhattan_plot import ManhattanPlot`
  continues to work unchanged.
- `pyproject.toml` with PEP 517/518 build system and `pip install neat-plots[gui|dev|docs]` extras.
