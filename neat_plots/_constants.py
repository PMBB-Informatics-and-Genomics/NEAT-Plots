"""
Shared constants for NEAT-Plots.

Chromosome lengths sourced from:
    https://en.wikipedia.org/wiki/Human_genome
"""

# ---------------------------------------------------------------------------
# Genome reference
# ---------------------------------------------------------------------------

# Chromosome lengths (GRCh38).  Chr 23 = X (treated as numeric internally).
CHR_LENGTHS: dict[int, int] = {
    1:  248956422,
    2:  242193529,
    3:  198295559,
    4:  190214555,
    5:  181538259,
    6:  170805979,
    7:  159345973,
    8:  145138636,
    9:  138394717,
    10: 133797422,
    11: 135086622,
    12: 133275309,
    13: 114364328,
    14: 107043718,
    15: 101991189,
    16:  90338345,
    17:  83257441,
    18:  80373285,
    19:  58617616,
    20:  64444167,
    21:  46709983,
    22:  50818468,
    23: 156040895,   # X chromosome
}

# ---------------------------------------------------------------------------
# Default color palette
# ---------------------------------------------------------------------------

DEFAULT_COLORS: dict[str, str] = {
    "DARK_CHR_COLOR":    "#5841bf",
    "LIGHT_CHR_COLOR":   "#648fff",
    "NOVEL_HIT_COLOR":   "#dc267f",
    "NOVEL_TABLE_COLOR": "#eb7fb3",
    "REP_HIT_COLOR":     "#ffbb00",
    "REP_TABLE_COLOR":   "#ffdc7a",
    "FIFTH_COLOR":       "#d45c00",  # threshold line / accent
    "TABLE_HEAD_COLOR":  "#9e9e9e",
    "COLOR_MAP":         "turbo_r",  # matplotlib colormap name
}

# ---------------------------------------------------------------------------
# Display geometry
# ---------------------------------------------------------------------------

DEFAULT_TABLE_FONTSIZE: int = 12

# Genomic-position rounding bucket for thinning (50 kb)
CHR_POS_ROUND: float = 5e4

# Scatter-point size range for continuous-scale features
MIN_PT_SZ:  int = 5
MAX_PT_SZ:  int = 200

# Triangle marker size range (directional effect plots)
MIN_TRI_SZ: int = 5
MAX_TRI_SZ: int = 200

# Maximum columns in a top-positioned legend
TOP_LEGEND_COLS: int = 6
