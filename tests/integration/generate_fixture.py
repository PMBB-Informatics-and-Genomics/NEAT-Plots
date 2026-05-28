"""
Generate the chr22 synthetic GWAS fixture for integration tests.

Run once (or after deleting the .tsv.gz) from the repo root:

    python tests/integration/generate_fixture.py

Output: tests/integration/chr22_synthetic_gwas.tsv.gz  (~233 KB)

The fixture contains 10,000 variants on GRCh38 chromosome 22 with:
  - Three genome-wide-significant loci centred at 15.2 Mb, 28.9 Mb, 42.1 Mb
  - Realistic LD-block decay within ±200 kb of each signal peak
  - Mild genomic inflation (λ ≈ 1.05)
  - Columns: #CHROM, POS, ID, REF, ALT, P, BETA, SE, MAF
"""

import pathlib
import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2

HERE = pathlib.Path(__file__).parent

def main():
    rng = np.random.default_rng(seed=20240101)

    CHR   = 22
    START = 10_500_000
    END   = 50_800_000
    N     = 10_000

    positions = np.sort(rng.integers(START, END, N))
    ids       = [f"rs22_{i:05d}" for i in range(N)]
    alleles   = [("A", "G"), ("C", "T"), ("A", "C"), ("G", "T")]
    refs      = [alleles[i % 4][0] for i in range(N)]
    alts      = [alleles[i % 4][1] for i in range(N)]
    maf       = rng.beta(1, 8, N).clip(0.01, 0.49)

    # Null distribution with mild inflation
    chi2_null     = rng.chisquare(df=1, size=N)
    chi2_inflated = chi2_null * 1.05
    p_vals = _chi2.sf(chi2_inflated, df=1)

    # Three GWS signals
    signal_positions = [15_200_000, 28_900_000, 42_100_000]
    peak_chi2        = [35.0, 40.0, 45.0]
    half_block       = 200_000

    for sig_pos, peak in zip(signal_positions, peak_chi2):
        mask = np.abs(positions - sig_pos) < half_block
        n_in = mask.sum()
        if n_in == 0:
            continue
        dist_norm  = np.abs(positions[mask] - sig_pos) / half_block
        block_chi2 = peak * (1 - 0.85 * dist_norm) + rng.exponential(0.5, n_in)
        p_vals[mask] = _chi2.sf(block_chi2.clip(0.1), df=1)

    se   = rng.uniform(0.01, 0.05, N)
    beta = rng.normal(0, se)

    df = pd.DataFrame({
        "#CHROM": CHR,
        "POS":    positions,
        "ID":     ids,
        "REF":    refs,
        "ALT":    alts,
        "P":      p_vals,
        "BETA":   beta.round(4),
        "SE":     se.round(4),
        "MAF":    maf.round(4),
    })

    out = HERE / "chr22_synthetic_gwas.tsv.gz"
    df.to_csv(out, sep="\t", index=False, compression="gzip")

    n_sig = (df["P"] < 5e-8).sum()
    print(
        f"Wrote {len(df):,} variants | {n_sig} GWS hits (P<5e-8) "
        f"| min P={df['P'].min():.2e} | {out.stat().st_size / 1024:.1f} KB"
    )
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
