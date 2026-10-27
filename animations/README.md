# Animations — *Normal Modes of the Market* in motion

Manim (Community Edition) scenes that build the paper's idea visually:
fifty years of real asset relationships, then coupled oscillators for
N = 2, 3, 5, 8, 16 (each mode separately, then combined), then the
real market's modes and eigenvalue spectrum.

**Full film:** [`normal_modes_of_the_market.mp4`](normal_modes_of_the_market.mp4) (1080p30).
Individual scenes are in [`videos/`](videos/).

| # | Scene | What it shows | 3-D |
|---|---|---|---|
| 0 | `S0_Title` | title card with two coupled masses beating | |
| 1 | `S1_History` | growth of $1 in stocks, bonds and gold 1975–2026 (log value vs. time) with the 36-month rolling correlations drawn live beneath, key events marked | |
| 2 | `S2_Relationship3D` | every month as a point (stock return × bond return × time); each decade's principal axis rotates as the correlation flips sign | ✓ |
| 3 | `S3_CorrelationRibbons` | time × asset pair × correlation as three ribbons (stocks–bonds, stocks–gold, bonds–gold) | ✓ |
| 4 | `S4_TwoOscillators` | N = 2: stocks and bonds as coupled masses; in-phase mode, out-of-phase mode, then their combination (beats); for two assets the modes are always (1,1) and (1,−1) with λ = 1 ± ρ, animated over 1978–2026 | |
| 5 | `S5_ChainModes` | N = 3, 5, 8 beads on a string: every mode on its own, then all modes combined with their modal amplitudes | |
| 6 | `S6_ModeSurface3D` | N = 16: bead × mode × displacement, all modes oscillating at once, plus a chain built as their sum | ✓ |
| 7 | `S7_MarketModes` | 12 stocks: correlation matrix, eigenvalues vs. the Marchenko–Pastur edge, modes 0–3 and 11 shown separately, then 16 March 2020 rebuilt mode by mode | |
| 8 | `S8_Spectrum3D` | rolling eigenvalue spectrum of 417 stocks, 2011–2024 (time × rank × log λ) against the MP noise floor | ✓ |
| 9 | `S9_Findings` | Σ = (s²/2)K⁻¹: small λ = stiff = fast reversion, and the paper's out-of-sample result by spectrum bucket | |

## Data

`data_prep.py` writes the small files in `data/`:

* **Stocks:** S&P 500 index (`^GSPC`, price only — excludes dividends), Yahoo Finance.
* **Bonds:** 10-year Treasury total return reconstructed from the 10-year yield (`^TNX`):
  each month a par bond is bought at last month's yield and repriced at this month's.
* **Gold:** monthly prices from the datahub.io `gold-prices` dataset.
* **Market modes / spectrum:** the paper's 2010–2024 S&P 500 price cache
  (run `scripts/run_study.py` first); findings come from `output/results.json`.

## Rebuild

```bash
uv venv --python 3.13 .venv-anim
uv pip install --python .venv-anim/bin/python manim pandas numpy scipy yfinance pyarrow
cd animations
../.venv-anim/bin/python data_prep.py            # needs data/cache from the study
bash render_all.sh ../.venv-anim/bin/manim      # ~6 min on a laptop; writes the film
```

Single scene preview: `../.venv-anim/bin/manim -pql s2_oscillators.py S5_ChainModes`.
Requires a LaTeX install for the equations.
