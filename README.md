# Normal Modes of the Market

Code accompanying the working paper *Normal Modes of the Market*, which
connects the coupled-oscillator normal-mode decomposition from classical
mechanics to statistical arbitrage in equity markets.

## Thesis

A system of `N` coupled harmonic oscillators with symmetric coupling
matrix `K` is diagonalized by finding its eigenvectors — the **normal
modes** — each of which oscillates independently at a fixed frequency.
Modes with the smallest restoring frequencies are the slowest to
equilibrate and dominate long-time dynamics; modes with the largest are
the fastest to relax back to equilibrium.

The analogue in equity markets is the return covariance matrix `Σ`. Its
eigenvectors are portfolios whose returns are (in-sample) uncorrelated,
and its eigenvalues are the variances carried by each portfolio. The
top eigenvector is empirically close to the equal-weighted market
portfolio; intermediate eigenvectors correspond to sector-like
groupings; the smallest non-noise eigenvectors — those above the
Marchenko–Pastur bound but far below the market mode — are near-
stationary linear combinations of assets that mean-revert.

The paper argues that these low-eigenvalue signal modes are the natural
building blocks of a statistical arbitrage portfolio, and that the
Marchenko–Pastur bound provides a principled way to separate them from
noise.

## What this repo does

1. Downloads S&P 500 daily prices from Yahoo Finance.
2. Decomposes the return correlation matrix into normal modes.
3. Identifies signal modes using the Marchenko–Pastur upper bound.
4. Interprets the top mode empirically (market factor).
5. Runs a walk-forward backtest of a mode-reversion stat arb strategy.
6. Reports Sharpe, drawdown, turnover, and per-mode contributions.

## Install

```bash
git clone https://github.com/<your-username>/normal-modes-of-the-market.git
cd normal-modes-of-the-market
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Run

```bash
python scripts/reproduce.py
```

The script downloads data (cached locally after the first run),
decomposes modes, runs the backtest, and prints a summary. Outputs
(figures, PnL series) are saved to `output/`.

For an interactive walkthrough, see `notebooks/reproduce.ipynb` (also
buildable from `scripts/reproduce.py`).

## Repo structure

```
normal-modes-of-the-market/
├── paper/                  # Working paper PDF
├── src/
│   ├── data.py             # S&P 500 loader (yfinance)
│   ├── modes.py            # Covariance decomposition + Marchenko–Pastur
│   ├── strategy.py         # Mode-reversion signal + portfolio construction
│   └── backtest.py         # Walk-forward backtest + metrics
├── scripts/
│   └── reproduce.py        # End-to-end reproduction entry point
├── tests/
│   └── test_modes.py       # Sanity tests for the decomposition
└── notebooks/
    └── reproduce.ipynb     # Interactive version of reproduce.py
```

## Results (indicative)

The backtest is intentionally simple — no factor hedging beyond
excluding the top mode, no shrinkage beyond MP cleaning, no cross-
sectional volatility normalization at the asset level. Reported
numbers are indicative of the framework's raw signal and should not be
interpreted as a production strategy.

Typical output on 2015–2024 S&P 500 data:

| Metric               | Value       |
|----------------------|-------------|
| Gross Sharpe         | ~0.8–1.4    |
| Net Sharpe (1bps)    | ~0.3–0.8    |
| Avg daily turnover   | ~30–50%     |
| Max drawdown         | ~-8 to -15% |

Exact numbers vary with universe, sample period, and hyperparameters.
See `scripts/reproduce.py` for the specific configuration.

## Caveats

- Universe is the *current* S&P 500 constituents, which introduces
  survivorship bias. Correcting this requires point-in-time membership
  data (e.g., CRSP), which is not free.
- Transaction cost model is a flat basis-point charge on turnover;
  realistic costs depend on market impact, borrow, and financing.
- Covariance is estimated on a rolling window with monthly re-
  estimation; more sophisticated regime-aware estimators would likely
  improve stability.

These are all fixable with better data and more careful modeling. The
purpose of this repo is to demonstrate the framework, not to ship a
production strategy.

## Reference

Working paper: `paper/normal-modes-of-the-market.pdf`

## License

MIT — see `LICENSE`.
