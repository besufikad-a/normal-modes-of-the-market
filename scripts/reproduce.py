"""End-to-end reproduction of the Normal Modes of the Market results.

Usage
-----
From the repo root:

    python scripts/reproduce.py

Downloads S&P 500 data (cached), decomposes the return covariance into
normal modes, and runs a walk-forward backtest of the mode-reversion
strategy. Prints a summary and writes figures to ``output/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python scripts/reproduce.py`` from repo root without install.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np

from src.backtest import walk_forward_backtest
from src.data import compute_returns, download_prices, get_sp500_tickers
from src.modes import decompose


# --------------------------------------------------------------------- config

CACHE_PATH = REPO_ROOT / "data" / "cache" / "sp500_prices.parquet"
OUTPUT_DIR = REPO_ROOT / "output"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

BACKTEST_CONFIG = dict(
    estimation_window=252,
    reestimation_freq=21,
    signal_lookback=20,
    market_mode_skip=1,
    max_modes=20,
    transaction_cost_bps=1.0,
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- data
    print("=" * 60)
    print("STEP 1: Load S&P 500 data")
    print("=" * 60)
    tickers = get_sp500_tickers()
    print(f"[data] {len(tickers)} tickers from Wikipedia")

    prices = download_prices(
        tickers, start=START_DATE, end=END_DATE, cache_path=CACHE_PATH
    )
    returns = compute_returns(prices, method="log")
    print(f"[data] returns shape: {returns.shape}")

    # ---------------------------------------------------- illustrative modes
    print()
    print("=" * 60)
    print("STEP 2: Decompose full-sample returns into normal modes")
    print("=" * 60)
    modes = decompose(returns, standardize=True)
    print(f"  T = {returns.shape[0]}, N = {returns.shape[1]}")
    print(f"  q = N/T = {modes.q:.4f}")
    print(f"  Marchenko-Pastur upper bound: lambda_+ = {modes.mp_upper:.4f}")
    print(f"  Signal modes (lambda > lambda_+): {modes.n_signal_modes}")
    print()
    print("  Top-10 eigenvalues:")
    for i, lam in enumerate(modes.eigenvalues[:10]):
        marker = "  <-- market factor" if i == 0 else ""
        print(f"    lambda_{i:<3d} = {lam:>10.4f}{marker}")

    # Sanity check: the top eigenvector should correlate strongly with a
    # simple equal-weight portfolio (that's the market factor).
    top_mode = modes.top_mode_weights(0)
    equal_weight = np.ones(len(top_mode)) / np.sqrt(len(top_mode))
    ew_alignment = float(np.abs(top_mode.values @ equal_weight))
    print()
    print(f"  |<mode_0, equal_weight_normalized>| = {ew_alignment:.4f}")
    print("  (Values near 1 indicate mode_0 is approximately the market portfolio.)")

    # ---------------------------------------- eigenvalue spectrum figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(modes.eigenvalues, "o-", markersize=3, linewidth=0.8)
    ax.axhline(
        modes.mp_upper,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"MP upper = {modes.mp_upper:.3f}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Mode index (descending eigenvalue)")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Return correlation eigenvalue spectrum vs. Marchenko-Pastur bound")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "eigenvalue_spectrum.png", dpi=150)
    plt.close(fig)
    print(f"  [figure] {OUTPUT_DIR / 'eigenvalue_spectrum.png'}")

    # ------------------------------------------------------------ backtest
    print()
    print("=" * 60)
    print("STEP 3: Walk-forward backtest of mode-reversion strategy")
    print("=" * 60)
    print("  Config:")
    for k, v in BACKTEST_CONFIG.items():
        print(f"    {k:22s} = {v}")
    print()

    result = walk_forward_backtest(returns, verbose=True, **BACKTEST_CONFIG)

    print()
    print(result.summary())

    # ------------------------------------------------- cumulative PnL fig
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    cum_gross = result.pnl_gross.cumsum()
    cum_net = result.pnl.cumsum()
    ax1.plot(cum_gross, label="Gross", linewidth=1.2)
    ax1.plot(cum_net, label=f"Net ({BACKTEST_CONFIG['transaction_cost_bps']}bps)", linewidth=1.2)
    ax1.set_ylabel("Cumulative log-return")
    ax1.set_title("Mode-reversion strategy: cumulative PnL")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown of net PnL.
    running_max = cum_net.cummax()
    drawdown = cum_net - running_max
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pnl_and_drawdown.png", dpi=150)
    plt.close(fig)
    print(f"  [figure] {OUTPUT_DIR / 'pnl_and_drawdown.png'}")

    # Save PnL series for downstream analysis.
    result.pnl.to_csv(OUTPUT_DIR / "pnl_net.csv")
    result.pnl_gross.to_csv(OUTPUT_DIR / "pnl_gross.csv")
    print(f"  [data]   {OUTPUT_DIR / 'pnl_net.csv'}")
    print(f"  [data]   {OUTPUT_DIR / 'pnl_gross.csv'}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
