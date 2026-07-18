"""Data loading utilities for S&P 500 equities.

The universe used here is the *current* S&P 500, which introduces
survivorship bias. For a production study, replace with point-in-time
constituent data (e.g. from CRSP).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituents from Wikipedia.

    Yahoo Finance uses ``-`` instead of ``.`` for tickers like BRK.B
    (``BRK-B``) and BF.B (``BF-B``), so we substitute accordingly.
    """
    tables = pd.read_html(SP500_WIKI_URL)
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    return sorted(set(tickers))


def download_prices(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
    cache_path: Path | str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols (Yahoo format).
    start, end : str or None
        Date range (YYYY-MM-DD). ``end=None`` means through today.
    cache_path : Path or str or None
        If given, cache the downloaded frame as parquet at this path.
        Subsequent calls with the same path reuse the cache.
    verbose : bool
        Print a summary after loading.

    Returns
    -------
    pd.DataFrame
        Rows are trading days, columns are tickers. Adjusted close prices.
        Tickers with more than 10% missing data in the requested window are
        dropped; remaining gaps are forward-filled.
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path is not None and cache_path.exists():
        prices = pd.read_parquet(cache_path)
        if verbose:
            print(f"[data] loaded cache: {prices.shape[0]} days × {prices.shape[1]} tickers")
        return prices

    # Lazy import so tests without yfinance still run
    import yfinance as yf

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if isinstance(data.columns, pd.MultiIndex):
        # yfinance returns a MultiIndex when multiple tickers are requested
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs("Close", level=1, axis=1).copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers[:1]

    # Drop tickers with too much missing data
    threshold = int(0.9 * len(prices))
    prices = prices.dropna(axis=1, thresh=threshold)
    prices = prices.ffill().dropna()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)

    if verbose:
        print(f"[data] downloaded: {prices.shape[0]} days × {prices.shape[1]} tickers")
    return prices


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute daily returns from a price series.

    Parameters
    ----------
    prices : pd.DataFrame
        Rows are dates, columns are assets.
    method : {"log", "simple"}
        Log returns are approximately additive across days and are the
        default for covariance / eigen-decomposition work. Simple
        returns are appropriate when reporting PnL.
    """
    if method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    if method == "simple":
        return prices.pct_change().dropna()
    raise ValueError(f"Unknown method: {method!r}. Use 'log' or 'simple'.")
