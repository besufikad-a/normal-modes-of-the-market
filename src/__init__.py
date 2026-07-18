"""Normal Modes of the Market — statistical arbitrage via covariance eigenstructure."""

from .modes import NormalModes, decompose, marchenko_pastur_bounds, clean_covariance
from .strategy import mode_signals, build_positions
from .backtest import BacktestResult, walk_forward_backtest, compute_metrics
from .data import get_sp500_tickers, download_prices, compute_returns

__all__ = [
    "NormalModes",
    "decompose",
    "marchenko_pastur_bounds",
    "clean_covariance",
    "mode_signals",
    "build_positions",
    "BacktestResult",
    "walk_forward_backtest",
    "compute_metrics",
    "get_sp500_tickers",
    "download_prices",
    "compute_returns",
]

__version__ = "0.1.0"
