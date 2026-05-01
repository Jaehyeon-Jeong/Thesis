"""Stage 2 BTC time-series trading metrics.

역할:
    BTC는 단일 자산이라 원논문 stock cross-sectional H-L decile portfolio를 만들 수
    없다. 이 module은 prediction을 daily BTC return에 맞춰 long/flat,
    long/short strategy metric으로 변환한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_trading_metrics(
    predictions: pd.DataFrame,
    annualization_periods: int = 365,
    transaction_cost_bps: float = 10.0,
) -> dict[str, Any]:
    """prediction row에서 long/flat과 long/short strategy metric을 계산한다.

    단순화:
        Stage 2 baseline에서는 signal date의 다음 label end close까지의
        `future_return`을 sample-level trade return으로 사용한다. full daily
        overlapping equity curve는 이후 robustness 항목에서 더 정교화할 수 있다.
    """

    frame = predictions.copy().sort_values("Date")
    future = frame["future_return"].astype(float).to_numpy()
    pred = frame["pred_class"].astype(int).to_numpy()
    prob = frame["prob_up"].astype(float).to_numpy()

    long_flat_position = pred.astype(float)
    long_short_position = np.where(pred == 1, 1.0, -1.0)
    results = {
        "long_flat": _strategy_metrics(
            future,
            long_flat_position,
            prob,
            annualization_periods,
            transaction_cost_bps,
        ),
        "long_short": _strategy_metrics(
            future,
            long_short_position,
            prob,
            annualization_periods,
            transaction_cost_bps,
        ),
        "note": (
            "BTC is a single asset; this replaces stock H-L decile evaluation "
            "with time-series strategy metrics."
        ),
    }
    return results


def _strategy_metrics(
    future_return: np.ndarray,
    position: np.ndarray,
    probability: np.ndarray,
    annualization_periods: int,
    transaction_cost_bps: float,
) -> dict[str, float]:
    """position과 future return으로 strategy summary를 계산한다."""

    gross = position * future_return
    turnover = np.abs(np.diff(position, prepend=0.0))
    cost = turnover * float(transaction_cost_bps) / 10_000.0
    net = gross - cost
    return {
        "mean_trade_return_gross": float(np.mean(gross)),
        "mean_trade_return_net": float(np.mean(net)),
        "annualized_return_gross": float(np.mean(gross) * annualization_periods),
        "annualized_return_net": float(np.mean(net) * annualization_periods),
        "annualized_volatility": float(np.std(gross, ddof=1) * np.sqrt(annualization_periods)),
        "sharpe_gross": _sharpe(gross, annualization_periods),
        "sharpe_net": _sharpe(net, annualization_periods),
        "max_drawdown_gross": _max_drawdown(gross),
        "max_drawdown_net": _max_drawdown(net),
        "turnover_mean": float(np.mean(turnover)),
        "transaction_cost_bps": float(transaction_cost_bps),
        "mean_prob_up": float(np.mean(probability)),
    }


def _sharpe(returns: np.ndarray, annualization_periods: int) -> float:
    """연율화 Sharpe ratio를 계산한다."""

    std = float(np.std(returns, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(annualization_periods))


def _max_drawdown(returns: np.ndarray) -> float:
    """simple compounded equity curve 기준 max drawdown을 계산한다."""

    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return float(np.min(drawdown))
