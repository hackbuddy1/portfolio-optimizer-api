import numpy as np

from app.loader import TRADING_DAYS_PER_YEAR


def portfolio_returns(returns, weights):
    """Daily return series of the weighted portfolio.

    returns : (T x N) DataFrame of daily returns
    weights : (N,) array summing to 1
    """
    return np.asarray(returns) @ np.asarray(weights)


def annualized_volatility(weights, cov):
    """Portfolio standard deviation, annualised.

    sigma_p = sqrt(w' * Sigma * w)

    where,  w     = weight vector
            Sigma = annualised covariance matrix of returns
    """
    w = np.asarray(weights)
    return float(np.sqrt(w @ cov @ w))


def annualized_return(weights, mean_returns):
    """Weighted mean return, annualised.

    where,  mean_returns = daily mean return per asset x 252
    """
    return float(np.asarray(weights) @ np.asarray(mean_returns))


def sharpe_ratio(weights, mean_returns, cov, risk_free_rate=0.0):
    """Return earned per unit of risk taken.

                 R_p - R_f
    Sharpe  =  -------------
                  sigma_p

    where,  R_p     = annualised portfolio return
            R_f     = risk-free rate (0% per the brief)
            sigma_p = annualised portfolio volatility
    """
    vol = annualized_volatility(weights, cov)
    if vol == 0:
        return 0.0
    return (annualized_return(weights, mean_returns) - risk_free_rate) / vol


def max_drawdown(daily_returns):
    """Worst peak-to-trough fall of the equity curve. Returns a negative number.

    Build the cumulative growth curve, track the running maximum, and take
    the deepest proportional fall below it.
    """
    r = np.asarray(daily_returns)
    equity = np.cumprod(1.0 + r)
    running_peak = np.maximum.accumulate(equity)
    return float(np.min(equity / running_peak - 1.0))


def cagr(daily_returns):
    """Compound annual growth rate.

                  ( 1 / years )
    CAGR = ( V_f )              - 1

    where,  V_f   = final value of 1 unit invested
            years = number of trading days / 252
    """
    r = np.asarray(daily_returns)
    final_value = float(np.prod(1.0 + r))
    years = len(r) / TRADING_DAYS_PER_YEAR
    if years <= 0 or final_value <= 0:
        return 0.0
    return final_value ** (1.0 / years) - 1.0


def portfolio_dividend_yield(weights, yields):
    """Weighted average dividend yield, as a decimal."""
    return float(np.asarray(weights) @ np.asarray(yields))