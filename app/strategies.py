import numpy as np

from app.metrics import (
    annualized_volatility,
    max_drawdown,
    portfolio_returns,
    sharpe_ratio,
)
from app.optimizer import solve


def equal_weights(n_assets):
    """Baseline. No optimisation needed -- 1/N each."""
    return np.full(n_assets, 1.0 / n_assets)


def minimize_volatility(cov, bounds=None, constraints=None):
    """Lowest-risk portfolio.

    minimise   sigma_p = sqrt(w' * Sigma * w)

    where,  w     = weight vector
            Sigma = annualised covariance matrix
    """
    return solve(
        lambda w: annualized_volatility(w, cov),
        n_assets=cov.shape[0],
        bounds=bounds,
        constraints=constraints,
    )


def maximize_sharpe(mean_returns, cov, risk_free_rate=0.0,
                    bounds=None, constraints=None):
    """Best risk-adjusted return.

    scipy only minimises, so we minimise the negative Sharpe ratio.

                     R_p - R_f
    maximise      -------------
                      sigma_p

    where,  R_p     = annualised portfolio return
            R_f     = risk-free rate
            sigma_p = annualised portfolio volatility
    """
    return solve(
        lambda w: -sharpe_ratio(w, mean_returns, cov, risk_free_rate),
        n_assets=cov.shape[0],
        bounds=bounds,
        constraints=constraints,
    )


def risk_parity(cov, bounds=None, constraints=None):
    """Every security contributes the same share of total portfolio risk.

    Risk contribution of asset i:

                 w_i * (Sigma * w)_i
        RC_i =  ---------------------
                       sigma_p

    where,  w_i     = weight of asset i
            Sigma   = annualised covariance matrix
            sigma_p = annualised portfolio volatility

    Perfect parity means every RC_i is equal, so we minimise the sum of
    squared differences between every pair of contributions -- which is
    zero only when they all match.
    """
    def objective(w):
        vol = annualized_volatility(w, cov)
        if vol == 0:
            return 1e6
        marginal = cov @ w              # d(sigma_p) / d(w_i), unscaled
        contributions = w * marginal / vol
        diffs = contributions[:, None] - contributions[None, :]
        return float(np.sum(diffs ** 2))

    return solve(
        objective,
        n_assets=cov.shape[0],
        bounds=bounds,
        constraints=constraints,
    )


def minimize_drawdown(returns, bounds=None, constraints=None):
    """Shallowest worst-case peak-to-trough loss.

    max_drawdown() returns a negative number, and a shallower drawdown is
    a larger (less negative) value -- so minimising its negation is what
    makes the loss smaller.
    """
    matrix = np.asarray(returns)

    def objective(w):
        return -max_drawdown(portfolio_returns(matrix, w))

    return solve(
        objective,
        n_assets=matrix.shape[1],
        bounds=bounds,
        constraints=constraints,
    )