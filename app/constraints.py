import numpy as np

from app.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    portfolio_returns,
)


class InfeasibleConstraintsError(ValueError):
    """Raised when no allocation can possibly satisfy the request."""


def build_bounds(constraints, n_assets):
    """Per-security weight limits, converted from percent to fractions."""
    low = 0.0
    high = 1.0
    if constraints is not None:
        if constraints.min_weight is not None:
            low = constraints.min_weight / 100.0
        if constraints.max_weight is not None:
            high = constraints.max_weight / 100.0
    return [(low, high)] * n_assets


def max_achievable_yield(yields, low, high):
    """Highest portfolio dividend yield the weight limits allow.

    Give every security its minimum, then pour the remaining weight into
    the highest-yielding securities until each hits its cap. Nothing can
    beat that, so if it still falls short of the target, the request is
    impossible and no amount of solving will help.
    """
    n = len(yields)
    weights = np.full(n, low)
    remaining = 1.0 - low * n
    for i in np.argsort(-np.asarray(yields)):
        added = min(high - low, remaining)
        weights[i] += added
        remaining -= added
        if remaining <= 1e-12:
            break
    return float(weights @ np.asarray(yields))


def check_feasibility(constraints, yields, n_assets):
    """Reject impossible requests up front, with a message naming the cause."""
    if constraints is None:
        return

    low, high = build_bounds(constraints, n_assets)[0]

    if low > high:
        raise InfeasibleConstraintsError(
            f"min_weight ({low * 100:.2f}%) exceeds "
            f"max_weight ({high * 100:.2f}%)."
        )

    if low * n_assets > 1.0 + 1e-9:
        raise InfeasibleConstraintsError(
            f"min_weight of {low * 100:.2f}% across {n_assets} securities "
            f"requires {low * n_assets * 100:.2f}%, which exceeds 100%."
        )

    if high * n_assets < 1.0 - 1e-9:
        raise InfeasibleConstraintsError(
            f"max_weight of {high * 100:.2f}% across {n_assets} securities "
            f"allows only {high * n_assets * 100:.2f}%, which cannot reach 100%."
        )

    if constraints.min_dividend_yield is not None:
        target = constraints.min_dividend_yield / 100.0
        best = max_achievable_yield(yields, low, high)
        if best < target - 1e-9:
            raise InfeasibleConstraintsError(
                f"Min dividend yield of {target * 100:.2f}% is unreachable. "
                f"The highest achievable within the weight limits is "
                f"{best * 100:.2f}%."
            )

    if (constraints.min_volatility is not None
            and constraints.max_volatility is not None
            and constraints.min_volatility > constraints.max_volatility):
        raise InfeasibleConstraintsError(
            "min_volatility exceeds max_volatility."
        )


def build_constraints(constraints, returns, cov, yields):
    """Portfolio-level limits as scipy inequality constraints.

    scipy treats an 'ineq' constraint as satisfied when fun(w) >= 0, so
    every rule below is rearranged into that form.
    """
    if constraints is None:
        return []

    matrix = np.asarray(returns)
    yields = np.asarray(yields)
    built = []

    if constraints.min_dividend_yield is not None:
        target = constraints.min_dividend_yield / 100.0
        built.append({
            "type": "ineq",
            "fun": lambda w, t=target: float(w @ yields) - t,
        })

    if constraints.min_cagr is not None:
        target = constraints.min_cagr / 100.0
        built.append({
            "type": "ineq",
            "fun": lambda w, t=target: cagr(portfolio_returns(matrix, w)) - t,
        })

    if constraints.max_drawdown is not None:
        # max_drawdown() is negative; the request gives a positive limit.
        # Allowing a 20% drawdown means dd >= -0.20, i.e. dd + 0.20 >= 0.
        limit = constraints.max_drawdown / 100.0
        built.append({
            "type": "ineq",
            "fun": lambda w, l=limit: max_drawdown(
                portfolio_returns(matrix, w)
            ) + l,
        })

    if constraints.min_volatility is not None:
        target = constraints.min_volatility / 100.0
        built.append({
            "type": "ineq",
            "fun": lambda w, t=target: annualized_volatility(w, cov) - t,
        })

    if constraints.max_volatility is not None:
        limit = constraints.max_volatility / 100.0
        built.append({
            "type": "ineq",
            "fun": lambda w, l=limit: l - annualized_volatility(w, cov),
        })

    return built