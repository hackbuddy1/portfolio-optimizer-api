import numpy as np

FACTOR_NAMES = ("momentum", "value", "size")


def _design_matrix(factors):
    """Factor returns with a leading column of ones for the intercept (alpha).

    Column order is [alpha, momentum, value, size].
    """
    return np.column_stack([
        np.ones(len(factors)),
        factors["momentum"].values,
        factors["value"].values,
        factors["size"].values,
    ])


def beta_matrix(returns, factors):
    """Precompute M such that betas(w) == M @ w.

    Portfolio Return = alpha + b1(Momentum) + b2(Value) + b3(Size) + error

    OLS gives coefficients (X'X)^-1 X' y. The portfolio return series is
    y = R @ w, so substituting:

        coefficients(w) = (X'X)^-1 X' R @ w

    The bracketed part does not depend on w, so it is computed once here.
    Row 0 is alpha; rows 1-3 are the momentum, value and size betas.
    """
    X = _design_matrix(factors)
    R = np.asarray(returns)
    return np.linalg.solve(X.T @ X, X.T @ R)


def factor_betas(M, weights):
    """Momentum, value and size betas for a given allocation."""
    coefficients = M @ np.asarray(weights)
    return {
        "momentum": round(float(coefficients[1]), 4),
        "value": round(float(coefficients[2]), 4),
        "size": round(float(coefficients[3]), 4),
    }


def optimize_factor_exposure(M, factor, direction="maximize",
                             bounds=None, constraints=None):
    """Push exposure to one factor as high (or low) as the limits allow.

    Because betas are linear in the weights, the objective is linear and
    the solver cannot get stuck in a local optimum.
    """
    from app.optimizer import solve

    if factor not in FACTOR_NAMES:
        raise ValueError(
            f"Unknown factor '{factor}'. Choose from {', '.join(FACTOR_NAMES)}."
        )

    row = M[1 + FACTOR_NAMES.index(factor)]
    sign = 1.0 if direction == "maximize" else -1.0

    return solve(
        lambda w: -sign * float(row @ w),
        n_assets=M.shape[1],
        bounds=bounds,
        constraints=constraints,
    )