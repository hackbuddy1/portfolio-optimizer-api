import numpy as np
from scipy.optimize import minimize


class OptimizationError(RuntimeError):
    """Raised when the solver cannot find a feasible allocation."""


def solve(objective, n_assets, bounds=None, constraints=None,
          n_restarts=25, seed=42):
    """Minimise `objective` over portfolio weights.

    Always enforced:
      - weights sum to 1
      - no weight is negative (no short selling)

    objective   : callable taking a weight array, returning a float to minimise
    bounds      : list of (min, max) per asset, defaults to (0, 1)
    constraints : extra scipy constraint dicts (portfolio-level limits)
    n_restarts  : random starting points, on top of the equal-weight start
    """
    if bounds is None:
        bounds = [(0.0, 1.0)] * n_assets

    all_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if constraints:
        all_constraints.extend(constraints)

    # Fixed seed so the same request always returns the same weights.
    rng = np.random.default_rng(seed)
    starts = [np.full(n_assets, 1.0 / n_assets)]
    starts += [rng.dirichlet(np.ones(n_assets)) for _ in range(n_restarts)]

    best = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=all_constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success and (best is None or result.fun < best.fun):
            best = result

    if best is None:
        raise OptimizationError(
            "No feasible allocation found. The constraints may be too tight."
        )

    # Clean up floating-point dust, then renormalise so weights sum to
    # exactly 1 before we convert to percentages.
    weights = np.clip(best.x, 0.0, 1.0)
    return weights / weights.sum()