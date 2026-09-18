import numpy as np
from app import factors
from app.schemas import FactorBetaComparison, FactorBetas
from app.loader import TRADING_DAYS_PER_YEAR
from app.optimizer import OptimizationError
from app import strategies
from app.constraints import (
    InfeasibleConstraintsError,
    build_bounds,
    build_constraints,
    check_feasibility,
)

from fastapi import Depends, FastAPI, HTTPException

from app.loader import MarketData, UnknownTickerError, get_market_data
from app.schemas import (
    AllocationChange,
    OptimizationRequest,
    OptimizationResponse,
    Strategy,
)

app = FastAPI(
    title="Portfolio Optimizer API",
    description="Portfolio optimization across several allocation strategies.",
    version="1.0.0",
)


@app.get("/health")
def health(md: MarketData = Depends(get_market_data)):
    """Cheap liveness check that also proves the workbook loaded."""
    return {"status": "ok", "tickers": md.available_tickers}

def run_strategy(strategy, returns, constraints, yields, factor_objective=None, beta_matrix=None):
    """Dispatch to the right optimiser and return weights as fractions.

    Covariance and mean returns are annualised once here, so every
    strategy works in the same units.
    """
    cov = returns.cov().values * TRADING_DAYS_PER_YEAR
    mean_returns = returns.mean().values * TRADING_DAYS_PER_YEAR
    n_assets = returns.shape[1]

    # Reject impossible requests before burning time in the solver.
    check_feasibility(constraints, yields, n_assets)

    bounds = build_bounds(constraints, n_assets)
    extra = build_constraints(constraints, returns, cov, yields)

    if strategy is Strategy.EQUAL_WEIGHTS:
        return strategies.equal_weights(n_assets)
    if strategy is Strategy.MINIMIZE_VOLATILITY:
        return strategies.minimize_volatility(cov, bounds, extra)
    if strategy is Strategy.MAXIMIZE_SHARPE:
        return strategies.maximize_sharpe(mean_returns, cov, 0.0, bounds, extra)
    if strategy is Strategy.RISK_PARITY:
        return strategies.risk_parity(cov, bounds, extra)
    if strategy is Strategy.MINIMIZE_DRAWDOWN:
        return strategies.minimize_drawdown(returns, bounds, extra)
    if strategy is Strategy.OPTIMIZE_FACTOR_EXPOSURE:
        return factors.optimize_factor_exposure(
            beta_matrix,
            factor_objective.factor.value,
            factor_objective.direction.value,
            bounds,
            extra,
        )

    raise HTTPException(
        status_code=501,
        detail=f"Strategy '{strategy.value}' not implemented yet.",
    )

@app.post("/optimize", response_model=OptimizationResponse)
def optimize(
    request: OptimizationRequest,
    md: MarketData = Depends(get_market_data),
):
    tickers = [s.ticker for s in request.securities]

    try:
        returns = md.returns_for(tickers)
        yields = [md.dividend_yield(t) for t in tickers]

        # Factor betas need the dates the funds and factors share, which is
        # a shorter window than the funds alone. Only built when needed.
        beta_matrix = None
        if request.strategy is Strategy.OPTIMIZE_FACTOR_EXPOSURE:
            fund_returns, factor_returns = md.aligned_with_factors(tickers)
            beta_matrix = factors.beta_matrix(fund_returns, factor_returns)

        weights = run_strategy(
            request.strategy,
            returns,
            request.constraints,
            yields,
            request.factor_objective,
            beta_matrix,
        )
    except UnknownTickerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except InfeasibleConstraintsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OptimizationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    optimized = weights * 100.0

    # Betas for the caller's original allocation and the optimised one,
    # so the two can be compared directly.
    factor_comparison = None
    if beta_matrix is not None:
        current = np.array([s.weight for s in request.securities]) / 100.0
        factor_comparison = FactorBetaComparison(
            current_portfolio=FactorBetas(
                **factors.factor_betas(beta_matrix, current)
            ),
            optimized_portfolio=FactorBetas(
                **factors.factor_betas(beta_matrix, weights)
            ),
        )

    changes = [
        AllocationChange(
            ticker=security.ticker,
            security_name=md.security_name(security.ticker),
            current_weight=round(security.weight, 2),
            optimized_weight=round(weight, 2),
            change=round(weight - security.weight, 2),
        )
        for security, weight in zip(request.securities, optimized)
    ]

    return OptimizationResponse(
        optimization_strategy=request.strategy.value,
        allocation_changes=changes,
        factor_betas=factor_comparison,
    )