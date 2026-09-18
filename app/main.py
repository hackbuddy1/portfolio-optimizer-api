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


@app.post("/optimize", response_model=OptimizationResponse)
def optimize(
    request: OptimizationRequest,
    md: MarketData = Depends(get_market_data),
):
    tickers = [s.ticker for s in request.securities]

    try:
        md.validate_tickers(tickers)
    except UnknownTickerError as exc:
        # 400, not 500: the caller sent a bad ticker, the server is fine.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.strategy is Strategy.EQUAL_WEIGHTS:
        n = len(tickers)
        optimized = [100.0 / n] * n
    else:
        raise HTTPException(
            status_code=501,
            detail=f"Strategy '{request.strategy.value}' not implemented yet.",
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
    )