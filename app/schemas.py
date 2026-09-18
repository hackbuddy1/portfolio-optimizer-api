from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class Strategy(str, Enum):
    EQUAL_WEIGHTS = "equal_weights"
    RISK_PARITY = "risk_parity"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MINIMIZE_VOLATILITY = "minimize_volatility"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    OPTIMIZE_FACTOR_EXPOSURE = "optimize_factor_exposure"


class SecurityInput(BaseModel):
    ticker: str
    weight: float = Field(ge=0, le=100, description="Current weight in percent")

    @field_validator("ticker")
    @classmethod
    def uppercase(cls, v):
        return v.strip().upper()


class Constraints(BaseModel):
    """All optional. Percentages, not decimals."""
    min_weight: float | None = Field(default=None, ge=0, le=100)
    max_weight: float | None = Field(default=None, ge=0, le=100)
    min_dividend_yield: float | None = Field(default=None, ge=0)
    min_cagr: float | None = None
    min_volatility: float | None = Field(default=None, ge=0)
    max_volatility: float | None = Field(default=None, ge=0)
    max_drawdown: float | None = Field(default=None, ge=0)

class FactorName(str, Enum):
    MOMENTUM = "momentum"
    VALUE = "value"
    SIZE = "size"


class Direction(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class FactorObjective(BaseModel):
    factor: FactorName
    direction: Direction = Direction.MAXIMIZE


class FactorBetas(BaseModel):
    momentum: float
    value: float
    size: float


class FactorBetaComparison(BaseModel):
    current_portfolio: FactorBetas
    optimized_portfolio: FactorBetas

class OptimizationRequest(BaseModel):
    securities: list[SecurityInput] = Field(min_length=1)
    strategy: Strategy
    constraints: Constraints | None = None
    factor_objective: FactorObjective | None = None
    risk_free_rate: float = Field(
        default=0.0, ge=0, le=20,
        description="Annual risk-free rate in percent. The brief permits 0%.",
    )

    @model_validator(mode="after")
    def weights_must_sum_to_100(self):
        total = sum(s.weight for s in self.securities)
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Current weights must sum to 100, got {total:.2f}")
        return self

    @model_validator(mode="after")
    def no_duplicate_tickers(self):
        tickers = [s.ticker for s in self.securities]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Duplicate tickers in request")
        return self
    @model_validator(mode="after")
    def factor_strategy_needs_an_objective(self):
        if (self.strategy is Strategy.OPTIMIZE_FACTOR_EXPOSURE
                and self.factor_objective is None):
            raise ValueError(
                "optimize_factor_exposure requires 'factor_objective', "
                "e.g. {\"factor\": \"momentum\", \"direction\": \"maximize\"}"
            )
        return self


class AllocationChange(BaseModel):
    ticker: str
    security_name: str
    current_weight: float
    optimized_weight: float
    change: float


class OptimizationResponse(BaseModel):
    optimization_strategy: str
    allocation_changes: list[AllocationChange]
    factor_betas: FactorBetaComparison | None = None