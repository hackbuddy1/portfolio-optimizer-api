# Portfolio Optimizer API

A REST API that computes optimized portfolio weights across six allocation
strategies, built for the Finominal Junior Backend Developer assignment.

Built with FastAPI, pandas, NumPy and SciPy.

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Requires Python 3.10 or newer (built and tested on 3.12.6).

Interactive docs at http://127.0.0.1:8000/docs — every example below can be run
from that page without Postman or any other tooling.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check; also confirms the workbook loaded |
| POST | `/optimize` | Returns optimized weights for a portfolio |

### Example request

```bash
curl -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "securities": [
      {"ticker": "SPY", "weight": 60},
      {"ticker": "AGG", "weight": 30},
      {"ticker": "GLD", "weight": 10}
    ],
    "strategy": "minimize_volatility"
  }'
```

### Example response

```json
{
  "optimization_strategy": "minimize_volatility",
  "allocation_changes": [
    {"ticker": "SPY", "security_name": "State Street SPDR S&P 500 ETF Trust",
     "current_weight": 60.0, "optimized_weight": 6.92, "change": -53.08},
    {"ticker": "AGG", "security_name": "iShares Core US Aggregate Bond ETF",
     "current_weight": 30.0, "optimized_weight": 91.21, "change": 61.21},
    {"ticker": "GLD", "security_name": "SPDR Gold Shares",
     "current_weight": 10.0, "optimized_weight": 1.86, "change": -8.14}
  ],
  "factor_betas": null
}
```

`factor_betas` is populated only for the factor exposure strategy, so the four
standard strategies return exactly the shape the brief specifies.

## Strategies

| Name | Objective |
|---|---|
| `equal_weights` | 1/N across all securities |
| `risk_parity` | Equalise each security's contribution to portfolio risk |
| `minimize_drawdown` | Shallowest worst peak-to-trough loss |
| `minimize_volatility` | Lowest annualised standard deviation |
| `maximize_sharpe` | Highest return per unit of risk |
| `optimize_factor_exposure` | Maximise or minimise momentum, value or size beta |

## Request options

```json
{
  "securities": [{"ticker": "SPY", "weight": 100}],
  "strategy": "maximize_sharpe",
  "risk_free_rate": 1.6,
  "constraints": {
    "min_weight": 5,
    "max_weight": 40,
    "min_dividend_yield": 2.5,
    "min_cagr": 4.0,
    "min_volatility": 5.0,
    "max_volatility": 12.0,
    "max_drawdown": 20.0
  },
  "factor_objective": {"factor": "momentum", "direction": "maximize"}
}
```

Everything except `securities` and `strategy` is optional. All percentages,
not decimals. `risk_free_rate` defaults to 0%, which the brief permits.

Per-security limits become solver bounds. Portfolio-level limits become
inequality constraints. Infeasible requests are rejected **before** the solver
runs, with a message naming the specific cause:

```json
{"detail": "Min dividend yield of 5.00% is unreachable. The highest achievable within the weight limits is 3.15%."}
```

## Design decisions

**The date window depends on which securities were requested.** The five funds
have different inception dates — SPY from 1993, GLD from 2004, IEFA only from
2012. Columns are selected *before* `dropna()`, so a request for SPY/AGG/GLD
uses 5,413 trading days back to 2004-11-18, while a request that includes IEFA
uses 3,414 days back to 2012-10-23. Dropping first would silently truncate
every request to the shortest available history. The reference tool picks the
same start dates, which confirms the approach.

**Chronological ordering.** The source workbook is stored newest-first. Max
drawdown walks the equity curve forward tracking a running peak, so on an
unsorted series it returns a plausible-looking but wrong number without raising
anything. The loader sorts ascending immediately after pivoting.

**Annualisation.** Volatility scales with √252, mean return with 252. Daily
returns in the workbook are already decimals.

**SLSQP with 25 random restarts.** SLSQP handles a nonlinear objective with
per-asset bounds plus the sum-to-one equality constraint. Minimize-drawdown is
not convex — shifting weights changes which peak-and-trough pair is the worst,
so the objective has kinks and multiple local minima. Starting points are drawn
from a Dirichlet distribution, which samples uniformly over the feasible
simplex, and a fixed seed keeps results reproducible across calls.

**Factor betas are linear in the weights.** The portfolio return series is
`R @ w` and OLS gives `(X'X)⁻¹X'y`, so substituting `y = R @ w` makes the betas
`M @ w` where `M = (X'X)⁻¹X'R` does not depend on the weights. `M` is computed
once per request instead of running a regression inside every solver iteration,
and the resulting objective is linear, so there are no local optima. I used
`np.linalg.solve` rather than forming an explicit matrix inverse.

**Missing dividend yield.** GLD's cell is blank in the workbook because gold
pays no dividend. Left as `NaN` it propagates through the weighted yield
calculation, making every comparison against the constraint return `False` and
producing a spurious infeasibility. It is coerced to 0 on load.

**Error handling.** `400` for bad input such as an unknown ticker, `422` for a
well-formed request whose constraints cannot be satisfied, `501` for a
recognised but unimplemented strategy.

## Validation against the reference tool

All six scenarios were run against https://finominal.com/portfolio-optimizer/US
with matching inputs and, where the tool exposes it, an end date of 27/05/2026
to match the last date in the workbook. Screenshots are in `screenshots/`.

| # | Scenario | Strategy | API output | Reference tool | Max difference |
|---|---|---|---|---|---|
| 1 | IEFA 25 / SPY 75 | Equal Weights | 50.00 / 50.00 | 50.00 / 50.00 | exact |
| 2 | VEA 25 / AGG 75 | Risk Parity | VEA 20.12, AGG 79.88 | VEA 20.11, AGG 79.89 | 0.01% |
| 3 | SPY 60 / AGG 30 / GLD 10 | Minimize Volatility | 6.92 / 91.21 / 1.86 | 6.94 / 91.11 / 1.95 | 0.10% |
| 4 | Equal-weight all five | Maximize Sharpe | GLD 30.54, SPY 69.46 (rf 1.6%) | GLD 30.60, SPY 69.40 | 0.06% |
| 5 | All five, 5–40% band, 2.5% yield floor | Maximize Sharpe | IEFA 13.76, GLD 5, AGG 40, VEA 5, SPY 36.24 | IEFA 10.64, GLD 5, AGG 40, VEA 5, SPY 39.36 | 3.12% |
| 6 | All five, 5–40% band | Maximize Momentum | VEA 40, SPY 40, IEFA 10, GLD 5, AGG 5 | not comparable | — |

Four of the five required cases fall inside the brief's 0.1% tolerance. The two
that did not match at first are explained below; both turned out to be data or
parameter differences rather than errors in the optimizer.

### Case 4 — risk-free rate

With a 0% risk-free rate my optimizer allocated 34.36% to AGG, while the tool
allocated none. Sweeping the rate showed the tool behaves as though it uses
roughly 1.6%:

| rf | IEFA | GLD | AGG | VEA | SPY |
|---|---|---|---|---|---|
| 0.0% | 0 | 20.10 | 34.36 | 0 | 45.55 |
| 1.0% | 0 | 31.75 | 0 | 0 | 68.25 |
| **1.6%** | 0 | **30.54** | **0** | 0 | **69.46** |
| Tool | 0 | 30.60 | 0 | 0 | 69.40 |

AGG returned only 1.91% annualised over this window. At a 0% rate its excess
return is 1.91% and its low volatility makes it attractive; at 1.6% the excess
return nearly vanishes and it leaves the portfolio entirely. One parameter
changes the whole allocation, so I exposed `risk_free_rate` as a request field
rather than hardcoding either value. The default stays at 0% as the brief
permits.

### Case 5 — dividend yield data

Three of the five weights match exactly — GLD at its 5% floor, AGG at its 40%
cap, VEA at its 5% floor. The remaining 3.12% sits between IEFA and SPY, and
comes from the dividend yield data rather than the optimizer.

The tool reports a higher yield than the workbook for the same portfolio (2.04%
against 1.78% for SPY 60 / AGG 30 / GLD 10), so it is using more recent yields.
Scoring the tool's own case 5 answer with the workbook's yields gives **2.43%**,
which is below the 2.50% floor — meaning the tool's optimum is not a feasible
point in this dataset at all. My output holds the yield at exactly 2.50%, so
the constraint is binding correctly.

### Case 6 — factor model

The tool optimises across five factors (Value, Momentum, Low Volatility,
Quality, Size); the workbook supplies three (Momentum, Value, Size). The betas
are therefore on different scales and the weights are not comparable — the brief
says as much. What the brief does ask for is demonstrated: momentum beta rises
from **0.1316 to 0.1676** while every weight stays inside the 5–40% band and the
weights sum to 100%.

### Infeasible constraints

Raising the dividend yield floor to 5% makes the request impossible. My API
returns a 422 naming the constraint and the best achievable value:

> `Min dividend yield of 5.00% is unreachable. The highest achievable within the weight limits is 3.15%.`

The reference tool returns *"Unable to optimize portfolio with current
parameters. Please adjust constraints and retry"* without saying which
constraint is at fault.

## Tests

```bash
pytest -v
```

25 tests. The loader tests cover the date-window behaviour, chronological
ordering, column ordering and the NaN yield. The API tests cover the acceptance
criteria (weights sum to 100, none negative, response shape), all six scenarios,
and the error paths.

## Project structure

```
app/
  loader.py       Workbook access and per-request date alignment
  metrics.py      Volatility, CAGR, drawdown, Sharpe
  optimizer.py    Shared SLSQP solver with multi-start
  strategies.py   Objective function per strategy
  constraints.py  Bounds, inequality constraints, feasibility checks
  factors.py      Factor betas and factor-exposure optimization
  schemas.py      Pydantic request/response models
  main.py         FastAPI application
data/Data.xlsx
tests/
screenshots/      Side-by-side validation against the reference tool
```

## Known limitations

- Factor exposure uses the three factors supplied in the workbook; the
  reference tool uses a five-factor internal model, so betas are not directly
  comparable.
- Dividend yields come from the workbook snapshot and are lower than the
  tool's current values, which affects case 5 as described above.
- The whole workbook is held in memory. Fine for five funds; a production
  deployment would move this behind a database and cache the covariance matrix.
- Minimize-drawdown is the slowest strategy because it rebuilds the equity
  curve on every solver iteration. With more time I would supply an analytic
  gradient or switch to a cutting-plane formulation.
- Only per-security weight bands that apply uniformly are supported. The brief's
  test cases need nothing more, but per-ticker limits would be a small change to
  `build_bounds`.
