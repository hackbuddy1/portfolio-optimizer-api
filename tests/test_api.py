"""End-to-end checks against the acceptance criteria in the brief."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ALL_FIVE = [
    {"ticker": "IEFA", "weight": 20},
    {"ticker": "GLD", "weight": 20},
    {"ticker": "AGG", "weight": 20},
    {"ticker": "VEA", "weight": 20},
    {"ticker": "SPY", "weight": 20},
]


def optimize(payload):
    return client.post("/optimize", json=payload)


def weights_of(body):
    return {c["ticker"]: c["optimized_weight"] for c in body["allocation_changes"]}


# --- acceptance checks that must hold for every strategy -------------------

@pytest.mark.parametrize("strategy", [
    "equal_weights",
    "risk_parity",
    "minimize_volatility",
    "minimize_drawdown",
    "maximize_sharpe",
])
def test_weights_sum_to_100_and_are_non_negative(strategy):
    response = optimize({"securities": ALL_FIVE, "strategy": strategy})
    assert response.status_code == 200

    weights = weights_of(response.json())
    assert sum(weights.values()) == pytest.approx(100.0, abs=0.05)
    assert all(w >= 0 for w in weights.values())


def test_response_contains_every_required_field():
    response = optimize({"securities": ALL_FIVE, "strategy": "equal_weights"})
    entry = response.json()["allocation_changes"][0]
    assert set(entry) == {
        "ticker", "security_name", "current_weight",
        "optimized_weight", "change",
    }


# --- the brief's numbered scenarios ---------------------------------------

def test_case_1_equal_weights():
    response = optimize({
        "securities": [
            {"ticker": "IEFA", "weight": 25},
            {"ticker": "SPY", "weight": 75},
        ],
        "strategy": "equal_weights",
    })
    assert weights_of(response.json()) == {"IEFA": 50.0, "SPY": 50.0}


def test_case_2_risk_parity_favours_the_low_volatility_fund():
    response = optimize({
        "securities": [
            {"ticker": "VEA", "weight": 25},
            {"ticker": "AGG", "weight": 75},
        ],
        "strategy": "risk_parity",
    })
    weights = weights_of(response.json())
    # AGG's volatility is roughly a quarter of VEA's, so it needs far more
    # capital to contribute an equal share of risk.
    assert weights["AGG"] > weights["VEA"]
    assert weights["AGG"] == pytest.approx(79.88, abs=0.5)


def test_case_3_minimize_volatility():
    response = optimize({
        "securities": [
            {"ticker": "SPY", "weight": 60},
            {"ticker": "AGG", "weight": 30},
            {"ticker": "GLD", "weight": 10},
        ],
        "strategy": "minimize_volatility",
    })
    assert weights_of(response.json())["AGG"] == pytest.approx(91.21, abs=0.5)


def test_case_5_respects_weight_band_and_dividend_floor():
    response = optimize({
        "securities": ALL_FIVE,
        "strategy": "maximize_sharpe",
        "constraints": {
            "min_weight": 5,
            "max_weight": 40,
            "min_dividend_yield": 2.5,
        },
    })
    assert response.status_code == 200
    weights = weights_of(response.json())

    assert all(5 - 0.01 <= w <= 40 + 0.01 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(100.0, abs=0.05)

    yields = {"IEFA": 3.278, "GLD": 0.0, "AGG": 3.974, "VEA": 2.034, "SPY": 0.987}
    achieved = sum(weights[t] * yields[t] for t in weights) / 100.0
    assert achieved >= 2.5 - 0.01


def test_case_6_momentum_exposure_increases():
    response = optimize({
        "securities": ALL_FIVE,
        "strategy": "optimize_factor_exposure",
        "factor_objective": {"factor": "momentum", "direction": "maximize"},
        "constraints": {"min_weight": 5, "max_weight": 40},
    })
    betas = response.json()["factor_betas"]
    assert betas["optimized_portfolio"]["momentum"] > betas["current_portfolio"]["momentum"]


# --- error handling -------------------------------------------------------

def test_unknown_ticker_returns_400():
    response = optimize({
        "securities": [
            {"ticker": "TSLA", "weight": 50},
            {"ticker": "SPY", "weight": 50},
        ],
        "strategy": "equal_weights",
    })
    assert response.status_code == 400
    assert "TSLA" in response.json()["detail"]


def test_unsupported_strategy_is_rejected():
    response = optimize({"securities": ALL_FIVE, "strategy": "make_me_rich"})
    assert response.status_code == 422


def test_weights_not_summing_to_100_are_rejected():
    response = optimize({
        "securities": [{"ticker": "SPY", "weight": 60}],
        "strategy": "equal_weights",
    })
    assert response.status_code == 422


def test_impossible_dividend_yield_returns_a_clear_error():
    response = optimize({
        "securities": ALL_FIVE,
        "strategy": "maximize_sharpe",
        "constraints": {"min_weight": 5, "max_weight": 40,
                        "min_dividend_yield": 5.0},
    })
    assert response.status_code == 422
    assert "unreachable" in response.json()["detail"].lower()