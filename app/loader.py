from functools import lru_cache
from pathlib import Path

import pandas as pd

#consedering 252 trading days in a year
TRADING_DAYS_PER_YEAR = 252

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "Data.xlsx"
# The factor sheet labels its series with these exact strings.
FACTOR_COLUMN_MAP = {
    "Momentum Factor": "momentum",
    "Value Factor": "value",
    "Size Factor": "size",
}

class UnknownTickerError(ValueError):
     """Raised when a request names a ticker that isn't in the workbook."""
     
class MarketData:
    def __init__(self, path=DATA_FILE):
        
        
        if not path.exists():
            raise FileNotFoundError(f"Data workbook not found at {path}")

        info = pd.read_excel(path, sheet_name="Fund Info")
        
        info["dividend_yield"] = info["dividend_yield"].fillna(0.0)

        self._names = dict(zip(info["ticker"], info["fund_name"]))
        self._yields = dict(zip(info["ticker"], info["dividend_yield"]))

        self._fund_returns = self._pivot(
            pd.read_excel(path, sheet_name="Fund Returns"),
            columns="ticker",
        )

        factors = self._pivot(
            pd.read_excel(path, sheet_name="Factor Returns"),
            columns="index_ticker",
        )
        self._factor_returns = factors.rename(columns=FACTOR_COLUMN_MAP)
    @staticmethod
    def _pivot(frame, columns):
        wide = frame.pivot(index="date", columns=columns, values="total_return")
        return wide.sort_index()
    @property
    def available_tickers(self):
        return sorted(self._names)

    def security_name(self, ticker):
        try:
            return self._names[ticker]
        except KeyError:
            raise UnknownTickerError(ticker) from None

    def dividend_yield(self, ticker):
        """Annual dividend yield as a decimal (0.03974 == 3.97%)."""
        try:
            return float(self._yields[ticker])
        except KeyError:
            raise UnknownTickerError(ticker) from None

    def validate_tickers(self, tickers):
        unknown = [t for t in tickers if t not in self._names]
        if unknown:
            raise UnknownTickerError(
                f"Unknown ticker(s): {', '.join(unknown)}. "
                f"Available: {', '.join(self.available_tickers)}"
            )

    def returns_for(self, tickers):
        """Daily returns for `tickers`, over the dates they all share."""
        self.validate_tickers(tickers)
        frame = self._fund_returns[tickers].dropna()

        if len(frame) < 2:
            raise ValueError(
                f"Only {len(frame)} overlapping observations for "
                f"{', '.join(tickers)}."
            )
        return frame
    def aligned_with_factors(self, tickers):
        funds = self.returns_for(tickers)
        factors = self.factor_returns()
        common = funds.index.intersection(factors.index)
        if len(common) < 2:
            raise ValueError("No overlapping dates between funds and factors.")
        return funds.loc[common], factors.loc[common]

    def factor_returns(self):
        return self._factor_returns.dropna()
        
@lru_cache(maxsize=1)
def get_market_data():
    return MarketData()