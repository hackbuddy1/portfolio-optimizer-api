from app.loader import get_market_data

md = get_market_data()

cases = {
    "Case 1": ["IEFA", "SPY"],
    "Case 2": ["VEA", "AGG"],
    "Case 3": ["SPY", "AGG", "GLD"],
    "Case 4": ["IEFA", "GLD", "AGG", "VEA", "SPY"],
}

for name, tickers in cases.items():
    f = md.returns_for(tickers)
    print(f"{name}  {'+'.join(tickers):24} {f.index.min().date()}  {len(f):>5} rows")

print()
print("column order:", md.returns_for(["SPY", "AGG"]).columns.tolist())
print("reversed:    ", md.returns_for(["AGG", "SPY"]).columns.tolist())