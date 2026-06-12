"""Smoke test: call every indicator in TechnicalAnalysisTool via TA-Lib."""
import asyncio
import sys
sys.path.insert(0, ".")

from backend.tools.financial_analysis import TechnicalAnalysisTool

tool = TechnicalAnalysisTool()

# Synthetic OHLCV data (60 bars)
import numpy as np
np.random.seed(42)
n = 60
base = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
close = base.tolist()
high = (base + np.abs(np.random.randn(n)) * 0.3).tolist()
low = (base - np.abs(np.random.randn(n)) * 0.3).tolist()
open_ = (base + np.random.randn(n) * 0.1).tolist()
volume = (np.random.randint(1000, 10000, n).astype(float)).tolist()

ALL_INDICATORS = {
    "rsi":            {"indicator": "rsi", "close": close},
    "macd":           {"indicator": "macd", "close": close},
    "stoch":          {"indicator": "stoch", "close": close, "high": high, "low": low},
    "stochrsi":       {"indicator": "stochrsi", "close": close},
    "cci":            {"indicator": "cci", "close": close, "high": high, "low": low},
    "willr":          {"indicator": "willr", "close": close, "high": high, "low": low},
    "roc":            {"indicator": "roc", "close": close},
    "ao":             {"indicator": "ao", "close": close, "high": high, "low": low},
    "tsi":            {"indicator": "tsi", "close": close},
    "uo":             {"indicator": "uo", "close": close, "high": high, "low": low},
    "sma":            {"indicator": "sma", "close": close},
    "ema":            {"indicator": "ema", "close": close},
    "dema":           {"indicator": "dema", "close": close},
    "tema":           {"indicator": "tema", "close": close},
    "wma":            {"indicator": "wma", "close": close},
    "hma":            {"indicator": "hma", "close": close},
    "kama":           {"indicator": "kama", "close": close},
    "vwma":           {"indicator": "vwma", "close": close, "volume": volume},
    "zlma":           {"indicator": "zlma", "close": close},
    "supertrend":     {"indicator": "supertrend", "close": close, "high": high, "low": low},
    "ichimoku":       {"indicator": "ichimoku", "close": close, "high": high, "low": low},
    "bbands":         {"indicator": "bbands", "close": close},
    "atr":            {"indicator": "atr", "close": close, "high": high, "low": low},
    "natr":           {"indicator": "natr", "close": close, "high": high, "low": low},
    "kc":             {"indicator": "kc", "close": close, "high": high, "low": low},
    "donchian":       {"indicator": "donchian", "close": close, "high": high, "low": low},
    "true_range":     {"indicator": "true_range", "close": close, "high": high, "low": low},
    "obv":            {"indicator": "obv", "close": close, "volume": volume},
    "ad":             {"indicator": "ad", "close": close, "high": high, "low": low, "volume": volume},
    "adosc":          {"indicator": "adosc", "close": close, "high": high, "low": low, "volume": volume},
    "cmf":            {"indicator": "cmf", "close": close, "high": high, "low": low, "volume": volume},
    "mfi":            {"indicator": "mfi", "close": close, "high": high, "low": low, "volume": volume},
    "vwap":           {"indicator": "vwap", "close": close, "high": high, "low": low, "volume": volume},
    "nvi":            {"indicator": "nvi", "close": close, "volume": volume},
    "pvi":            {"indicator": "pvi", "close": close, "volume": volume},
    "adx":            {"indicator": "adx", "close": close, "high": high, "low": low},
    "aroon":          {"indicator": "aroon", "close": close, "high": high, "low": low},
    "chop":           {"indicator": "chop", "close": close, "high": high, "low": low},
    "psar":           {"indicator": "psar", "close": close, "high": high, "low": low},
    "vortex":         {"indicator": "vortex", "close": close, "high": high, "low": low},
    "dpo":            {"indicator": "dpo", "close": close},
    "zscore":         {"indicator": "zscore", "close": close},
    "variance":       {"indicator": "variance", "close": close},
    "stdev":          {"indicator": "stdev", "close": close},
    "skew":           {"indicator": "skew", "close": close},
    "kurtosis":       {"indicator": "kurtosis", "close": close},
    "entropy":        {"indicator": "entropy", "close": close},
    "ha":             {"indicator": "ha", "close": close, "high": high, "low": low, "open": open_},
    "cdl_doji":       {"indicator": "cdl_doji", "close": close, "high": high, "low": low, "open": open_},
    "cdl_inside":     {"indicator": "cdl_inside", "close": close, "high": high, "low": low},
    "cdl_z":          {"indicator": "cdl_z", "close": close, "open": open_},
    "log_return":     {"indicator": "log_return", "close": close},
    "percent_return": {"indicator": "percent_return", "close": close},
}

ALIASES = {
    "bollinger":                  {"indicator": "bollinger", "close": close},
    "stochastic":                 {"indicator": "stochastic", "close": close, "high": high, "low": low},
    "moving_average":             {"indicator": "moving_average", "close": close},
    "exponential_moving_average": {"indicator": "exponential_moving_average", "close": close},
    "average_true_range":         {"indicator": "average_true_range", "close": close, "high": high, "low": low},
    "relative_strength_index":    {"indicator": "relative_strength_index", "close": close},
    "heikin_ashi":                {"indicator": "heikin_ashi", "close": close, "high": high, "low": low, "open": open_},
    "keltner":                    {"indicator": "keltner", "close": close, "high": high, "low": low},
}


async def main():
    passed = 0
    failed = 0
    total = len(ALL_INDICATORS) + len(ALIASES)

    print(f"Testing {len(ALL_INDICATORS)} indicators + {len(ALIASES)} aliases = {total} total")
    print("=" * 70)

    for name, kwargs in ALL_INDICATORS.items():
        try:
            result = await tool.execute(**kwargs)
            cols = result.get("columns", {})
            col_names = list(cols.keys())
            sample = None
            for v in cols.values():
                non_null = [x for x in v if x is not None]
                if non_null:
                    sample = non_null[-1]
                    break
            print(f"  PASS  {name:20s}  cols={col_names}  sample={sample}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name:20s}  {e}")
            failed += 1

    print("-" * 70)
    print("Aliases:")
    for name, kwargs in ALIASES.items():
        try:
            result = await tool.execute(**kwargs)
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}  {e}")
            failed += 1

    print("=" * 70)
    print(f"Result: {passed}/{total} passed, {failed}/{total} failed")
    if failed:
        sys.exit(1)


asyncio.run(main())
