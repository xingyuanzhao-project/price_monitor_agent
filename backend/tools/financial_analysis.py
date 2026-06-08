"""
Financial analysis tools for technical indicators, quantitative metrics,
signal detection, and model diagnostics.

What it does:
    Provides four tool implementations for financial analysis:
    - TechnicalAnalysisTool: Computes TA indicators via TA-Lib.
    - QuantitativeAnalysisTool: Portfolio statistics, regressions, and ML models.
    - SignalAnalysisTool: Detects trading signals and patterns in price data.
    - DiagnosticAnalysisTool: Evaluates regression diagnostics and model quality.

Entities in it:
    - TechnicalAnalysisTool: TA-Lib wrapper for momentum, trend, volatility,
      volume, overlap, candle, statistics, and performance indicators.
    - QuantitativeAnalysisTool: Sharpe, drawdown, alpha, beta, VaR/CVaR,
      OLS/WLS/ARIMA/VAR regressions, Ridge/Lasso/PCA models.
    - SignalAnalysisTool: threshold crossing, support/resistance, divergence,
      z-score extremes, volume spikes, cointegration break, HMM regime change, etc.
    - DiagnosticAnalysisTool: autocorrelation, heteroskedasticity, multicollinearity,
      lookahead bias, and overfitting detection.

How used by other modules:
    - Registered in the ToolRegistry at application startup.
    - Called by agents during workflow execution for financial analysis.
    - Receives numeric arrays from upstream data acquisition results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.tools.base import BaseTool, ToolExecutionError


# ---------------------------------------------------------------------------
# Technical Analysis
# ---------------------------------------------------------------------------

class TechnicalAnalysisTool(BaseTool):
    """Computes technical analysis indicators on OHLCV data via TA-Lib.

    Uses TA-Lib for natively supported indicators and numpy/pandas for the
    rest.  Covers momentum, trend, volatility, volume, overlap, candle,
    statistics, and performance categories.
    """

    ALIASES: dict[str, str] = {
        "bollinger": "bbands",
        "stochastic": "stoch",
        "moving_average": "sma",
        "exponential_moving_average": "ema",
        "average_true_range": "atr",
        "relative_strength_index": "rsi",
        "heikin_ashi": "ha",
        "keltner": "kc",
    }

    @property
    def name(self) -> str:
        return "technical_analysis"

    @property
    def description(self) -> str:
        return (
            "Computes technical analysis indicators on OHLCV price data using "
            "TA-Lib. Supports indicators across categories: "
            "momentum (rsi, macd, stoch, stochrsi, cci, willr, roc, ao, tsi, uo), "
            "overlap (sma, ema, dema, tema, wma, hma, kama, vwma, zlma, supertrend, ichimoku), "
            "volatility (bbands, atr, natr, kc, donchian, true_range), "
            "volume (obv, ad, adosc, cmf, mfi, vwap, nvi, pvi), "
            "trend (adx, aroon, chop, psar, vortex, dpo), "
            "statistics (zscore, variance, stdev, skew, kurtosis, entropy), "
            "candles (ha, cdl_doji, cdl_inside, cdl_z), "
            "and performance (log_return, percent_return). "
            "Pass the indicator name and OHLCV arrays; optional params dict "
            "for indicator-specific settings (e.g. {'length': 14})."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "indicator": {
                    "type": "string",
                    "description": (
                        "Indicator name. Examples: 'rsi', 'macd', "
                        "'bbands', 'atr', 'vwap', 'sma', 'ema', 'stoch', 'adx', "
                        "'obv', 'supertrend', 'ichimoku'. Aliases accepted: "
                        "'bollinger' -> 'bbands', 'stochastic' -> 'stoch'."
                    ),
                },
                "close": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of closing prices",
                },
                "high": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of high prices (required for atr, bbands, stoch, vwap, etc.)",
                },
                "low": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of low prices (required for atr, bbands, stoch, vwap, etc.)",
                },
                "open": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of open prices (required for candle indicators, ha, etc.)",
                },
                "volume": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of volume data (required for obv, vwap, mfi, cmf, etc.)",
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Indicator-specific keyword arguments. "
                        "Common: 'length' (lookback period), 'fast'/'slow'/'signal' (MACD), "
                        "'std' (Bollinger std dev multiplier)."
                    ),
                },
            },
            "required": ["indicator", "close"],
        }

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _period(params: dict, default: int = 14) -> int:
        return int(params.get("length", params.get("timeperiod", default)))

    @staticmethod
    def _require(name: str, arr: np.ndarray | None) -> np.ndarray:
        if arr is None:
            raise ToolExecutionError(
                f"'{name}' array is required for this indicator."
            )
        return arr

    # ── execute ────────────────────────────────────────────────────────────

    async def execute(self, **kwargs: Any) -> dict:
        import talib  # noqa: F401 — fail fast if not installed

        indicator = kwargs["indicator"].strip().lower()
        indicator = self.ALIASES.get(indicator, indicator)
        params = dict(kwargs.get("params", None) or {})

        handler = self._DISPATCH.get(indicator)
        if handler is None:
            raise ToolExecutionError(
                f"Unknown indicator: '{indicator}'. "
                f"Supported: {', '.join(sorted(self._DISPATCH.keys()))}"
            )

        close = np.asarray(kwargs["close"], dtype=np.float64)
        high = np.asarray(kwargs["high"], dtype=np.float64) if kwargs.get("high") else None
        low = np.asarray(kwargs["low"], dtype=np.float64) if kwargs.get("low") else None
        open_ = np.asarray(kwargs["open"], dtype=np.float64) if kwargs.get("open") else None
        volume = np.asarray(kwargs["volume"], dtype=np.float64) if kwargs.get("volume") else None

        try:
            result = handler(
                self, close=close, high=high, low=low,
                open_=open_, volume=volume, params=params,
            )
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"Error computing '{indicator}': {exc}"
            ) from exc

        if result is None or (
            isinstance(result, (pd.Series, pd.DataFrame)) and result.empty
        ):
            raise ToolExecutionError(
                f"Indicator '{indicator}' returned no data. "
                f"Check that enough data points are provided for the "
                f"requested lookback period."
            )

        return self._format_result(indicator, result)

    # ── momentum ───────────────────────────────────────────────────────────

    def _ta_rsi(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 14)
        return pd.Series(talib.RSI(close, timeperiod=p), name=f"RSI_{p}")

    def _ta_macd(self, *, close, high, low, open_, volume, params):
        import talib
        fast = int(params.get("fast", params.get("fastperiod", 12)))
        slow = int(params.get("slow", params.get("slowperiod", 26)))
        sig = int(params.get("signal", params.get("signalperiod", 9)))
        macd, signal_line, hist = talib.MACD(
            close, fastperiod=fast, slowperiod=slow, signalperiod=sig,
        )
        return pd.DataFrame({
            f"MACD_{fast}_{slow}_{sig}": macd,
            f"MACDs_{fast}_{slow}_{sig}": signal_line,
            f"MACDh_{fast}_{slow}_{sig}": hist,
        })

    def _ta_stoch(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        fastk = int(params.get("fast", params.get("fastk_period", 14)))
        slowk = int(params.get("slow", params.get("slowk_period", 3)))
        slowd = int(params.get("signal", params.get("slowd_period", 3)))
        k, d = talib.STOCH(
            high, low, close,
            fastk_period=fastk, slowk_period=slowk, slowd_period=slowd,
        )
        return pd.DataFrame({
            f"STOCHk_{fastk}_{slowk}_{slowd}": k,
            f"STOCHd_{fastk}_{slowk}_{slowd}": d,
        })

    def _ta_stochrsi(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 14)
        fastk = int(params.get("fastk_period", 5))
        fastd = int(params.get("fastd_period", 3))
        k, d = talib.STOCHRSI(
            close, timeperiod=p, fastk_period=fastk, fastd_period=fastd,
        )
        return pd.DataFrame({f"STOCHRSIk_{p}": k, f"STOCHRSId_{p}": d})

    def _ta_cci(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        p = self._period(params, 14)
        return pd.Series(talib.CCI(high, low, close, timeperiod=p), name=f"CCI_{p}")

    def _ta_willr(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        p = self._period(params, 14)
        return pd.Series(
            talib.WILLR(high, low, close, timeperiod=p), name=f"WILLR_{p}",
        )

    def _ta_roc(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 10)
        return pd.Series(talib.ROC(close, timeperiod=p), name=f"ROC_{p}")

    def _ta_ao(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        fast = int(params.get("fast", 5))
        slow = int(params.get("slow", 34))
        median = (high + low) / 2.0
        return pd.Series(
            talib.SMA(median, timeperiod=fast) - talib.SMA(median, timeperiod=slow),
            name=f"AO_{fast}_{slow}",
        )

    def _ta_tsi(self, *, close, high, low, open_, volume, params):
        fast = int(params.get("fast", 13))
        slow = int(params.get("slow", 25))
        diff = pd.Series(np.diff(close, prepend=np.nan))
        num = diff.ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()
        den = diff.abs().ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()
        den = den.replace(0, 1e-10)
        tsi = 100.0 * num / den
        tsi.name = f"TSI_{fast}_{slow}"
        return tsi

    def _ta_uo(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        p1 = int(params.get("fast", params.get("timeperiod1", 7)))
        p2 = int(params.get("medium", params.get("timeperiod2", 14)))
        p3 = int(params.get("slow", params.get("timeperiod3", 28)))
        return pd.Series(
            talib.ULTOSC(high, low, close, timeperiod1=p1, timeperiod2=p2, timeperiod3=p3),
            name=f"UO_{p1}_{p2}_{p3}",
        )

    # ── overlap ────────────────────────────────────────────────────────────

    def _ta_sma(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 30)
        return pd.Series(talib.SMA(close, timeperiod=p), name=f"SMA_{p}")

    def _ta_ema(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 30)
        return pd.Series(talib.EMA(close, timeperiod=p), name=f"EMA_{p}")

    def _ta_dema(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 30)
        return pd.Series(talib.DEMA(close, timeperiod=p), name=f"DEMA_{p}")

    def _ta_tema(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 30)
        return pd.Series(talib.TEMA(close, timeperiod=p), name=f"TEMA_{p}")

    def _ta_wma(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 30)
        return pd.Series(talib.WMA(close, timeperiod=p), name=f"WMA_{p}")

    def _ta_hma(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 9)
        half_p = max(int(p / 2), 1)
        sqrt_p = max(int(np.sqrt(p)), 1)
        wma_half = talib.WMA(close, timeperiod=half_p)
        wma_full = talib.WMA(close, timeperiod=p)
        return pd.Series(
            talib.WMA(2.0 * wma_half - wma_full, timeperiod=sqrt_p),
            name=f"HMA_{p}",
        )

    def _ta_kama(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 30)
        return pd.Series(talib.KAMA(close, timeperiod=p), name=f"KAMA_{p}")

    def _ta_vwma(self, *, close, high, low, open_, volume, params):
        vol = self._require("volume", volume)
        p = self._period(params, 20)
        s_cv = pd.Series(close * vol)
        s_v = pd.Series(vol)
        vwma = s_cv.rolling(p).sum() / s_v.rolling(p).sum()
        vwma.name = f"VWMA_{p}"
        return vwma

    def _ta_zlma(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 10)
        ema1 = talib.EMA(close, timeperiod=p)
        return pd.Series(
            talib.EMA(2.0 * close - ema1, timeperiod=p), name=f"ZLMA_{p}",
        )

    def _ta_supertrend(self, *, close, high, low, open_, volume, params):
        import talib
        high_a, low_a = self._require("high", high), self._require("low", low)
        p = self._period(params, 7)
        mult = float(params.get("multiplier", params.get("scalar", 3.0)))
        atr = talib.ATR(high_a, low_a, close, timeperiod=p)
        hl2 = (high_a + low_a) / 2.0
        upper = hl2 + mult * atr
        lower = hl2 - mult * atr
        n = len(close)
        direction = np.ones(n)
        st = np.full(n, np.nan)
        fu, fl = np.copy(upper), np.copy(lower)
        for i in range(1, n):
            if np.isnan(atr[i]):
                continue
            if not (fl[i] > fl[i - 1] or close[i - 1] < fl[i - 1]):
                fl[i] = fl[i - 1]
            if not (fu[i] < fu[i - 1] or close[i - 1] > fu[i - 1]):
                fu[i] = fu[i - 1]
            if direction[i - 1] == 1:
                direction[i] = 1 if close[i] >= fl[i] else -1
            else:
                direction[i] = -1 if close[i] <= fu[i] else 1
            st[i] = fl[i] if direction[i] == 1 else fu[i]
        return pd.DataFrame({
            f"SUPERT_{p}_{mult}": st,
            f"SUPERTd_{p}_{mult}": direction,
            f"SUPERTl_{p}_{mult}": fl,
            f"SUPERTs_{p}_{mult}": fu,
        })

    def _ta_ichimoku(self, *, close, high, low, open_, volume, params):
        high_a, low_a = self._require("high", high), self._require("low", low)
        tenkan_p = int(params.get("tenkan", 9))
        kijun_p = int(params.get("kijun", 26))
        senkou_p = int(params.get("senkou", 52))
        s_h, s_l = pd.Series(high_a), pd.Series(low_a)
        tenkan = (s_h.rolling(tenkan_p).max() + s_l.rolling(tenkan_p).min()) / 2
        kijun = (s_h.rolling(kijun_p).max() + s_l.rolling(kijun_p).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_p)
        senkou_b = (
            (s_h.rolling(senkou_p).max() + s_l.rolling(senkou_p).min()) / 2
        ).shift(kijun_p)
        chikou = pd.Series(close).shift(-kijun_p)
        return pd.DataFrame({
            f"ISA_{tenkan_p}": senkou_a,
            f"ISB_{kijun_p}": senkou_b,
            f"ITS_{tenkan_p}": tenkan,
            f"IKS_{kijun_p}": kijun,
            f"ICS_{kijun_p}": chikou,
        })

    # ── volatility ─────────────────────────────────────────────────────────

    def _ta_bbands(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 5)
        std = float(params.get("std", params.get("nbdevup", 2.0)))
        upper, middle, lower = talib.BBANDS(
            close, timeperiod=p, nbdevup=std, nbdevdn=std, matype=0,
        )
        return pd.DataFrame({
            f"BBU_{p}_{std}": upper,
            f"BBM_{p}_{std}": middle,
            f"BBL_{p}_{std}": lower,
        })

    def _ta_atr(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        p = self._period(params, 14)
        return pd.Series(talib.ATR(high, low, close, timeperiod=p), name=f"ATR_{p}")

    def _ta_natr(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        p = self._period(params, 14)
        return pd.Series(
            talib.NATR(high, low, close, timeperiod=p), name=f"NATR_{p}",
        )

    def _ta_kc(self, *, close, high, low, open_, volume, params):
        import talib
        high_a, low_a = self._require("high", high), self._require("low", low)
        p = self._period(params, 20)
        mult = float(params.get("scalar", params.get("multiplier", 1.5)))
        basis = talib.EMA(close, timeperiod=p)
        atr = talib.ATR(high_a, low_a, close, timeperiod=p)
        return pd.DataFrame({
            f"KCU_{p}_{mult}": basis + mult * atr,
            f"KCB_{p}_{mult}": basis,
            f"KCL_{p}_{mult}": basis - mult * atr,
        })

    def _ta_donchian(self, *, close, high, low, open_, volume, params):
        high_a, low_a = self._require("high", high), self._require("low", low)
        p = self._period(params, 20)
        s_h, s_l = pd.Series(high_a), pd.Series(low_a)
        upper = s_h.rolling(p).max()
        lower = s_l.rolling(p).min()
        return pd.DataFrame({
            f"DCU_{p}": upper, f"DCM_{p}": (upper + lower) / 2, f"DCL_{p}": lower,
        })

    def _ta_true_range(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        return pd.Series(talib.TRANGE(high, low, close), name="TRANGE")

    # ── volume ─────────────────────────────────────────────────────────────

    def _ta_obv(self, *, close, high, low, open_, volume, params):
        import talib
        vol = self._require("volume", volume)
        return pd.Series(talib.OBV(close, vol), name="OBV")

    def _ta_ad(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        vol = self._require("volume", volume)
        return pd.Series(talib.AD(high, low, close, vol), name="AD")

    def _ta_adosc(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        vol = self._require("volume", volume)
        fast = int(params.get("fast", params.get("fastperiod", 3)))
        slow = int(params.get("slow", params.get("slowperiod", 10)))
        return pd.Series(
            talib.ADOSC(high, low, close, vol, fastperiod=fast, slowperiod=slow),
            name=f"ADOSC_{fast}_{slow}",
        )

    def _ta_cmf(self, *, close, high, low, open_, volume, params):
        high_a, low_a = self._require("high", high), self._require("low", low)
        vol = self._require("volume", volume)
        p = self._period(params, 20)
        hl_range = high_a - low_a
        hl_range[hl_range == 0] = 1e-10
        mfv = ((close - low_a) - (high_a - close)) / hl_range
        s_mfv_v = pd.Series(mfv * vol)
        s_v = pd.Series(vol)
        return pd.Series(
            s_mfv_v.rolling(p).sum() / s_v.rolling(p).sum(), name=f"CMF_{p}",
        )

    def _ta_mfi(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        vol = self._require("volume", volume)
        p = self._period(params, 14)
        return pd.Series(
            talib.MFI(high, low, close, vol, timeperiod=p), name=f"MFI_{p}",
        )

    def _ta_vwap(self, *, close, high, low, open_, volume, params):
        high_a, low_a = self._require("high", high), self._require("low", low)
        vol = self._require("volume", volume)
        typical = (high_a + low_a + close) / 3.0
        cum_tp_v = np.cumsum(typical * vol)
        cum_v = np.cumsum(vol)
        cum_v[cum_v == 0] = 1e-10
        return pd.Series(cum_tp_v / cum_v, name="VWAP")

    def _ta_nvi(self, *, close, high, low, open_, volume, params):
        vol = self._require("volume", volume)
        ret = pd.Series(close).pct_change().fillna(0).values
        n = len(close)
        nvi = np.empty(n)
        nvi[0] = 1000.0
        for i in range(1, n):
            nvi[i] = nvi[i - 1] * (1 + ret[i]) if vol[i] < vol[i - 1] else nvi[i - 1]
        return pd.Series(nvi, name="NVI")

    def _ta_pvi(self, *, close, high, low, open_, volume, params):
        vol = self._require("volume", volume)
        ret = pd.Series(close).pct_change().fillna(0).values
        n = len(close)
        pvi = np.empty(n)
        pvi[0] = 1000.0
        for i in range(1, n):
            pvi[i] = pvi[i - 1] * (1 + ret[i]) if vol[i] > vol[i - 1] else pvi[i - 1]
        return pd.Series(pvi, name="PVI")

    # ── trend ──────────────────────────────────────────────────────────────

    def _ta_adx(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        p = self._period(params, 14)
        return pd.Series(talib.ADX(high, low, close, timeperiod=p), name=f"ADX_{p}")

    def _ta_aroon(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        p = self._period(params, 14)
        down, up = talib.AROON(high, low, timeperiod=p)
        return pd.DataFrame({f"AROONd_{p}": down, f"AROONu_{p}": up})

    def _ta_chop(self, *, close, high, low, open_, volume, params):
        import talib
        high_a, low_a = self._require("high", high), self._require("low", low)
        p = self._period(params, 14)
        tr = talib.TRANGE(high_a, low_a, close)
        s_tr = pd.Series(tr)
        s_h, s_l = pd.Series(high_a), pd.Series(low_a)
        sum_tr = s_tr.rolling(p).sum()
        diff = s_h.rolling(p).max() - s_l.rolling(p).min()
        diff = diff.replace(0, 1e-10)
        chop = 100.0 * np.log10(sum_tr / diff) / np.log10(p)
        chop.name = f"CHOP_{p}"
        return chop

    def _ta_psar(self, *, close, high, low, open_, volume, params):
        import talib
        high, low = self._require("high", high), self._require("low", low)
        accel = float(params.get("acceleration", params.get("af", 0.02)))
        maximum = float(params.get("maximum", params.get("af_max", 0.2)))
        return pd.Series(
            talib.SAR(high, low, acceleration=accel, maximum=maximum),
            name=f"PSAR_{accel}_{maximum}",
        )

    def _ta_vortex(self, *, close, high, low, open_, volume, params):
        high_a, low_a = self._require("high", high), self._require("low", low)
        p = self._period(params, 14)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(
            high_a - low_a,
            np.maximum(np.abs(high_a - prev_close), np.abs(low_a - prev_close)),
        )
        prev_low = np.roll(low_a, 1)
        prev_low[0] = low_a[0]
        prev_high = np.roll(high_a, 1)
        prev_high[0] = high_a[0]
        vm_plus = np.abs(high_a - prev_low)
        vm_minus = np.abs(low_a - prev_high)
        s_tr = pd.Series(tr).rolling(p).sum().replace(0, 1e-10)
        vi_plus = pd.Series(vm_plus).rolling(p).sum() / s_tr
        vi_minus = pd.Series(vm_minus).rolling(p).sum() / s_tr
        return pd.DataFrame({f"VTXp_{p}": vi_plus, f"VTXm_{p}": vi_minus})

    def _ta_dpo(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 20)
        shift = p // 2 + 1
        sma = talib.SMA(close, timeperiod=p)
        dpo = pd.Series(close) - pd.Series(sma).shift(shift)
        dpo.name = f"DPO_{p}"
        return dpo

    # ── statistics ─────────────────────────────────────────────────────────

    def _ta_zscore(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 30)
        sma = talib.SMA(close, timeperiod=p)
        std = talib.STDDEV(close, timeperiod=p, nbdev=1)
        std[std == 0] = 1e-10
        return pd.Series((close - sma) / std, name=f"ZS_{p}")

    def _ta_variance(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 30)
        return pd.Series(talib.VAR(close, timeperiod=p, nbdev=1), name=f"VAR_{p}")

    def _ta_stdev(self, *, close, high, low, open_, volume, params):
        import talib
        p = self._period(params, 30)
        return pd.Series(
            talib.STDDEV(close, timeperiod=p, nbdev=1), name=f"STDEV_{p}",
        )

    def _ta_skew(self, *, close, high, low, open_, volume, params):
        p = self._period(params, 30)
        return pd.Series(close).rolling(p).skew().rename(f"SKEW_{p}")

    def _ta_kurtosis(self, *, close, high, low, open_, volume, params):
        p = self._period(params, 30)
        return pd.Series(close).rolling(p).kurt().rename(f"KURT_{p}")

    def _ta_entropy(self, *, close, high, low, open_, volume, params):
        p = self._period(params, 10)

        def _rolling_entropy(window: np.ndarray) -> float:
            bins = max(int(np.sqrt(len(window))), 2)
            counts = np.histogram(window, bins=bins)[0]
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log2(probs)))

        result = pd.Series(close).rolling(p).apply(_rolling_entropy, raw=True)
        result.name = f"ENTROPY_{p}"
        return result

    # ── candles ─────────────────────────────────────────────────────────────

    def _ta_ha(self, *, close, high, low, open_, volume, params):
        open_a = self._require("open", open_)
        high_a = self._require("high", high)
        low_a = self._require("low", low)
        ha_close = (open_a + high_a + low_a + close) / 4.0
        ha_open = np.empty_like(close)
        ha_open[0] = (open_a[0] + close[0]) / 2.0
        for i in range(1, len(close)):
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high = np.maximum(high_a, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(low_a, np.minimum(ha_open, ha_close))
        return pd.DataFrame({
            "HA_open": ha_open, "HA_high": ha_high,
            "HA_low": ha_low, "HA_close": ha_close,
        })

    def _ta_cdl_doji(self, *, close, high, low, open_, volume, params):
        import talib
        open_a = self._require("open", open_)
        high_a = self._require("high", high)
        low_a = self._require("low", low)
        return pd.Series(
            talib.CDLDOJI(open_a, high_a, low_a, close), name="CDL_DOJI",
        )

    def _ta_cdl_inside(self, *, close, high, low, open_, volume, params):
        high_a = self._require("high", high)
        low_a = self._require("low", low)
        inside = np.zeros(len(close))
        for i in range(1, len(close)):
            if high_a[i] < high_a[i - 1] and low_a[i] > low_a[i - 1]:
                inside[i] = 100
        return pd.Series(inside, name="CDL_INSIDE")

    def _ta_cdl_z(self, *, close, high, low, open_, volume, params):
        open_a = self._require("open", open_)
        p = self._period(params, 30)
        body = close - open_a
        s = pd.Series(body)
        mean = s.rolling(p).mean()
        std = s.rolling(p).std()
        std = std.replace(0, 1e-10)
        return pd.Series((body - mean) / std, name=f"CDL_Z_{p}")

    # ── performance ────────────────────────────────────────────────────────

    def _ta_log_return(self, *, close, high, low, open_, volume, params):
        p = int(params.get("length", 1))
        s = pd.Series(close)
        result = np.log(s / s.shift(p))
        result.name = f"LOGRET_{p}"
        return result

    def _ta_percent_return(self, *, close, high, low, open_, volume, params):
        p = int(params.get("length", 1))
        s = pd.Series(close)
        result = s.pct_change(periods=p)
        result.name = f"PCTRET_{p}"
        return result

    # ── dispatch table ─────────────────────────────────────────────────────

    _DISPATCH: dict[str, Any] = {
        "rsi": _ta_rsi, "macd": _ta_macd, "stoch": _ta_stoch,
        "stochrsi": _ta_stochrsi, "cci": _ta_cci, "willr": _ta_willr,
        "roc": _ta_roc, "ao": _ta_ao, "tsi": _ta_tsi, "uo": _ta_uo,
        "sma": _ta_sma, "ema": _ta_ema, "dema": _ta_dema,
        "tema": _ta_tema, "wma": _ta_wma, "hma": _ta_hma,
        "kama": _ta_kama, "vwma": _ta_vwma, "zlma": _ta_zlma,
        "supertrend": _ta_supertrend, "ichimoku": _ta_ichimoku,
        "bbands": _ta_bbands, "atr": _ta_atr, "natr": _ta_natr,
        "kc": _ta_kc, "donchian": _ta_donchian, "true_range": _ta_true_range,
        "obv": _ta_obv, "ad": _ta_ad, "adosc": _ta_adosc,
        "cmf": _ta_cmf, "mfi": _ta_mfi, "vwap": _ta_vwap,
        "nvi": _ta_nvi, "pvi": _ta_pvi,
        "adx": _ta_adx, "aroon": _ta_aroon, "chop": _ta_chop,
        "psar": _ta_psar, "vortex": _ta_vortex, "dpo": _ta_dpo,
        "zscore": _ta_zscore, "variance": _ta_variance, "stdev": _ta_stdev,
        "skew": _ta_skew, "kurtosis": _ta_kurtosis, "entropy": _ta_entropy,
        "ha": _ta_ha, "cdl_doji": _ta_cdl_doji,
        "cdl_inside": _ta_cdl_inside, "cdl_z": _ta_cdl_z,
        "log_return": _ta_log_return, "percent_return": _ta_percent_return,
    }

    @staticmethod
    def _format_result(indicator: str, result: Any) -> dict:
        def _clean(series: pd.Series) -> list:
            return [None if pd.isna(v) else float(v) for v in series]

        if isinstance(result, pd.DataFrame):
            columns = {col: _clean(result[col]) for col in result.columns}
            return {"indicator": indicator, "columns": columns}
        if isinstance(result, pd.Series):
            col_name = result.name if result.name else indicator
            return {"indicator": indicator, "columns": {str(col_name): _clean(result)}}
        return {"indicator": indicator, "result": result}


# ---------------------------------------------------------------------------
# Quantitative Analysis
# ---------------------------------------------------------------------------

class QuantitativeAnalysisTool(BaseTool):
    """Portfolio statistics, regressions, and ML models for quantitative analysis.

    Covers three analysis types:
    - statistic: portfolio/risk metrics (Sharpe, Sortino, drawdown, VaR, etc.)
    - regression: OLS, WLS, logistic, ARIMA, VAR, cross-sectional factor models
    - ml_model: Ridge, Lasso, PCA
    """

    SUPPORTED_STATISTICS = {
        "sharpe", "drawdown", "alpha", "beta", "correlation",
        "volatility", "sortino", "information_ratio", "var", "cvar",
    }
    SUPPORTED_REGRESSIONS = {
        "ols", "wls", "logistic", "cross_sectional", "arima", "var_model",
    }
    SUPPORTED_ML_MODELS = {
        "ridge", "lasso", "pca",
    }

    @property
    def name(self) -> str:
        return "quantitative_analysis"

    @property
    def description(self) -> str:
        return (
            "Performs quantitative analysis: portfolio/risk statistics, "
            "regressions, and ML models. "
            "Statistics: sharpe, drawdown, alpha, beta, correlation, volatility, "
            "sortino, information_ratio, var, cvar. "
            "Regressions (statsmodels/linearmodels): ols, wls, logistic, "
            "cross_sectional (Fama-MacBeth), arima, var_model. "
            "ML models (scikit-learn): ridge, lasso, pca."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": ["statistic", "regression", "ml_model"],
                    "description": "Category of analysis to perform",
                },
                "method": {
                    "type": "string",
                    "description": (
                        "Specific method within the analysis_type. "
                        "statistic: sharpe, drawdown, alpha, beta, correlation, "
                        "volatility, sortino, information_ratio, var, cvar. "
                        "regression: ols, wls, logistic, cross_sectional, arima, var_model. "
                        "ml_model: ridge, lasso, pca."
                    ),
                },
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Period returns array (for statistics)",
                },
                "benchmark_returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Benchmark returns (for alpha, beta, correlation, information_ratio)",
                },
                "y": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Dependent variable (for regressions)",
                },
                "X": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Independent variables as 2D array, each row one observation (for regressions/ML)",
                },
                "target": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Target variable (for ML models)",
                },
                "features": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Feature matrix as 2D array (for ML models / PCA)",
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Method-specific parameters. Common: "
                        "risk_free_rate (default 0.02), periods_per_year (default 252), "
                        "confidence (default 0.95 for VaR/CVaR), "
                        "add_constant (default true for regressions), "
                        "cov_type ('nonrobust', 'HC0'-'HC3', 'HAC'), "
                        "order [p,d,q] (for ARIMA), maxlags (for VAR), "
                        "alpha (regularization for ridge/lasso), "
                        "n_components (for PCA), "
                        "factor_model ('ff3'/'ff5' for cross_sectional with getFamaFrenchFactors), "
                        "frequency ('m'/'d' for Fama-French data)."
                    ),
                },
            },
            "required": ["analysis_type", "method"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        analysis_type = kwargs.get("analysis_type")
        method = kwargs.get("method")
        params = kwargs.get("params") or {}

        if analysis_type == "statistic":
            return self._dispatch_statistic(method, kwargs, params)
        if analysis_type == "regression":
            return self._dispatch_regression(method, kwargs, params)
        if analysis_type == "ml_model":
            return self._dispatch_ml_model(method, kwargs, params)

        raise ToolExecutionError(
            f"Unknown analysis_type: '{analysis_type}'. "
            f"Must be one of: statistic, regression, ml_model"
        )

    # --- statistic dispatch ---

    def _dispatch_statistic(self, method: str, kwargs: dict, params: dict) -> dict:
        if method not in self.SUPPORTED_STATISTICS:
            raise ToolExecutionError(
                f"Unknown statistic: '{method}'. "
                f"Must be one of: {sorted(self.SUPPORTED_STATISTICS)}"
            )
        raw = kwargs.get("returns")
        if not raw or len(raw) < 2:
            raise ToolExecutionError("Statistics require a 'returns' array with at least 2 values")
        returns = np.array(raw, dtype=np.float64)
        benchmark = (
            np.array(kwargs["benchmark_returns"], dtype=np.float64)
            if kwargs.get("benchmark_returns") else None
        )
        handler = getattr(self, f"_stat_{method}")
        result = handler(returns, benchmark, params)
        return {"analysis_type": "statistic", "method": method, "result": result}

    def _require_benchmark(self, benchmark: np.ndarray | None, method: str) -> np.ndarray:
        if benchmark is None or len(benchmark) == 0:
            raise ToolExecutionError(f"'{method}' requires benchmark_returns")
        return benchmark

    def _stat_sharpe(self, returns: np.ndarray, _bench: Any, params: dict) -> dict:
        rfr = params.get("risk_free_rate", 0.02)
        ppy = params.get("periods_per_year", 252)
        excess = returns - rfr / ppy
        std = np.std(excess, ddof=1)
        if std == 0:
            raise ToolExecutionError("Cannot compute Sharpe: zero standard deviation")
        return {"sharpe": float(np.mean(excess) / std * np.sqrt(ppy))}

    def _stat_sortino(self, returns: np.ndarray, _bench: Any, params: dict) -> dict:
        rfr = params.get("risk_free_rate", 0.02)
        ppy = params.get("periods_per_year", 252)
        excess = returns - rfr / ppy
        downside = np.where(excess < 0, excess, 0.0)
        down_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(ppy)
        if down_dev == 0:
            raise ToolExecutionError("Cannot compute Sortino: no downside deviation")
        return {"sortino": float(np.mean(excess) * ppy / down_dev)}

    def _stat_drawdown(self, returns: np.ndarray, _bench: Any, _params: dict) -> dict:
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        dd_series = (cumulative - running_max) / running_max
        return {
            "max_drawdown": float(np.min(dd_series)),
            "drawdown_series": dd_series.tolist(),
        }

    def _stat_volatility(self, returns: np.ndarray, _bench: Any, params: dict) -> dict:
        ppy = params.get("periods_per_year", 252)
        return {"volatility": float(np.std(returns, ddof=1) * np.sqrt(ppy))}

    def _stat_alpha(self, returns: np.ndarray, benchmark: np.ndarray | None, params: dict) -> dict:
        benchmark = self._require_benchmark(benchmark, "alpha")
        rfr = params.get("risk_free_rate", 0.02)
        ppy = params.get("periods_per_year", 252)
        beta_val = self._raw_beta(returns, benchmark)
        ann_ret = np.mean(returns) * ppy
        ann_bench = np.mean(benchmark) * ppy
        return {"alpha": float(ann_ret - (rfr + beta_val * (ann_bench - rfr)))}

    def _stat_beta(self, returns: np.ndarray, benchmark: np.ndarray | None, _params: dict) -> dict:
        benchmark = self._require_benchmark(benchmark, "beta")
        return {"beta": self._raw_beta(returns, benchmark)}

    def _stat_correlation(self, returns: np.ndarray, benchmark: np.ndarray | None, _params: dict) -> dict:
        benchmark = self._require_benchmark(benchmark, "correlation")
        return {"correlation": float(np.corrcoef(returns, benchmark)[0, 1])}

    def _stat_information_ratio(self, returns: np.ndarray, benchmark: np.ndarray | None, params: dict) -> dict:
        benchmark = self._require_benchmark(benchmark, "information_ratio")
        ppy = params.get("periods_per_year", 252)
        active = returns - benchmark
        te = np.std(active, ddof=1)
        if te == 0:
            raise ToolExecutionError("Cannot compute Information Ratio: zero tracking error")
        return {"information_ratio": float(np.mean(active) / te * np.sqrt(ppy))}

    def _stat_var(self, returns: np.ndarray, _bench: Any, params: dict) -> dict:
        confidence = params.get("confidence", 0.95)
        var_val = np.percentile(returns, (1 - confidence) * 100)
        return {"var": float(var_val), "confidence": confidence}

    def _stat_cvar(self, returns: np.ndarray, _bench: Any, params: dict) -> dict:
        confidence = params.get("confidence", 0.95)
        var_val = np.percentile(returns, (1 - confidence) * 100)
        tail = returns[returns <= var_val]
        cvar_val = float(np.mean(tail)) if len(tail) > 0 else float(var_val)
        return {"cvar": cvar_val, "var": float(var_val), "confidence": confidence}

    @staticmethod
    def _raw_beta(returns: np.ndarray, benchmark: np.ndarray) -> float:
        cov = np.cov(returns, benchmark, ddof=1)
        if cov[1, 1] == 0:
            raise ToolExecutionError("Cannot compute beta: benchmark has zero variance")
        return float(cov[0, 1] / cov[1, 1])

    # --- regression dispatch ---

    def _dispatch_regression(self, method: str, kwargs: dict, params: dict) -> dict:
        if method not in self.SUPPORTED_REGRESSIONS:
            raise ToolExecutionError(
                f"Unknown regression: '{method}'. "
                f"Must be one of: {sorted(self.SUPPORTED_REGRESSIONS)}"
            )
        handler = getattr(self, f"_reg_{method}")
        result = handler(kwargs, params)
        return {"analysis_type": "regression", "method": method, "result": result}

    def _require_y_X(self, kwargs: dict, params: dict) -> tuple[np.ndarray, np.ndarray]:
        y_raw = kwargs.get("y")
        X_raw = kwargs.get("X")
        if y_raw is None:
            raise ToolExecutionError("Regressions require 'y' (dependent variable)")
        if X_raw is None:
            raise ToolExecutionError("Regressions require 'X' (independent variables)")
        y = np.array(y_raw, dtype=np.float64)
        X = np.array(X_raw, dtype=np.float64)
        try:
            import statsmodels.api as sm
        except ImportError:
            raise ToolExecutionError("statsmodels is required for regressions: pip install statsmodels")
        if params.get("add_constant", True):
            X = sm.add_constant(X)
        return y, X

    @staticmethod
    def _extract_linear_results(result: Any) -> dict:
        out: dict[str, Any] = {
            "coefficients": result.params.tolist() if hasattr(result.params, 'tolist') else list(result.params),
            "std_errors": result.bse.tolist() if hasattr(result.bse, 'tolist') else list(result.bse),
            "t_statistics": result.tvalues.tolist() if hasattr(result.tvalues, 'tolist') else list(result.tvalues),
            "p_values": result.pvalues.tolist() if hasattr(result.pvalues, 'tolist') else list(result.pvalues),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "nobs": int(result.nobs),
        }
        if hasattr(result, "rsquared"):
            out["r_squared"] = float(result.rsquared)
        if hasattr(result, "rsquared_adj"):
            out["adj_r_squared"] = float(result.rsquared_adj)
        if hasattr(result, "fvalue") and result.fvalue is not None:
            out["f_statistic"] = float(result.fvalue)
            out["f_pvalue"] = float(result.f_pvalue)
        if hasattr(result, "resid"):
            resid = result.resid
            out["residuals"] = resid.tolist() if hasattr(resid, 'tolist') else list(resid)
        return out

    def _reg_ols(self, kwargs: dict, params: dict) -> dict:
        import statsmodels.api as sm
        y, X = self._require_y_X(kwargs, params)
        model = sm.OLS(y, X)
        cov_type = params.get("cov_type", "nonrobust")
        cov_kwds = params.get("cov_kwds", {})
        fit = model.fit(cov_type=cov_type, cov_kwds=cov_kwds)
        return self._extract_linear_results(fit)

    def _reg_wls(self, kwargs: dict, params: dict) -> dict:
        import statsmodels.api as sm
        y, X = self._require_y_X(kwargs, params)
        weights = params.get("weights")
        if weights is None:
            raise ToolExecutionError("WLS requires 'weights' in params")
        fit = sm.WLS(y, X, weights=np.array(weights, dtype=np.float64)).fit()
        return self._extract_linear_results(fit)

    def _reg_logistic(self, kwargs: dict, params: dict) -> dict:
        import statsmodels.api as sm
        y, X = self._require_y_X(kwargs, params)
        fit = sm.Logit(y, X).fit(disp=0)
        out = {
            "coefficients": fit.params.tolist(),
            "std_errors": fit.bse.tolist(),
            "z_statistics": fit.tvalues.tolist(),
            "p_values": fit.pvalues.tolist(),
            "pseudo_r_squared": float(fit.prsquared),
            "log_likelihood": float(fit.llf),
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "nobs": int(fit.nobs),
        }
        return out

    def _reg_cross_sectional(self, kwargs: dict, params: dict) -> dict:
        factor_model = params.get("factor_model")
        y_raw = kwargs.get("y")
        X_raw = kwargs.get("X")

        if factor_model and X_raw is None:
            try:
                import getFamaFrenchFactors as gff
            except ImportError:
                raise ToolExecutionError(
                    "getFamaFrenchFactors is required for auto-fetching factor data: "
                    "pip install getFamaFrenchFactors"
                )
            freq = params.get("frequency", "m")
            if factor_model in ("ff3", "fama_french_3"):
                factors_df = gff.famaFrench3Factor(frequency=freq)
            elif factor_model in ("ff5", "fama_french_5"):
                factors_df = gff.famaFrench5Factor(frequency=freq)
            else:
                raise ToolExecutionError(
                    f"Unknown factor_model: '{factor_model}'. Use 'ff3' or 'ff5'."
                )
            factor_cols = [c for c in factors_df.columns if c != "date_ff_factors"]
            if y_raw is None:
                return {
                    "factors": {col: factors_df[col].tolist() for col in factor_cols},
                    "dates": factors_df.index.astype(str).tolist(),
                    "note": "No y provided; returning factor data only. "
                            "Supply y (portfolio returns) to run the regression.",
                }
            n = min(len(y_raw), len(factors_df))
            factors_df = factors_df.iloc[-n:]
            X_arr = factors_df[factor_cols].values
            y_arr = np.array(y_raw[-n:], dtype=np.float64)
        else:
            if y_raw is None:
                raise ToolExecutionError("cross_sectional requires 'y'")
            if X_raw is None:
                raise ToolExecutionError(
                    "cross_sectional requires 'X' (factor returns) or 'factor_model' param"
                )
            y_arr = np.array(y_raw, dtype=np.float64)
            X_arr = np.array(X_raw, dtype=np.float64)

        try:
            import statsmodels.api as sm
        except ImportError:
            raise ToolExecutionError("statsmodels required: pip install statsmodels")

        X_c = sm.add_constant(X_arr)
        fit = sm.OLS(y_arr, X_c).fit(cov_type=params.get("cov_type", "nonrobust"))
        return self._extract_linear_results(fit)

    def _reg_arima(self, kwargs: dict, params: dict) -> dict:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            raise ToolExecutionError("statsmodels required: pip install statsmodels")

        y_raw = kwargs.get("y") or kwargs.get("returns")
        if y_raw is None:
            raise ToolExecutionError("ARIMA requires 'y' (time series data)")
        y = np.array(y_raw, dtype=np.float64)

        order = tuple(params.get("order", [1, 1, 1]))
        forecast_steps = params.get("forecast_steps", 5)
        fit = ARIMA(y, order=order).fit()
        forecast = fit.forecast(steps=forecast_steps)

        return {
            "coefficients": fit.params.tolist(),
            "std_errors": fit.bse.tolist(),
            "p_values": fit.pvalues.tolist(),
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "residuals": fit.resid.tolist(),
            "forecast": forecast.tolist(),
            "order": list(order),
        }

    def _reg_var_model(self, kwargs: dict, params: dict) -> dict:
        try:
            from statsmodels.tsa.api import VAR
        except ImportError:
            raise ToolExecutionError("statsmodels required: pip install statsmodels")

        X_raw = kwargs.get("X") or kwargs.get("features")
        if X_raw is None:
            raise ToolExecutionError("VAR requires 'X' (multiple time series as 2D array)")
        data = np.array(X_raw, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ToolExecutionError("VAR requires at least 2 time series columns in X")

        maxlags = params.get("maxlags", None)
        forecast_steps = params.get("forecast_steps", 5)
        fit = VAR(data).fit(maxlags=maxlags)
        forecast = fit.forecast(data[-fit.k_ar:], steps=forecast_steps)

        return {
            "aic": float(fit.aic),
            "bic": float(fit.bic),
            "k_ar": int(fit.k_ar),
            "nobs": int(fit.nobs),
            "forecast": forecast.tolist(),
            "residuals": fit.resid.tolist(),
        }

    # --- ml_model dispatch ---

    def _dispatch_ml_model(self, method: str, kwargs: dict, params: dict) -> dict:
        if method not in self.SUPPORTED_ML_MODELS:
            raise ToolExecutionError(
                f"Unknown ML model: '{method}'. "
                f"Must be one of: {sorted(self.SUPPORTED_ML_MODELS)}"
            )
        handler = getattr(self, f"_ml_{method}")
        result = handler(kwargs, params)
        return {"analysis_type": "ml_model", "method": method, "result": result}

    def _require_features_target(self, kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
        features = kwargs.get("features") or kwargs.get("X")
        target = kwargs.get("target") or kwargs.get("y")
        if features is None:
            raise ToolExecutionError("ML models require 'features' (or 'X')")
        if target is None:
            raise ToolExecutionError("ML models require 'target' (or 'y')")
        return np.array(features, dtype=np.float64), np.array(target, dtype=np.float64)

    def _ml_ridge(self, kwargs: dict, params: dict) -> dict:
        try:
            from sklearn.linear_model import Ridge
        except ImportError:
            raise ToolExecutionError("scikit-learn required: pip install scikit-learn")
        X, y = self._require_features_target(kwargs)
        model = Ridge(alpha=params.get("alpha", 1.0))
        model.fit(X, y)
        return {
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "r_squared": float(model.score(X, y)),
        }

    def _ml_lasso(self, kwargs: dict, params: dict) -> dict:
        try:
            from sklearn.linear_model import Lasso
        except ImportError:
            raise ToolExecutionError("scikit-learn required: pip install scikit-learn")
        X, y = self._require_features_target(kwargs)
        model = Lasso(alpha=params.get("alpha", 1.0))
        model.fit(X, y)
        nonzero = int(np.count_nonzero(model.coef_))
        return {
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "r_squared": float(model.score(X, y)),
            "nonzero_coefficients": nonzero,
            "selected_features": int(nonzero),
        }

    def _ml_pca(self, kwargs: dict, params: dict) -> dict:
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ToolExecutionError("scikit-learn required: pip install scikit-learn")
        features = kwargs.get("features") or kwargs.get("X")
        if features is None:
            raise ToolExecutionError("PCA requires 'features' (or 'X')")
        X = np.array(features, dtype=np.float64)
        n_components = params.get("n_components", min(X.shape))
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(X)
        return {
            "components": pca.components_.tolist(),
            "explained_variance": pca.explained_variance_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "n_components": int(pca.n_components_),
            "transformed": transformed.tolist(),
        }


# ---------------------------------------------------------------------------
# Signal Analysis
# ---------------------------------------------------------------------------

class SignalAnalysisTool(BaseTool):
    """Detects trading signals and patterns in price, volume, and indicator data.

    Signal categories:
    - Classic TA: threshold_crossing, support_resistance, divergence,
      pattern_breakout, trend_reversal
    - Momentum / Mean Reversion: z_score_extreme, roc_threshold
    - Volume: volume_spike, price_volume_divergence
    - Volatility: volatility_regime_change, bollinger_squeeze
    - Cross-Asset / Spread: spread_crossing, cointegration_break
    - Regime: regime_change_hmm
    - Multi-Timeframe: fractal_multitimeframe
    """

    SUPPORTED_SIGNALS = {
        "threshold_crossing", "support_resistance", "divergence",
        "pattern_breakout", "trend_reversal",
        "z_score_extreme", "roc_threshold",
        "volume_spike", "price_volume_divergence",
        "volatility_regime_change", "bollinger_squeeze",
        "spread_crossing", "cointegration_break",
        "regime_change_hmm", "fractal_multitimeframe",
    }

    @property
    def name(self) -> str:
        return "signal_analysis"

    @property
    def description(self) -> str:
        return (
            "Detects trading signals and patterns in price/volume/indicator data. "
            "Classic: threshold_crossing, support_resistance, divergence, "
            "pattern_breakout, trend_reversal. "
            "Momentum: z_score_extreme, roc_threshold. "
            "Volume: volume_spike, price_volume_divergence. "
            "Volatility: volatility_regime_change, bollinger_squeeze. "
            "Cross-asset: spread_crossing, cointegration_break. "
            "Regime: regime_change_hmm (Hidden Markov Model). "
            "Multi-timeframe: fractal_multitimeframe."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "signal_type": {
                    "type": "string",
                    "enum": sorted(self.SUPPORTED_SIGNALS),
                    "description": "The type of signal to detect",
                },
                "prices": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Primary price series",
                },
                "prices_b": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Second price series (for spread_crossing, cointegration_break)",
                },
                "volumes": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Volume data (for volume_spike, price_volume_divergence)",
                },
                "indicator_values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Indicator values (for divergence detection)",
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold value (for threshold_crossing, roc_threshold, etc.)",
                },
                "direction": {
                    "type": "string",
                    "enum": ["above", "below", "both"],
                    "description": "Direction filter for crossing detection",
                    "default": "both",
                },
                "lookback_period": {
                    "type": "integer",
                    "description": "Number of periods to look back (default 20)",
                    "default": 20,
                },
                "z_threshold": {
                    "type": "number",
                    "description": "Z-score threshold for extreme detection (default 2.0)",
                    "default": 2.0,
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Signal-specific parameters: "
                        "volume_multiplier (default 2.0 for volume_spike), "
                        "n_states (default 3 for HMM), "
                        "timeframe_multipliers (default [1,5,20] for fractal), "
                        "confidence (default 0.05 for cointegration)."
                    ),
                },
            },
            "required": ["signal_type", "prices"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        signal_type = kwargs.get("signal_type")
        if signal_type not in self.SUPPORTED_SIGNALS:
            raise ToolExecutionError(
                f"Unknown signal_type: '{signal_type}'. "
                f"Must be one of: {sorted(self.SUPPORTED_SIGNALS)}"
            )
        prices = np.array(kwargs["prices"], dtype=np.float64)
        if len(prices) < 3:
            raise ToolExecutionError(f"Need at least 3 price data points, got {len(prices)}")

        handler = getattr(self, f"_sig_{signal_type}")
        return handler(prices, kwargs)

    # --- Classic TA signals ---

    def _sig_threshold_crossing(self, prices: np.ndarray, kw: dict) -> dict:
        threshold = kw.get("threshold")
        if threshold is None:
            raise ToolExecutionError("threshold_crossing requires 'threshold'")
        direction = kw.get("direction", "both")
        crossings = []
        for i in range(1, len(prices)):
            up = prices[i - 1] < threshold <= prices[i]
            down = prices[i - 1] > threshold >= prices[i]
            if direction == "above" and up:
                crossings.append({"index": i, "direction": "above", "price": float(prices[i])})
            elif direction == "below" and down:
                crossings.append({"index": i, "direction": "below", "price": float(prices[i])})
            elif direction == "both" and (up or down):
                crossings.append({"index": i, "direction": "above" if up else "below",
                                   "price": float(prices[i])})
        return {"signal_type": "threshold_crossing", "threshold": threshold, "crossings": crossings}

    def _sig_support_resistance(self, prices: np.ndarray, kw: dict) -> dict:
        lb = kw.get("lookback_period", 20)
        supports, resistances = [], []
        for i in range(1, len(prices) - 1):
            window = prices[max(0, i - lb): min(len(prices), i + lb + 1)]
            if prices[i] == np.min(window):
                supports.append({"index": i, "price": float(prices[i])})
            elif prices[i] == np.max(window):
                resistances.append({"index": i, "price": float(prices[i])})
        return {"signal_type": "support_resistance",
                "support_levels": supports, "resistance_levels": resistances}

    def _sig_divergence(self, prices: np.ndarray, kw: dict) -> dict:
        ind_raw = kw.get("indicator_values")
        if not ind_raw:
            raise ToolExecutionError("divergence requires 'indicator_values'")
        ind = np.array(ind_raw, dtype=np.float64)
        if len(ind) != len(prices):
            raise ToolExecutionError(
                f"prices length ({len(prices)}) != indicator_values length ({len(ind)})"
            )
        lb = kw.get("lookback_period", 20)
        divs = []
        for i in range(lb, len(prices)):
            p_trend = prices[i] - prices[i - lb]
            i_trend = ind[i] - ind[i - lb]
            if p_trend < 0 < i_trend:
                divs.append({"index": i, "type": "bullish",
                             "price": float(prices[i]), "indicator": float(ind[i])})
            elif p_trend > 0 > i_trend:
                divs.append({"index": i, "type": "bearish",
                             "price": float(prices[i]), "indicator": float(ind[i])})
        return {"signal_type": "divergence", "divergences": divs}

    def _sig_pattern_breakout(self, prices: np.ndarray, kw: dict) -> dict:
        lb = kw.get("lookback_period", 20)
        breakouts = []
        for i in range(lb, len(prices)):
            window = prices[i - lb: i]
            w_high, w_low = np.max(window), np.min(window)
            if prices[i] > w_high:
                breakouts.append({"index": i, "direction": "bullish",
                                  "price": float(prices[i]), "broken_level": float(w_high)})
            elif prices[i] < w_low:
                breakouts.append({"index": i, "direction": "bearish",
                                  "price": float(prices[i]), "broken_level": float(w_low)})
        return {"signal_type": "pattern_breakout", "breakouts": breakouts}

    def _sig_trend_reversal(self, prices: np.ndarray, kw: dict) -> dict:
        lb = kw.get("lookback_period", 20)
        short_p = max(lb // 4, 2)
        short_sma = self._rolling_mean(prices, short_p)
        long_sma = self._rolling_mean(prices, lb)
        reversals = []
        for i in range(lb, len(prices)):
            if short_sma[i - 1] < long_sma[i - 1] and short_sma[i] >= long_sma[i]:
                reversals.append({"index": i, "type": "bullish_reversal",
                                  "price": float(prices[i])})
            elif short_sma[i - 1] > long_sma[i - 1] and short_sma[i] <= long_sma[i]:
                reversals.append({"index": i, "type": "bearish_reversal",
                                  "price": float(prices[i])})
        return {"signal_type": "trend_reversal", "reversals": reversals}

    # --- Momentum / Mean Reversion ---

    def _sig_z_score_extreme(self, prices: np.ndarray, kw: dict) -> dict:
        lb = kw.get("lookback_period", 20)
        z_thresh = kw.get("z_threshold", 2.0)
        signals = []
        for i in range(lb, len(prices)):
            window = prices[i - lb: i]
            mu, sigma = np.mean(window), np.std(window, ddof=1)
            if sigma == 0:
                continue
            z = (prices[i] - mu) / sigma
            if z >= z_thresh:
                signals.append({"index": i, "direction": "overbought",
                                "z_score": float(z), "price": float(prices[i])})
            elif z <= -z_thresh:
                signals.append({"index": i, "direction": "oversold",
                                "z_score": float(z), "price": float(prices[i])})
        return {"signal_type": "z_score_extreme", "z_threshold": z_thresh, "signals": signals}

    def _sig_roc_threshold(self, prices: np.ndarray, kw: dict) -> dict:
        lb = kw.get("lookback_period", 20)
        threshold = kw.get("threshold")
        if threshold is None:
            raise ToolExecutionError("roc_threshold requires 'threshold' (e.g. 0.05 for 5%)")
        signals = []
        for i in range(lb, len(prices)):
            if prices[i - lb] == 0:
                continue
            roc = (prices[i] - prices[i - lb]) / prices[i - lb]
            if roc >= threshold:
                signals.append({"index": i, "direction": "momentum_up",
                                "roc": float(roc), "price": float(prices[i])})
            elif roc <= -threshold:
                signals.append({"index": i, "direction": "momentum_down",
                                "roc": float(roc), "price": float(prices[i])})
        return {"signal_type": "roc_threshold", "threshold": threshold, "signals": signals}

    # --- Volume-Based ---

    def _sig_volume_spike(self, prices: np.ndarray, kw: dict) -> dict:
        vol_raw = kw.get("volumes")
        if not vol_raw:
            raise ToolExecutionError("volume_spike requires 'volumes'")
        volumes = np.array(vol_raw, dtype=np.float64)
        lb = kw.get("lookback_period", 20)
        params = kw.get("params") or {}
        multiplier = params.get("volume_multiplier", 2.0)
        signals = []
        for i in range(lb, len(volumes)):
            avg_vol = np.mean(volumes[i - lb: i])
            if avg_vol > 0 and volumes[i] >= avg_vol * multiplier:
                signals.append({
                    "index": i,
                    "volume": float(volumes[i]),
                    "avg_volume": float(avg_vol),
                    "ratio": float(volumes[i] / avg_vol),
                    "price": float(prices[i]) if i < len(prices) else None,
                })
        return {"signal_type": "volume_spike", "multiplier": multiplier, "signals": signals}

    def _sig_price_volume_divergence(self, prices: np.ndarray, kw: dict) -> dict:
        vol_raw = kw.get("volumes")
        if not vol_raw:
            raise ToolExecutionError("price_volume_divergence requires 'volumes'")
        volumes = np.array(vol_raw, dtype=np.float64)
        lb = kw.get("lookback_period", 20)
        signals = []
        for i in range(lb, len(prices)):
            p_change = prices[i] - prices[i - lb]
            v_change = volumes[i] - volumes[i - lb]
            if p_change > 0 and v_change < 0:
                signals.append({"index": i, "type": "bearish_divergence",
                                "price": float(prices[i]),
                                "note": "price rising on declining volume (distribution)"})
            elif p_change < 0 and v_change > 0:
                signals.append({"index": i, "type": "bullish_divergence",
                                "price": float(prices[i]),
                                "note": "price falling on rising volume (accumulation)"})
        return {"signal_type": "price_volume_divergence", "signals": signals}

    # --- Volatility-Based ---

    def _sig_volatility_regime_change(self, prices: np.ndarray, kw: dict) -> dict:
        lb = kw.get("lookback_period", 20)
        params = kw.get("params") or {}
        vol_threshold = params.get("vol_threshold", 1.5)
        returns = np.diff(prices) / prices[:-1]
        signals = []
        for i in range(lb, len(returns)):
            current_vol = np.std(returns[i - lb // 2: i], ddof=1)
            prior_vol = np.std(returns[i - lb: i - lb // 2], ddof=1)
            if prior_vol > 0:
                ratio = current_vol / prior_vol
                if ratio >= vol_threshold:
                    signals.append({"index": i + 1, "type": "vol_expansion",
                                    "ratio": float(ratio), "price": float(prices[i + 1])})
                elif ratio <= 1.0 / vol_threshold:
                    signals.append({"index": i + 1, "type": "vol_contraction",
                                    "ratio": float(ratio), "price": float(prices[i + 1])})
        return {"signal_type": "volatility_regime_change", "signals": signals}

    def _sig_bollinger_squeeze(self, prices: np.ndarray, kw: dict) -> dict:
        lb = kw.get("lookback_period", 20)
        params = kw.get("params") or {}
        bandwidth_threshold = params.get("bandwidth_threshold", 0.04)
        signals = []
        for i in range(lb, len(prices)):
            window = prices[i - lb: i + 1]
            mu = np.mean(window)
            sigma = np.std(window, ddof=1)
            if mu == 0:
                continue
            bandwidth = (4 * sigma) / mu
            if bandwidth <= bandwidth_threshold:
                signals.append({"index": i, "type": "squeeze",
                                "bandwidth": float(bandwidth), "price": float(prices[i])})
        if signals:
            for i in range(1, len(signals)):
                curr_idx = signals[i]["index"]
                prev_idx = signals[i - 1]["index"]
                if curr_idx - prev_idx > lb:
                    idx_between = signals[i - 1]["index"] + 1
                    if idx_between < len(prices):
                        post_squeeze_prices = prices[prev_idx: curr_idx]
                        if len(post_squeeze_prices) > 1:
                            move = post_squeeze_prices[-1] - post_squeeze_prices[0]
                            signals[i - 1]["expansion_direction"] = "up" if move > 0 else "down"
                            signals[i - 1]["expansion_magnitude"] = float(abs(move))
        return {"signal_type": "bollinger_squeeze", "bandwidth_threshold": bandwidth_threshold,
                "signals": signals}

    # --- Cross-Asset / Spread ---

    def _sig_spread_crossing(self, prices: np.ndarray, kw: dict) -> dict:
        prices_b = kw.get("prices_b")
        if prices_b is None:
            raise ToolExecutionError("spread_crossing requires 'prices_b' (second price series)")
        b = np.array(prices_b, dtype=np.float64)
        if len(b) != len(prices):
            raise ToolExecutionError(
                f"prices ({len(prices)}) and prices_b ({len(b)}) must have same length"
            )
        spread = prices - b
        threshold = kw.get("threshold", 0.0)
        direction = kw.get("direction", "both")
        crossings = []
        for i in range(1, len(spread)):
            up = spread[i - 1] < threshold <= spread[i]
            down = spread[i - 1] > threshold >= spread[i]
            if direction == "above" and up:
                crossings.append({"index": i, "direction": "above",
                                  "spread": float(spread[i])})
            elif direction == "below" and down:
                crossings.append({"index": i, "direction": "below",
                                  "spread": float(spread[i])})
            elif direction == "both" and (up or down):
                crossings.append({"index": i, "direction": "above" if up else "below",
                                  "spread": float(spread[i])})
        return {"signal_type": "spread_crossing", "threshold": threshold, "crossings": crossings}

    def _sig_cointegration_break(self, prices: np.ndarray, kw: dict) -> dict:
        prices_b = kw.get("prices_b")
        if prices_b is None:
            raise ToolExecutionError("cointegration_break requires 'prices_b'")
        try:
            from statsmodels.tsa.stattools import coint
        except ImportError:
            raise ToolExecutionError("statsmodels required: pip install statsmodels")

        b = np.array(prices_b, dtype=np.float64)
        if len(b) != len(prices):
            raise ToolExecutionError("prices and prices_b must have same length")

        lb = kw.get("lookback_period", 20)
        params = kw.get("params") or {}
        confidence = params.get("confidence", 0.05)
        window_size = params.get("window_size", max(lb * 3, 60))

        signals = []
        step = max(lb // 2, 5)
        for i in range(window_size, len(prices), step):
            w_a = prices[i - window_size: i]
            w_b = b[i - window_size: i]
            _, p_value, _ = coint(w_a, w_b)
            cointegrated = p_value < confidence
            if not cointegrated:
                signals.append({
                    "index": i,
                    "p_value": float(p_value),
                    "cointegrated": False,
                    "note": "Cointegration broken — pair has decoupled",
                })
        return {"signal_type": "cointegration_break", "confidence": confidence, "signals": signals}

    # --- Regime ---

    def _sig_regime_change_hmm(self, prices: np.ndarray, kw: dict) -> dict:
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ToolExecutionError("hmmlearn required: pip install hmmlearn")

        params = kw.get("params") or {}
        n_states = params.get("n_states", 3)
        returns = np.diff(np.log(prices)).reshape(-1, 1)

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=params.get("random_state", 42),
        )
        model.fit(returns)
        states = model.predict(returns)

        state_means = {int(s): float(model.means_[s][0]) for s in range(n_states)}
        sorted_states = sorted(state_means, key=lambda s: state_means[s])
        label_map = {}
        if n_states >= 3:
            label_map[sorted_states[0]] = "bear"
            label_map[sorted_states[-1]] = "bull"
            for s in sorted_states[1:-1]:
                label_map[s] = "sideways"
        else:
            for idx, s in enumerate(sorted_states):
                label_map[s] = ["bear", "bull"][idx] if n_states == 2 else f"regime_{idx}"

        transitions = []
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                transitions.append({
                    "index": i + 1,
                    "from_regime": label_map.get(int(states[i - 1]), str(states[i - 1])),
                    "to_regime": label_map.get(int(states[i]), str(states[i])),
                    "price": float(prices[i + 1]),
                })

        return {
            "signal_type": "regime_change_hmm",
            "n_states": n_states,
            "state_means": {label_map.get(k, str(k)): v for k, v in state_means.items()},
            "regime_sequence": [label_map.get(int(s), str(s)) for s in states],
            "transitions": transitions,
        }

    # --- Multi-Timeframe ---

    def _sig_fractal_multitimeframe(self, prices: np.ndarray, kw: dict) -> dict:
        params = kw.get("params") or {}
        multipliers = params.get("timeframe_multipliers", [1, 5, 20])
        lb = kw.get("lookback_period", 20)

        tf_trends: dict[int, list] = {}
        for mult in multipliers:
            resampled = self._resample(prices, mult)
            short_p = max(lb // 4, 2)
            short_sma = self._rolling_mean(resampled, short_p)
            long_sma = self._rolling_mean(resampled, lb)
            trends = []
            for i in range(lb, len(resampled)):
                if short_sma[i] > long_sma[i]:
                    trends.append("up")
                elif short_sma[i] < long_sma[i]:
                    trends.append("down")
                else:
                    trends.append("flat")
            tf_trends[mult] = trends

        signals = []
        base_trends = tf_trends.get(multipliers[0], [])
        for i in range(len(base_trends)):
            aligned_trends = []
            for mult in multipliers:
                t_list = tf_trends.get(mult, [])
                mapped_idx = i // mult if mult > 1 else i
                if 0 <= mapped_idx < len(t_list):
                    aligned_trends.append(t_list[mapped_idx])
            if len(aligned_trends) == len(multipliers) and len(set(aligned_trends)) == 1:
                original_idx = i + lb
                if original_idx < len(prices):
                    signals.append({
                        "index": original_idx,
                        "direction": aligned_trends[0],
                        "timeframes_aligned": multipliers,
                        "price": float(prices[original_idx]),
                    })

        return {"signal_type": "fractal_multitimeframe", "timeframe_multipliers": multipliers,
                "signals": signals}

    # --- Shared helpers ---

    @staticmethod
    def _rolling_mean(data: np.ndarray, period: int) -> np.ndarray:
        out = np.empty(len(data))
        for i in range(len(data)):
            start = max(0, i - period + 1)
            out[i] = np.mean(data[start: i + 1])
        return out

    @staticmethod
    def _resample(data: np.ndarray, factor: int) -> np.ndarray:
        if factor <= 1:
            return data
        n = len(data) // factor
        return np.array([data[(i + 1) * factor - 1] for i in range(n)])


# ---------------------------------------------------------------------------
# Diagnostic Analysis
# ---------------------------------------------------------------------------

class DiagnosticAnalysisTool(BaseTool):
    """Evaluates regression diagnostics and model quality.

    Diagnostic types:
    - autocorrelation: Durbin-Watson, Ljung-Box, Newey-West standard errors
    - heteroskedasticity: Breusch-Pagan, White's test
    - multicollinearity: Variance Inflation Factor (VIF) per feature
    - lookahead_bias: detects features correlated more with future y than current y
    - overfitting: in-sample vs cross-validated R² comparison
    """

    SUPPORTED_DIAGNOSTICS = {
        "autocorrelation", "heteroskedasticity", "multicollinearity",
        "lookahead_bias", "overfitting",
    }

    @property
    def name(self) -> str:
        return "diagnostic_analysis"

    @property
    def description(self) -> str:
        return (
            "Evaluates regression diagnostics and model quality. "
            "autocorrelation: Durbin-Watson + Ljung-Box tests on residuals, "
            "plus Newey-West corrected standard errors. "
            "heteroskedasticity: Breusch-Pagan and White's tests. "
            "multicollinearity: Variance Inflation Factor (VIF) per feature. "
            "lookahead_bias: checks if features correlate more with future target "
            "than current target. "
            "overfitting: compares in-sample R² with time-series cross-validated R²."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "diagnostic_type": {
                    "type": "string",
                    "enum": sorted(self.SUPPORTED_DIAGNOSTICS),
                    "description": "The diagnostic test to run",
                },
                "y": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Dependent variable (target)",
                },
                "X": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Independent variables as 2D array",
                },
                "residuals": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Pre-computed residuals (optional; if omitted, OLS is run internally)",
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Test-specific parameters: "
                        "nlags (for autocorrelation, default 10), "
                        "test_type ('breusch_pagan' or 'white' for heteroskedasticity), "
                        "n_splits (for overfitting cross-validation, default 5)."
                    ),
                },
            },
            "required": ["diagnostic_type"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        diag_type = kwargs.get("diagnostic_type")
        if diag_type not in self.SUPPORTED_DIAGNOSTICS:
            raise ToolExecutionError(
                f"Unknown diagnostic_type: '{diag_type}'. "
                f"Must be one of: {sorted(self.SUPPORTED_DIAGNOSTICS)}"
            )
        handler = getattr(self, f"_diag_{diag_type}")
        result = handler(kwargs)
        return {"diagnostic_type": diag_type, "result": result}

    def _get_residuals(self, kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Return (residuals, X) — runs OLS internally if residuals not provided."""
        resid_raw = kwargs.get("residuals")
        X_raw = kwargs.get("X")
        if X_raw is None:
            raise ToolExecutionError("Diagnostic requires 'X' (independent variables)")
        X = np.array(X_raw, dtype=np.float64)

        if resid_raw is not None:
            return np.array(resid_raw, dtype=np.float64), X

        y_raw = kwargs.get("y")
        if y_raw is None:
            raise ToolExecutionError("Provide either 'residuals' or both 'y' and 'X'")

        try:
            import statsmodels.api as sm
        except ImportError:
            raise ToolExecutionError("statsmodels required: pip install statsmodels")

        y = np.array(y_raw, dtype=np.float64)
        X_c = sm.add_constant(X)
        fit = sm.OLS(y, X_c).fit()
        return np.array(fit.resid), X

    def _diag_autocorrelation(self, kwargs: dict) -> dict:
        try:
            import statsmodels.api as sm
            from statsmodels.stats.stattools import durbin_watson
            from statsmodels.stats.diagnostic import acorr_ljungbox
        except ImportError:
            raise ToolExecutionError("statsmodels required: pip install statsmodels")

        residuals, X = self._get_residuals(kwargs)
        params = kwargs.get("params") or {}
        nlags = params.get("nlags", 10)

        dw = float(durbin_watson(residuals))
        lb_result = acorr_ljungbox(residuals, lags=nlags, return_df=True)

        nw_info = None
        y_raw = kwargs.get("y")
        if y_raw is not None:
            y = np.array(y_raw, dtype=np.float64)
            X_c = sm.add_constant(X)
            nw_fit = sm.OLS(y, X_c).fit(cov_type='HAC', cov_kwds={'maxlags': nlags})
            nw_info = {
                "newey_west_std_errors": nw_fit.bse.tolist(),
                "newey_west_p_values": nw_fit.pvalues.tolist(),
            }

        return {
            "durbin_watson": dw,
            "durbin_watson_interpretation": (
                "positive autocorrelation" if dw < 1.5
                else "no significant autocorrelation" if dw <= 2.5
                else "negative autocorrelation"
            ),
            "ljung_box": {
                "statistics": lb_result["lb_stat"].tolist(),
                "p_values": lb_result["lb_pvalue"].tolist(),
            },
            **(nw_info or {}),
        }

    def _diag_heteroskedasticity(self, kwargs: dict) -> dict:
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan, het_white
            import statsmodels.api as sm
        except ImportError:
            raise ToolExecutionError("statsmodels required: pip install statsmodels")

        residuals, X = self._get_residuals(kwargs)
        params = kwargs.get("params") or {}
        test_type = params.get("test_type", "both")
        X_c = sm.add_constant(X)

        results = {}
        if test_type in ("breusch_pagan", "both"):
            bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(residuals, X_c)
            results["breusch_pagan"] = {
                "lm_statistic": float(bp_stat), "lm_p_value": float(bp_p),
                "f_statistic": float(bp_f), "f_p_value": float(bp_fp),
                "heteroskedastic": bp_p < 0.05,
            }
        if test_type in ("white", "both"):
            w_stat, w_p, w_f, w_fp = het_white(residuals, X_c)
            results["white"] = {
                "lm_statistic": float(w_stat), "lm_p_value": float(w_p),
                "f_statistic": float(w_f), "f_p_value": float(w_fp),
                "heteroskedastic": w_p < 0.05,
            }
        return results

    def _diag_multicollinearity(self, kwargs: dict) -> dict:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            import statsmodels.api as sm
        except ImportError:
            raise ToolExecutionError("statsmodels required: pip install statsmodels")

        X_raw = kwargs.get("X")
        if X_raw is None:
            raise ToolExecutionError("multicollinearity requires 'X'")
        X = np.array(X_raw, dtype=np.float64)
        X_c = sm.add_constant(X)
        vifs = []
        for i in range(X_c.shape[1]):
            vif_val = float(variance_inflation_factor(X_c, i))
            vifs.append({
                "feature_index": i,
                "vif": vif_val,
                "problematic": vif_val > 10,
            })
        return {
            "vif_scores": vifs[1:],  # skip the constant column
            "any_problematic": any(v["problematic"] for v in vifs[1:]),
            "threshold": 10.0,
        }

    def _diag_lookahead_bias(self, kwargs: dict) -> dict:
        y_raw = kwargs.get("y")
        X_raw = kwargs.get("X")
        if y_raw is None or X_raw is None:
            raise ToolExecutionError("lookahead_bias requires both 'y' and 'X'")
        y = np.array(y_raw, dtype=np.float64)
        X = np.array(X_raw, dtype=np.float64)
        params = kwargs.get("params") or {}
        ratio_threshold = params.get("ratio_threshold", 1.5)

        features = []
        for col in range(X.shape[1]):
            x_col = X[:, col]
            corr_current = float(np.corrcoef(x_col[:-1], y[:-1])[0, 1])
            corr_future = float(np.corrcoef(x_col[:-1], y[1:])[0, 1])
            suspicious = abs(corr_future) > abs(corr_current) * ratio_threshold
            features.append({
                "feature_index": col,
                "corr_with_current_y": corr_current,
                "corr_with_next_y": corr_future,
                "suspicious": suspicious,
            })

        return {
            "features": features,
            "any_suspicious": any(f["suspicious"] for f in features),
            "ratio_threshold": ratio_threshold,
        }

    def _diag_overfitting(self, kwargs: dict) -> dict:
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        except ImportError:
            raise ToolExecutionError("scikit-learn required: pip install scikit-learn")

        y_raw = kwargs.get("y")
        X_raw = kwargs.get("X")
        if y_raw is None or X_raw is None:
            raise ToolExecutionError("overfitting requires both 'y' and 'X'")
        y = np.array(y_raw, dtype=np.float64)
        X = np.array(X_raw, dtype=np.float64)
        params = kwargs.get("params") or {}
        n_splits = params.get("n_splits", 5)

        model = LinearRegression()
        model.fit(X, y)
        in_sample_r2 = float(model.score(X, y))

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
        cv_mean = float(np.mean(cv_scores))

        overfit_ratio = in_sample_r2 / max(cv_mean, 1e-10) if cv_mean > 0 else float("inf")

        return {
            "in_sample_r2": in_sample_r2,
            "cv_mean_r2": cv_mean,
            "cv_std_r2": float(np.std(cv_scores)),
            "cv_scores": cv_scores.tolist(),
            "overfit_ratio": float(overfit_ratio),
            "likely_overfit": overfit_ratio > 1.5,
            "n_splits": n_splits,
        }
