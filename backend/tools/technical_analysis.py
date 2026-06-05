"""
Technical analysis tools for computing indicators, statistics, and detecting signals.

What it does:
    Provides three concrete tool implementations for quantitative analysis:
    computing technical indicators (RSI, MACD, Bollinger, etc.), computing
    financial statistics (Sharpe, drawdown, etc.), and detecting trading
    signals (threshold crossing, pattern breakout, etc.).

Entities in it:
    - ComputeIndicatorTool: Computes technical indicators on price/volume data.
    - ComputeStatisticTool: Computes portfolio/risk statistics.
    - DetectSignalTool: Detects trading signals and patterns in price data.

How used by other modules:
    - Registered in the ToolRegistry at application startup.
    - Called by agents during workflow execution for quantitative analysis.
    - Receives numeric arrays from upstream data acquisition results.
"""

from typing import Any

import numpy as np

from backend.tools.base import BaseTool, ToolExecutionError


class ComputeIndicatorTool(BaseTool):
    """
    Computes technical indicators on price and volume data arrays.

    Description:
        Supports computing RSI, MACD, Bollinger Bands, ATR, VWAP, SMA, EMA,
        and Stochastic oscillator from input price/volume arrays.

    Attributes:
        SUPPORTED_INDICATORS: Class-level set of supported indicator names.

    Methods:
        name: Returns 'compute_indicator'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for indicator computation.
        execute: Computes the specified indicator on the input data.
    """

    SUPPORTED_INDICATORS = {"rsi", "macd", "bollinger", "atr", "vwap", "sma", "ema", "stochastic"}

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'compute_indicator'
        """
        return "compute_indicator"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the indicators that can be computed.

        Params:
            None

        Returns:
            str: Description string.
        """
        return (
            "Computes technical indicators on numeric data arrays. "
            "Supported: rsi, macd, bollinger, atr, vwap, sma, ema, stochastic."
        )

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines indicator_type, data arrays, and configuration parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "indicator_type": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_INDICATORS),
                    "description": "The indicator to compute",
                },
                "close_prices": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of closing prices",
                },
                "high_prices": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of high prices (for ATR, Bollinger, Stochastic)",
                },
                "low_prices": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of low prices (for ATR, Bollinger, Stochastic)",
                },
                "volumes": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of volume data (for VWAP)",
                },
                "period": {
                    "type": "integer",
                    "description": "Lookback period for the indicator",
                    "default": 14,
                },
            },
            "required": ["indicator_type", "close_prices"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Compute the specified technical indicator.

        Description:
            Dispatches to the appropriate computation method based on
            indicator_type and returns the result array(s).

        Params:
            **kwargs (Any): Must include indicator_type and close_prices at minimum.

        Returns:
            dict: Dictionary with indicator name as key and result values.

        Raises:
            ToolExecutionError: If indicator_type is unsupported or data is insufficient.
        """
        indicator_type = kwargs.get("indicator_type")
        if indicator_type not in self.SUPPORTED_INDICATORS:
            raise ToolExecutionError(
                f"Unsupported indicator_type: '{indicator_type}'. "
                f"Must be one of: {sorted(self.SUPPORTED_INDICATORS)}"
            )

        close_prices = np.array(kwargs["close_prices"], dtype=np.float64)
        period = kwargs.get("period", 14)

        if len(close_prices) < period:
            raise ToolExecutionError(
                f"Insufficient data: got {len(close_prices)} prices but "
                f"indicator '{indicator_type}' requires at least {period} data points"
            )

        if indicator_type == "rsi":
            return {"rsi": self._compute_rsi(close_prices, period).tolist()}
        elif indicator_type == "macd":
            return self._compute_macd(close_prices)
        elif indicator_type == "bollinger":
            return self._compute_bollinger(close_prices, period)
        elif indicator_type == "atr":
            high_prices = np.array(kwargs.get("high_prices", []), dtype=np.float64)
            low_prices = np.array(kwargs.get("low_prices", []), dtype=np.float64)
            if len(high_prices) == 0 or len(low_prices) == 0:
                raise ToolExecutionError("ATR requires high_prices and low_prices arrays")
            return {"atr": self._compute_atr(high_prices, low_prices, close_prices, period).tolist()}
        elif indicator_type == "vwap":
            high_prices = np.array(kwargs.get("high_prices", []), dtype=np.float64)
            low_prices = np.array(kwargs.get("low_prices", []), dtype=np.float64)
            volumes = np.array(kwargs.get("volumes", []), dtype=np.float64)
            if len(volumes) == 0:
                raise ToolExecutionError("VWAP requires volumes array")
            return {"vwap": self._compute_vwap(high_prices, low_prices, close_prices, volumes).tolist()}
        elif indicator_type == "sma":
            return {"sma": self._compute_sma(close_prices, period).tolist()}
        elif indicator_type == "ema":
            return {"ema": self._compute_ema(close_prices, period).tolist()}
        elif indicator_type == "stochastic":
            high_prices = np.array(kwargs.get("high_prices", []), dtype=np.float64)
            low_prices = np.array(kwargs.get("low_prices", []), dtype=np.float64)
            if len(high_prices) == 0 or len(low_prices) == 0:
                raise ToolExecutionError("Stochastic requires high_prices and low_prices arrays")
            return self._compute_stochastic(high_prices, low_prices, close_prices, period)

        raise ToolExecutionError(f"Unhandled indicator_type: '{indicator_type}'")

    def _compute_rsi(self, close_prices: np.ndarray, period: int) -> np.ndarray:
        """
        Compute Relative Strength Index.

        Description:
            Calculates RSI using exponential moving average of gains and losses.

        Params:
            close_prices (np.ndarray): Array of closing prices.
            period (int): RSI lookback period.

        Returns:
            np.ndarray: RSI values array.
        """
        deltas = np.diff(close_prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.zeros(len(deltas))
        avg_loss = np.zeros(len(deltas))

        avg_gain[period - 1] = np.mean(gains[:period])
        avg_loss[period - 1] = np.mean(losses[:period])

        for index in range(period, len(deltas)):
            avg_gain[index] = (avg_gain[index - 1] * (period - 1) + gains[index]) / period
            avg_loss[index] = (avg_loss[index - 1] * (period - 1) + losses[index]) / period

        relative_strength = np.divide(
            avg_gain[period - 1:],
            avg_loss[period - 1:],
            out=np.zeros_like(avg_gain[period - 1:]),
            where=avg_loss[period - 1:] != 0,
        )
        rsi_values = 100.0 - (100.0 / (1.0 + relative_strength))
        return rsi_values

    def _compute_macd(self, close_prices: np.ndarray) -> dict:
        """
        Compute MACD line, signal line, and histogram.

        Description:
            Uses 12-period and 26-period EMA for MACD line, 9-period EMA for signal.

        Params:
            close_prices (np.ndarray): Array of closing prices.

        Returns:
            dict: Dictionary with macd_line, signal_line, and histogram arrays.
        """
        ema_12 = self._compute_ema(close_prices, 12)
        ema_26 = self._compute_ema(close_prices, 26)
        macd_line = ema_12 - ema_26
        signal_line = self._compute_ema(macd_line, 9)
        histogram = macd_line - signal_line
        return {
            "macd_line": macd_line.tolist(),
            "signal_line": signal_line.tolist(),
            "histogram": histogram.tolist(),
        }

    def _compute_bollinger(self, close_prices: np.ndarray, period: int) -> dict:
        """
        Compute Bollinger Bands (middle, upper, lower).

        Description:
            Middle band is SMA, upper/lower are +/- 2 standard deviations.

        Params:
            close_prices (np.ndarray): Array of closing prices.
            period (int): Lookback period.

        Returns:
            dict: Dictionary with middle_band, upper_band, lower_band arrays.
        """
        middle_band = self._compute_sma(close_prices, period)
        rolling_std = np.array([
            np.std(close_prices[max(0, index - period + 1):index + 1])
            for index in range(len(close_prices))
        ])
        upper_band = middle_band + 2.0 * rolling_std
        lower_band = middle_band - 2.0 * rolling_std
        return {
            "middle_band": middle_band.tolist(),
            "upper_band": upper_band.tolist(),
            "lower_band": lower_band.tolist(),
        }

    def _compute_atr(
        self, high_prices: np.ndarray, low_prices: np.ndarray,
        close_prices: np.ndarray, period: int
    ) -> np.ndarray:
        """
        Compute Average True Range.

        Description:
            True range is max of (high-low, |high-prev_close|, |low-prev_close|).

        Params:
            high_prices (np.ndarray): Array of high prices.
            low_prices (np.ndarray): Array of low prices.
            close_prices (np.ndarray): Array of closing prices.
            period (int): ATR lookback period.

        Returns:
            np.ndarray: ATR values array.
        """
        high_low = high_prices[1:] - low_prices[1:]
        high_prev_close = np.abs(high_prices[1:] - close_prices[:-1])
        low_prev_close = np.abs(low_prices[1:] - close_prices[:-1])
        true_range = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))

        atr_values = np.zeros(len(true_range))
        atr_values[period - 1] = np.mean(true_range[:period])
        for index in range(period, len(true_range)):
            atr_values[index] = (atr_values[index - 1] * (period - 1) + true_range[index]) / period
        return atr_values[period - 1:]

    def _compute_vwap(
        self, high_prices: np.ndarray, low_prices: np.ndarray,
        close_prices: np.ndarray, volumes: np.ndarray
    ) -> np.ndarray:
        """
        Compute Volume-Weighted Average Price.

        Description:
            VWAP = cumulative(typical_price * volume) / cumulative(volume).

        Params:
            high_prices (np.ndarray): Array of high prices.
            low_prices (np.ndarray): Array of low prices.
            close_prices (np.ndarray): Array of closing prices.
            volumes (np.ndarray): Array of volume data.

        Returns:
            np.ndarray: VWAP values array.
        """
        typical_price = (high_prices + low_prices + close_prices) / 3.0
        cumulative_tp_volume = np.cumsum(typical_price * volumes)
        cumulative_volume = np.cumsum(volumes)
        vwap_values = np.divide(
            cumulative_tp_volume,
            cumulative_volume,
            out=np.zeros_like(cumulative_tp_volume),
            where=cumulative_volume != 0,
        )
        return vwap_values

    def _compute_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Compute Simple Moving Average.

        Description:
            Rolling mean over the specified period.

        Params:
            data (np.ndarray): Input data array.
            period (int): SMA window size.

        Returns:
            np.ndarray: SMA values array (same length as input, early values use available data).
        """
        sma_values = np.array([
            np.mean(data[max(0, index - period + 1):index + 1])
            for index in range(len(data))
        ])
        return sma_values

    def _compute_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Compute Exponential Moving Average.

        Description:
            Uses smoothing factor 2/(period+1) with initial value as first data point.

        Params:
            data (np.ndarray): Input data array.
            period (int): EMA period.

        Returns:
            np.ndarray: EMA values array.
        """
        multiplier = 2.0 / (period + 1)
        ema_values = np.zeros(len(data))
        ema_values[0] = data[0]
        for index in range(1, len(data)):
            ema_values[index] = (data[index] * multiplier) + (ema_values[index - 1] * (1 - multiplier))
        return ema_values

    def _compute_stochastic(
        self, high_prices: np.ndarray, low_prices: np.ndarray,
        close_prices: np.ndarray, period: int
    ) -> dict:
        """
        Compute Stochastic Oscillator (%K and %D).

        Description:
            %K = (close - lowest_low) / (highest_high - lowest_low) * 100.
            %D = 3-period SMA of %K.

        Params:
            high_prices (np.ndarray): Array of high prices.
            low_prices (np.ndarray): Array of low prices.
            close_prices (np.ndarray): Array of closing prices.
            period (int): Lookback period.

        Returns:
            dict: Dictionary with percent_k and percent_d arrays.
        """
        percent_k = np.zeros(len(close_prices))
        for index in range(period - 1, len(close_prices)):
            window_high = np.max(high_prices[index - period + 1:index + 1])
            window_low = np.min(low_prices[index - period + 1:index + 1])
            denominator = window_high - window_low
            if denominator != 0:
                percent_k[index] = ((close_prices[index] - window_low) / denominator) * 100
            else:
                percent_k[index] = 0.0

        percent_k_valid = percent_k[period - 1:]
        percent_d = self._compute_sma(percent_k_valid, 3)

        return {
            "percent_k": percent_k_valid.tolist(),
            "percent_d": percent_d.tolist(),
        }


class ComputeStatisticTool(BaseTool):
    """
    Computes portfolio and risk statistics on return/price data.

    Description:
        Supports computing Sharpe ratio, maximum drawdown, alpha, beta,
        correlation, volatility, and Sortino ratio from return arrays.

    Attributes:
        SUPPORTED_STATISTICS: Class-level set of supported statistic names.

    Methods:
        name: Returns 'compute_statistic'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for statistic computation.
        execute: Computes the specified statistic on the input data.
    """

    SUPPORTED_STATISTICS = {"sharpe", "drawdown", "alpha", "beta", "correlation", "volatility", "sortino"}

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'compute_statistic'
        """
        return "compute_statistic"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the statistics that can be computed.

        Params:
            None

        Returns:
            str: Description string.
        """
        return (
            "Computes portfolio and risk statistics. "
            "Supported: sharpe, drawdown, alpha, beta, correlation, volatility, sortino."
        )

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines statistic_type, return arrays, and optional benchmark data.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "statistic_type": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_STATISTICS),
                    "description": "The statistic to compute",
                },
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of period returns",
                },
                "benchmark_returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of benchmark returns (for alpha, beta, correlation)",
                },
                "risk_free_rate": {
                    "type": "number",
                    "description": "Annualized risk-free rate",
                    "default": 0.02,
                },
                "periods_per_year": {
                    "type": "integer",
                    "description": "Number of return periods per year",
                    "default": 252,
                },
            },
            "required": ["statistic_type", "returns"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Compute the specified financial statistic.

        Description:
            Dispatches to the appropriate computation based on statistic_type.

        Params:
            **kwargs (Any): Must include statistic_type and returns array.

        Returns:
            dict: Dictionary with the computed statistic value.

        Raises:
            ToolExecutionError: If statistic_type is unsupported or data is insufficient.
        """
        statistic_type = kwargs.get("statistic_type")
        if statistic_type not in self.SUPPORTED_STATISTICS:
            raise ToolExecutionError(
                f"Unsupported statistic_type: '{statistic_type}'. "
                f"Must be one of: {sorted(self.SUPPORTED_STATISTICS)}"
            )

        returns = np.array(kwargs["returns"], dtype=np.float64)
        risk_free_rate = kwargs.get("risk_free_rate", 0.02)
        periods_per_year = kwargs.get("periods_per_year", 252)

        if len(returns) < 2:
            raise ToolExecutionError(
                f"Insufficient data: got {len(returns)} return values but need at least 2"
            )

        if statistic_type == "sharpe":
            excess_returns = returns - (risk_free_rate / periods_per_year)
            sharpe_value = (np.mean(excess_returns) / np.std(excess_returns, ddof=1)) * np.sqrt(periods_per_year)
            return {"sharpe": float(sharpe_value)}

        elif statistic_type == "drawdown":
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            return {
                "max_drawdown": float(np.min(drawdowns)),
                "drawdown_series": drawdowns.tolist(),
            }

        elif statistic_type == "alpha":
            benchmark_returns = self._require_benchmark(kwargs)
            beta_value = self._compute_beta(returns, benchmark_returns)
            annualized_return = np.mean(returns) * periods_per_year
            annualized_benchmark = np.mean(benchmark_returns) * periods_per_year
            alpha_value = annualized_return - (risk_free_rate + beta_value * (annualized_benchmark - risk_free_rate))
            return {"alpha": float(alpha_value)}

        elif statistic_type == "beta":
            benchmark_returns = self._require_benchmark(kwargs)
            beta_value = self._compute_beta(returns, benchmark_returns)
            return {"beta": float(beta_value)}

        elif statistic_type == "correlation":
            benchmark_returns = self._require_benchmark(kwargs)
            correlation_value = float(np.corrcoef(returns, benchmark_returns)[0, 1])
            return {"correlation": correlation_value}

        elif statistic_type == "volatility":
            annualized_volatility = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
            return {"volatility": annualized_volatility}

        elif statistic_type == "sortino":
            excess_returns = returns - (risk_free_rate / periods_per_year)
            downside_returns = np.where(excess_returns < 0, excess_returns, 0.0)
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(periods_per_year)
            if downside_deviation == 0:
                raise ToolExecutionError("Cannot compute Sortino ratio: no downside deviation")
            sortino_value = (np.mean(excess_returns) * periods_per_year) / downside_deviation
            return {"sortino": float(sortino_value)}

        raise ToolExecutionError(f"Unhandled statistic_type: '{statistic_type}'")

    def _require_benchmark(self, kwargs: dict) -> np.ndarray:
        """
        Extract and validate benchmark returns from kwargs.

        Description:
            Retrieves benchmark_returns and ensures it matches the length
            of the primary returns array.

        Params:
            kwargs (dict): The full kwargs dictionary from execute().

        Returns:
            np.ndarray: Validated benchmark returns array.

        Raises:
            ToolExecutionError: If benchmark_returns is missing or length mismatched.
        """
        benchmark_data = kwargs.get("benchmark_returns")
        if benchmark_data is None or len(benchmark_data) == 0:
            raise ToolExecutionError(
                f"statistic_type '{kwargs['statistic_type']}' requires benchmark_returns"
            )
        benchmark_returns = np.array(benchmark_data, dtype=np.float64)
        returns = np.array(kwargs["returns"], dtype=np.float64)
        if len(benchmark_returns) != len(returns):
            raise ToolExecutionError(
                f"returns length ({len(returns)}) must match "
                f"benchmark_returns length ({len(benchmark_returns)})"
            )
        return benchmark_returns

    def _compute_beta(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """
        Compute beta coefficient against benchmark.

        Description:
            Beta = covariance(returns, benchmark) / variance(benchmark).

        Params:
            returns (np.ndarray): Portfolio returns.
            benchmark_returns (np.ndarray): Benchmark returns.

        Returns:
            float: Beta coefficient value.
        """
        covariance_matrix = np.cov(returns, benchmark_returns, ddof=1)
        benchmark_variance = covariance_matrix[1, 1]
        if benchmark_variance == 0:
            raise ToolExecutionError("Cannot compute beta: benchmark has zero variance")
        return float(covariance_matrix[0, 1] / benchmark_variance)


class DetectSignalTool(BaseTool):
    """
    Detects trading signals and patterns in price/indicator data.

    Description:
        Supports detection of threshold crossings, support/resistance levels,
        divergence patterns, pattern breakouts, and trend reversals.

    Attributes:
        SUPPORTED_SIGNALS: Class-level set of supported signal detection types.

    Methods:
        name: Returns 'detect_signal'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for signal detection.
        execute: Detects the specified signal type in the input data.
    """

    SUPPORTED_SIGNALS = {
        "threshold_crossing", "support_resistance", "divergence",
        "pattern_breakout", "trend_reversal"
    }

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'detect_signal'
        """
        return "detect_signal"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the signal types that can be detected.

        Params:
            None

        Returns:
            str: Description string.
        """
        return (
            "Detects trading signals in price/indicator data. "
            "Supported: threshold_crossing, support_resistance, divergence, "
            "pattern_breakout, trend_reversal."
        )

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines signal_type, data arrays, and signal-specific parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "signal_type": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_SIGNALS),
                    "description": "The type of signal to detect",
                },
                "prices": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Price data array",
                },
                "indicator_values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Indicator values array (for divergence)",
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold value for crossing detection",
                },
                "direction": {
                    "type": "string",
                    "enum": ["above", "below", "both"],
                    "description": "Crossing direction to detect",
                    "default": "both",
                },
                "lookback_period": {
                    "type": "integer",
                    "description": "Number of periods to look back",
                    "default": 20,
                },
            },
            "required": ["signal_type", "prices"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Detect the specified signal type in the input data.

        Description:
            Dispatches to the appropriate detection method based on signal_type.

        Params:
            **kwargs (Any): Must include signal_type and prices array.

        Returns:
            dict: Dictionary with detected signals and their positions.

        Raises:
            ToolExecutionError: If signal_type is unsupported or data is insufficient.
        """
        signal_type = kwargs.get("signal_type")
        if signal_type not in self.SUPPORTED_SIGNALS:
            raise ToolExecutionError(
                f"Unsupported signal_type: '{signal_type}'. "
                f"Must be one of: {sorted(self.SUPPORTED_SIGNALS)}"
            )

        prices = np.array(kwargs["prices"], dtype=np.float64)
        if len(prices) < 3:
            raise ToolExecutionError(
                f"Insufficient price data: got {len(prices)} values but need at least 3"
            )

        if signal_type == "threshold_crossing":
            return self._detect_threshold_crossing(prices, kwargs)
        elif signal_type == "support_resistance":
            return self._detect_support_resistance(prices, kwargs)
        elif signal_type == "divergence":
            return self._detect_divergence(prices, kwargs)
        elif signal_type == "pattern_breakout":
            return self._detect_pattern_breakout(prices, kwargs)
        elif signal_type == "trend_reversal":
            return self._detect_trend_reversal(prices, kwargs)

        raise ToolExecutionError(f"Unhandled signal_type: '{signal_type}'")

    def _detect_threshold_crossing(self, prices: np.ndarray, kwargs: dict) -> dict:
        """
        Detect points where price crosses a threshold value.

        Description:
            Identifies indices where price crosses above or below the threshold.

        Params:
            prices (np.ndarray): Price data array.
            kwargs (dict): Must contain 'threshold', optionally 'direction'.

        Returns:
            dict: Dictionary with crossing indices, directions, and prices.

        Raises:
            ToolExecutionError: If threshold is not provided.
        """
        threshold = kwargs.get("threshold")
        if threshold is None:
            raise ToolExecutionError("threshold_crossing requires a 'threshold' parameter")

        direction = kwargs.get("direction", "both")
        crossings = []

        for index in range(1, len(prices)):
            crossed_above = prices[index - 1] < threshold <= prices[index]
            crossed_below = prices[index - 1] > threshold >= prices[index]

            if direction == "above" and crossed_above:
                crossings.append({"index": index, "direction": "above", "price": float(prices[index])})
            elif direction == "below" and crossed_below:
                crossings.append({"index": index, "direction": "below", "price": float(prices[index])})
            elif direction == "both" and (crossed_above or crossed_below):
                crossing_direction = "above" if crossed_above else "below"
                crossings.append({"index": index, "direction": crossing_direction, "price": float(prices[index])})

        return {"signal_type": "threshold_crossing", "threshold": threshold, "crossings": crossings}

    def _detect_support_resistance(self, prices: np.ndarray, kwargs: dict) -> dict:
        """
        Detect support and resistance levels from local extrema.

        Description:
            Identifies local minima as support levels and local maxima as
            resistance levels within the lookback window.

        Params:
            prices (np.ndarray): Price data array.
            kwargs (dict): Optionally 'lookback_period'.

        Returns:
            dict: Dictionary with support_levels and resistance_levels arrays.
        """
        lookback_period = kwargs.get("lookback_period", 20)
        support_levels = []
        resistance_levels = []

        for index in range(1, len(prices) - 1):
            window_start = max(0, index - lookback_period)
            window_end = min(len(prices), index + lookback_period + 1)
            window = prices[window_start:window_end]

            if prices[index] == np.min(window):
                support_levels.append({"index": index, "price": float(prices[index])})
            elif prices[index] == np.max(window):
                resistance_levels.append({"index": index, "price": float(prices[index])})

        return {
            "signal_type": "support_resistance",
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
        }

    def _detect_divergence(self, prices: np.ndarray, kwargs: dict) -> dict:
        """
        Detect divergence between price and an indicator.

        Description:
            Identifies bullish divergence (price lower low, indicator higher low)
            and bearish divergence (price higher high, indicator lower high).

        Params:
            prices (np.ndarray): Price data array.
            kwargs (dict): Must contain 'indicator_values'.

        Returns:
            dict: Dictionary with divergence signals.

        Raises:
            ToolExecutionError: If indicator_values is not provided or length mismatched.
        """
        indicator_data = kwargs.get("indicator_values")
        if indicator_data is None or len(indicator_data) == 0:
            raise ToolExecutionError("divergence detection requires 'indicator_values' parameter")

        indicator_values = np.array(indicator_data, dtype=np.float64)
        if len(indicator_values) != len(prices):
            raise ToolExecutionError(
                f"prices length ({len(prices)}) must match "
                f"indicator_values length ({len(indicator_values)})"
            )

        lookback_period = kwargs.get("lookback_period", 20)
        divergences = []

        for index in range(lookback_period, len(prices)):
            window_prices = prices[index - lookback_period:index + 1]
            window_indicator = indicator_values[index - lookback_period:index + 1]

            price_trend = window_prices[-1] - window_prices[0]
            indicator_trend = window_indicator[-1] - window_indicator[0]

            if price_trend < 0 and indicator_trend > 0:
                divergences.append({
                    "index": index,
                    "type": "bullish",
                    "price": float(prices[index]),
                    "indicator": float(indicator_values[index]),
                })
            elif price_trend > 0 and indicator_trend < 0:
                divergences.append({
                    "index": index,
                    "type": "bearish",
                    "price": float(prices[index]),
                    "indicator": float(indicator_values[index]),
                })

        return {"signal_type": "divergence", "divergences": divergences}

    def _detect_pattern_breakout(self, prices: np.ndarray, kwargs: dict) -> dict:
        """
        Detect breakouts from consolidation ranges.

        Description:
            Identifies when price breaks above or below a lookback period's
            high/low range with significance.

        Params:
            prices (np.ndarray): Price data array.
            kwargs (dict): Optionally 'lookback_period'.

        Returns:
            dict: Dictionary with breakout signals.
        """
        lookback_period = kwargs.get("lookback_period", 20)
        breakouts = []

        for index in range(lookback_period, len(prices)):
            window = prices[index - lookback_period:index]
            window_high = np.max(window)
            window_low = np.min(window)

            if prices[index] > window_high:
                breakouts.append({
                    "index": index,
                    "direction": "bullish",
                    "price": float(prices[index]),
                    "broken_level": float(window_high),
                })
            elif prices[index] < window_low:
                breakouts.append({
                    "index": index,
                    "direction": "bearish",
                    "price": float(prices[index]),
                    "broken_level": float(window_low),
                })

        return {"signal_type": "pattern_breakout", "breakouts": breakouts}

    def _detect_trend_reversal(self, prices: np.ndarray, kwargs: dict) -> dict:
        """
        Detect trend reversals using moving average crossovers.

        Description:
            Uses short-term and long-term SMA crossovers to identify
            potential trend reversals.

        Params:
            prices (np.ndarray): Price data array.
            kwargs (dict): Optionally 'lookback_period'.

        Returns:
            dict: Dictionary with reversal signals.
        """
        lookback_period = kwargs.get("lookback_period", 20)
        short_period = max(lookback_period // 4, 2)
        long_period = lookback_period

        short_sma = np.array([
            np.mean(prices[max(0, index - short_period + 1):index + 1])
            for index in range(len(prices))
        ])
        long_sma = np.array([
            np.mean(prices[max(0, index - long_period + 1):index + 1])
            for index in range(len(prices))
        ])

        reversals = []
        for index in range(long_period, len(prices)):
            bullish_crossover = short_sma[index - 1] < long_sma[index - 1] and short_sma[index] >= long_sma[index]
            bearish_crossover = short_sma[index - 1] > long_sma[index - 1] and short_sma[index] <= long_sma[index]

            if bullish_crossover:
                reversals.append({
                    "index": index,
                    "type": "bullish_reversal",
                    "price": float(prices[index]),
                })
            elif bearish_crossover:
                reversals.append({
                    "index": index,
                    "type": "bearish_reversal",
                    "price": float(prices[index]),
                })

        return {"signal_type": "trend_reversal", "reversals": reversals}
