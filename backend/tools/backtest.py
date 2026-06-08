"""
Backtest tools for stochastic modeling, regime detection, and risk simulation.

What it does:
    Provides four concrete tool implementations for quantitative backtesting:
    Hidden Markov Model regime detection, Markov Chain Monte Carlo parameter
    estimation, stochastic differential equation path simulation, and Monte
    Carlo risk analysis (VaR, CVaR, drawdown distribution).

Entities in it:
    - DetectRegimeTool: Fits a Gaussian HMM to infer market regimes from returns.
    - EstimateParametersTool: Estimates model parameters via Metropolis-Hastings MCMC.
    - SimulateProcessTool: Simulates price paths from stochastic differential equations.
    - RunMonteCarloTool: Runs Monte Carlo simulation for portfolio risk metrics.

How used by other modules:
    - Registered in the ToolRegistry at application startup.
    - Called by agents during workflow execution for backtesting analysis.
    - Receives numeric arrays from upstream data acquisition results.
"""

from typing import Any

import numpy as np

from backend.tools.base import BaseTool, ToolExecutionError


# ===========================================================================
# Shared SDE simulation functions
# ===========================================================================

def _simulate_gbm_paths(
    initial_value: float,
    drift: float,
    volatility: float,
    time_horizon: float,
    time_steps: int,
    path_count: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths using the exact log-normal solution.

    Description:
        Uses the closed-form GBM solution: S(t+dt) = S(t) * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        to generate price paths without discretization error.

    Params:
        initial_value (float): Starting price S(0).
        drift (float): Annualized drift (mu).
        volatility (float): Annualized volatility (sigma).
        time_horizon (float): Total simulation time in years.
        time_steps (int): Number of discrete time steps.
        path_count (int): Number of independent paths to simulate.
        random_generator (np.random.Generator): Seeded random number generator.

    Returns:
        np.ndarray: Array of shape (path_count, time_steps + 1) with simulated prices.
    """
    time_step_size = time_horizon / time_steps
    noise = random_generator.standard_normal((path_count, time_steps))
    log_increments = (
        (drift - 0.5 * volatility ** 2) * time_step_size
        + volatility * np.sqrt(time_step_size) * noise
    )
    log_paths = np.zeros((path_count, time_steps + 1))
    log_paths[:, 0] = np.log(initial_value)
    np.cumsum(log_increments, axis=1, out=log_paths[:, 1:])
    log_paths[:, 1:] += log_paths[:, 0:1]
    return np.exp(log_paths)


def _simulate_ou_paths(
    initial_value: float,
    mean_reversion_speed: float,
    mean_reversion_level: float,
    volatility: float,
    time_horizon: float,
    time_steps: int,
    path_count: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """
    Simulate Ornstein-Uhlenbeck paths using the exact conditional Gaussian transition.

    Description:
        Uses the exact OU conditional distribution:
        X(t+dt) | X(t) ~ N(mu + (X(t) - mu)*exp(-theta*dt), sigma^2/(2*theta)*(1 - exp(-2*theta*dt)))
        which avoids Euler-Maruyama discretization error.

    Params:
        initial_value (float): Starting value X(0).
        mean_reversion_speed (float): Mean-reversion rate (theta), must be positive.
        mean_reversion_level (float): Long-run equilibrium level (mu).
        volatility (float): Diffusion coefficient (sigma).
        time_horizon (float): Total simulation time in years.
        time_steps (int): Number of discrete time steps.
        path_count (int): Number of independent paths to simulate.
        random_generator (np.random.Generator): Seeded random number generator.

    Returns:
        np.ndarray: Array of shape (path_count, time_steps + 1) with simulated values.
    """
    time_step_size = time_horizon / time_steps
    decay_factor = np.exp(-mean_reversion_speed * time_step_size)
    conditional_std = volatility * np.sqrt(
        (1.0 - decay_factor ** 2) / (2.0 * mean_reversion_speed)
    )

    paths = np.zeros((path_count, time_steps + 1))
    paths[:, 0] = initial_value
    noise = random_generator.standard_normal((path_count, time_steps))

    for step in range(time_steps):
        conditional_mean = (
            mean_reversion_level
            + (paths[:, step] - mean_reversion_level) * decay_factor
        )
        paths[:, step + 1] = conditional_mean + conditional_std * noise[:, step]

    return paths


def _simulate_heston_paths(
    initial_value: float,
    drift: float,
    initial_variance: float,
    variance_mean_reversion_speed: float,
    long_run_variance: float,
    volatility_of_variance: float,
    correlation: float,
    time_horizon: float,
    time_steps: int,
    path_count: int,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate Heston stochastic volatility model paths using Euler-Maruyama with truncation.

    Description:
        Discretizes the Heston SDE system:
          dS = mu*S*dt + sqrt(v)*S*dW_1
          dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW_2
          corr(dW_1, dW_2) = rho
        Applies truncation (floor at zero) to keep variance non-negative.

    Params:
        initial_value (float): Starting price S(0).
        drift (float): Price drift (mu).
        initial_variance (float): Starting variance v(0).
        variance_mean_reversion_speed (float): Variance mean-reversion rate (kappa).
        long_run_variance (float): Long-run variance level (theta).
        volatility_of_variance (float): Vol-of-vol (xi).
        correlation (float): Correlation between price and variance Brownian motions (rho).
        time_horizon (float): Total simulation time in years.
        time_steps (int): Number of discrete time steps.
        path_count (int): Number of independent paths.
        random_generator (np.random.Generator): Seeded random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: (price_paths, variance_paths), each of shape
            (path_count, time_steps + 1).
    """
    time_step_size = time_horizon / time_steps
    sqrt_time_step = np.sqrt(time_step_size)

    price_paths = np.zeros((path_count, time_steps + 1))
    variance_paths = np.zeros((path_count, time_steps + 1))
    price_paths[:, 0] = initial_value
    variance_paths[:, 0] = initial_variance

    noise_price = random_generator.standard_normal((path_count, time_steps))
    noise_independent = random_generator.standard_normal((path_count, time_steps))
    noise_variance = (
        correlation * noise_price
        + np.sqrt(1.0 - correlation ** 2) * noise_independent
    )

    for step in range(time_steps):
        current_variance = np.maximum(variance_paths[:, step], 0.0)
        sqrt_variance = np.sqrt(current_variance)

        price_paths[:, step + 1] = price_paths[:, step] * np.exp(
            (drift - 0.5 * current_variance) * time_step_size
            + sqrt_variance * sqrt_time_step * noise_price[:, step]
        )

        variance_paths[:, step + 1] = np.maximum(
            variance_paths[:, step]
            + variance_mean_reversion_speed
            * (long_run_variance - current_variance)
            * time_step_size
            + volatility_of_variance
            * sqrt_variance
            * sqrt_time_step
            * noise_variance[:, step],
            0.0,
        )

    return price_paths, variance_paths


def _simulate_cir_paths(
    initial_value: float,
    mean_reversion_speed: float,
    long_run_rate: float,
    volatility: float,
    time_horizon: float,
    time_steps: int,
    path_count: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """
    Simulate Cox-Ingersoll-Ross model paths using Euler-Maruyama with truncation.

    Description:
        Discretizes the CIR SDE: dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW.
        Applies truncation (floor at zero) to maintain non-negativity of the rate.

    Params:
        initial_value (float): Starting rate r(0).
        mean_reversion_speed (float): Mean-reversion rate (kappa).
        long_run_rate (float): Long-run equilibrium rate (theta).
        volatility (float): Diffusion coefficient (sigma).
        time_horizon (float): Total simulation time in years.
        time_steps (int): Number of discrete time steps.
        path_count (int): Number of independent paths.
        random_generator (np.random.Generator): Seeded random number generator.

    Returns:
        np.ndarray: Array of shape (path_count, time_steps + 1) with simulated rates.
    """
    time_step_size = time_horizon / time_steps
    sqrt_time_step = np.sqrt(time_step_size)

    paths = np.zeros((path_count, time_steps + 1))
    paths[:, 0] = initial_value
    noise = random_generator.standard_normal((path_count, time_steps))

    for step in range(time_steps):
        current = np.maximum(paths[:, step], 0.0)
        drift_component = (
            mean_reversion_speed * (long_run_rate - current) * time_step_size
        )
        diffusion_component = (
            volatility * np.sqrt(current) * sqrt_time_step * noise[:, step]
        )
        paths[:, step + 1] = np.maximum(
            paths[:, step] + drift_component + diffusion_component, 0.0
        )

    return paths


def _compute_terminal_statistics(terminal_values: np.ndarray) -> dict:
    """
    Compute descriptive statistics of terminal path values.

    Description:
        Calculates mean, std, min, max, and key percentiles of the
        distribution of terminal values across all simulated paths.

    Params:
        terminal_values (np.ndarray): 1-D array of terminal values from each path.

    Returns:
        dict: Dictionary with mean, std, min, max, and percentile fields.
    """
    return {
        "mean": float(np.mean(terminal_values)),
        "std": float(np.std(terminal_values, ddof=1)),
        "min": float(np.min(terminal_values)),
        "max": float(np.max(terminal_values)),
        "percentile_5": float(np.percentile(terminal_values, 5)),
        "percentile_25": float(np.percentile(terminal_values, 25)),
        "median": float(np.median(terminal_values)),
        "percentile_75": float(np.percentile(terminal_values, 75)),
        "percentile_95": float(np.percentile(terminal_values, 95)),
    }


def _compute_path_max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """
    Compute maximum drawdown for each simulated path.

    Description:
        For each path, computes the largest peak-to-trough decline as a
        fraction of the peak value. Returns positive values (e.g. 0.15
        means a 15% drawdown from peak).

    Params:
        paths (np.ndarray): Array of shape (path_count, time_steps + 1).

    Returns:
        np.ndarray: 1-D array of per-path maximum drawdown fractions.
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdown_fractions = np.where(
        running_max > 0,
        1.0 - paths / running_max,
        0.0,
    )
    return np.max(drawdown_fractions, axis=1)


# ===========================================================================
# HMM log-space helpers
# ===========================================================================

def _logsumexp(values: np.ndarray) -> float:
    """
    Compute log(sum(exp(values))) with numerical stability.

    Description:
        Subtracts the maximum before exponentiation to prevent overflow/underflow.

    Params:
        values (np.ndarray): Array of log-space values.

    Returns:
        float: The log-sum-exp result.
    """
    max_value = np.max(values)
    if max_value == -np.inf:
        return -np.inf
    return float(max_value + np.log(np.sum(np.exp(values - max_value))))


def _logsumexp_along_axis(
    values: np.ndarray, axis: int, keepdims: bool = False
) -> np.ndarray:
    """
    Compute log-sum-exp along a specific axis with numerical stability.

    Description:
        Generalisation of _logsumexp to operate along an axis of a 2-D array.

    Params:
        values (np.ndarray): 2-D array of log-space values.
        axis (int): Axis along which to reduce.
        keepdims (bool): Whether to keep the reduced dimension.

    Returns:
        np.ndarray: Result array after log-sum-exp reduction.
    """
    max_values = np.max(values, axis=axis, keepdims=True)
    result = max_values + np.log(
        np.sum(np.exp(values - max_values), axis=axis, keepdims=True)
    )
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


# ===========================================================================
# Tool: DetectRegimeTool (Hidden Markov Model)
# ===========================================================================

class DetectRegimeTool(BaseTool):
    """
    Fits a Gaussian Hidden Markov Model to detect market regimes from return data.

    Description:
        Models the market as transitioning between hidden regimes (e.g. bull, bear,
        sideways), each with a distinct Gaussian return distribution. Uses Baum-Welch
        (EM) for parameter estimation and Viterbi for most-likely state decoding.
        All forward-backward computations use log-space arithmetic for numerical
        stability on long sequences.

    Attributes:
        None specific beyond BaseTool.

    Methods:
        name: Returns 'detect_regime'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for regime detection parameters.
        execute: Fits the HMM and returns regime assignments and model parameters.
    """

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'detect_regime'
        """
        return "detect_regime"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the HMM regime detection capability.

        Params:
            None

        Returns:
            str: Description string.
        """
        return (
            "Fits a Gaussian Hidden Markov Model to return data for market "
            "regime detection. Infers hidden states (e.g. bull, bear, sideways) "
            "with transition probabilities and per-regime return distributions."
        )

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines the returns array, regime count, and EM convergence settings.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Array of period returns (e.g. daily log-returns)",
                },
                "regime_count": {
                    "type": "integer",
                    "description": "Number of hidden regimes to detect",
                    "default": 2,
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum Baum-Welch EM iterations",
                    "default": 100,
                },
                "convergence_threshold": {
                    "type": "number",
                    "description": (
                        "Log-likelihood improvement threshold below which EM stops"
                    ),
                    "default": 1e-6,
                },
            },
            "required": ["returns"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Fit the HMM and return regime assignments and model parameters.

        Description:
            Runs Baum-Welch EM to estimate transition probabilities and
            per-regime Gaussian emission parameters, then Viterbi-decodes
            the most likely state sequence.

        Params:
            **kwargs (Any): Must include 'returns'. Optional: regime_count,
                max_iterations, convergence_threshold.

        Returns:
            dict: Dictionary with regime_count, current_regime_probabilities,
                decoded_states, transition_matrix, regime_summaries,
                log_likelihood, and iterations_run.

        Raises:
            ToolExecutionError: If data is insufficient for the requested regime count.
        """
        returns = np.array(kwargs["returns"], dtype=np.float64)
        regime_count = kwargs.get("regime_count", 2)
        max_iterations = kwargs.get("max_iterations", 100)
        convergence_threshold = kwargs.get("convergence_threshold", 1e-6)

        minimum_observations = regime_count * 10
        if len(returns) < minimum_observations:
            raise ToolExecutionError(
                f"Insufficient data: {len(returns)} observations for "
                f"{regime_count} regimes. Need at least {minimum_observations}."
            )

        means, variances, transition_matrix, initial_distribution = (
            self._initialize_parameters(returns, regime_count)
        )

        observation_count = len(returns)
        previous_log_likelihood = -np.inf
        final_gamma = None
        iterations_completed = 0

        for iteration in range(max_iterations):
            log_emission = self._compute_log_emission(
                returns, means, variances
            )
            log_alpha = self._forward_pass(
                log_emission,
                np.log(transition_matrix + 1e-300),
                np.log(initial_distribution + 1e-300),
            )
            log_beta = self._backward_pass(
                log_emission, np.log(transition_matrix + 1e-300)
            )

            log_likelihood = _logsumexp(log_alpha[-1])
            iterations_completed = iteration + 1

            if (
                log_likelihood - previous_log_likelihood < convergence_threshold
                and iteration > 0
            ):
                break
            previous_log_likelihood = log_likelihood

            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp_along_axis(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            final_gamma = gamma

            log_transition = np.log(transition_matrix + 1e-300)
            xi_sum = np.zeros((regime_count, regime_count))
            for time_index in range(observation_count - 1):
                numerator = np.zeros((regime_count, regime_count))
                for state_from in range(regime_count):
                    for state_to in range(regime_count):
                        numerator[state_from, state_to] = (
                            log_alpha[time_index, state_from]
                            + log_transition[state_from, state_to]
                            + log_emission[time_index + 1, state_to]
                            + log_beta[time_index + 1, state_to]
                        )
                normalizer = _logsumexp(numerator.ravel())
                xi_sum += np.exp(numerator - normalizer)

            initial_distribution = gamma[0]

            for state_from in range(regime_count):
                denominator = np.sum(gamma[:-1, state_from])
                for state_to in range(regime_count):
                    transition_matrix[state_from, state_to] = (
                        xi_sum[state_from, state_to] / max(denominator, 1e-10)
                    )

            for regime_index in range(regime_count):
                weight_sum = np.sum(gamma[:, regime_index])
                means[regime_index] = (
                    np.sum(gamma[:, regime_index] * returns)
                    / max(weight_sum, 1e-10)
                )
                squared_deviations = (returns - means[regime_index]) ** 2
                variances[regime_index] = max(
                    np.sum(gamma[:, regime_index] * squared_deviations)
                    / max(weight_sum, 1e-10),
                    1e-10,
                )

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = transition_matrix / np.maximum(row_sums, 1e-10)

        log_emission_final = self._compute_log_emission(returns, means, variances)
        decoded_states = self._viterbi_decode(
            log_emission_final,
            np.log(transition_matrix + 1e-300),
            np.log(initial_distribution + 1e-300),
        )

        if final_gamma is None:
            final_gamma = np.exp(
                log_alpha + log_beta
                - _logsumexp_along_axis(
                    log_alpha + log_beta, axis=1, keepdims=True
                )
            )

        regime_summaries = []
        for regime_index in range(regime_count):
            regime_summaries.append(
                {
                    "regime_index": regime_index,
                    "emission_mean": float(means[regime_index]),
                    "emission_std": float(np.sqrt(variances[regime_index])),
                    "stationary_fraction": float(
                        np.mean(decoded_states == regime_index)
                    ),
                }
            )

        return {
            "regime_count": regime_count,
            "current_regime_probabilities": final_gamma[-1].tolist(),
            "decoded_states": decoded_states.tolist(),
            "transition_matrix": transition_matrix.tolist(),
            "regime_summaries": regime_summaries,
            "log_likelihood": float(log_likelihood),
            "iterations_run": iterations_completed,
        }

    def _initialize_parameters(
        self, returns: np.ndarray, regime_count: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize HMM parameters by sorted-partition of the observed returns.

        Description:
            Sorts returns and partitions them into regime_count equal groups.
            Each partition's sample mean and variance become the initial emission
            parameters. Transition matrix and initial distribution start uniform.

        Params:
            returns (np.ndarray): Observed return values.
            regime_count (int): Number of regimes.

        Returns:
            tuple: (means, variances, transition_matrix, initial_distribution).
        """
        sorted_indices = np.argsort(returns)
        partition_size = len(returns) // regime_count

        means = np.zeros(regime_count)
        variances = np.zeros(regime_count)
        for partition_index in range(regime_count):
            start = partition_index * partition_size
            end = (
                start + partition_size
                if partition_index < regime_count - 1
                else len(returns)
            )
            partition = returns[sorted_indices[start:end]]
            means[partition_index] = np.mean(partition)
            variances[partition_index] = max(np.var(partition), 1e-10)

        transition_matrix = np.full(
            (regime_count, regime_count), 1.0 / regime_count
        )
        initial_distribution = np.full(regime_count, 1.0 / regime_count)
        return means, variances, transition_matrix, initial_distribution

    def _compute_log_emission(
        self,
        observations: np.ndarray,
        means: np.ndarray,
        variances: np.ndarray,
    ) -> np.ndarray:
        """
        Compute log-probability of each observation under each Gaussian regime.

        Description:
            Evaluates the Gaussian log-PDF for every (observation, regime) pair.

        Params:
            observations (np.ndarray): 1-D array of observed values.
            means (np.ndarray): Per-regime emission means.
            variances (np.ndarray): Per-regime emission variances.

        Returns:
            np.ndarray: Shape (observation_count, regime_count) of log-probabilities.
        """
        regime_count = len(means)
        observation_count = len(observations)
        log_emission = np.zeros((observation_count, regime_count))
        for regime_index in range(regime_count):
            diff = observations - means[regime_index]
            log_emission[:, regime_index] = (
                -0.5 * np.log(2.0 * np.pi * variances[regime_index])
                - 0.5 * diff ** 2 / variances[regime_index]
            )
        return log_emission

    def _forward_pass(
        self,
        log_emission: np.ndarray,
        log_transition: np.ndarray,
        log_initial: np.ndarray,
    ) -> np.ndarray:
        """
        Forward algorithm in log space.

        Description:
            Computes log-alpha values: log P(o_1..o_t, state_t = j).

        Params:
            log_emission (np.ndarray): Log emission probabilities.
            log_transition (np.ndarray): Log transition matrix.
            log_initial (np.ndarray): Log initial state distribution.

        Returns:
            np.ndarray: Shape (observation_count, regime_count) log-alpha values.
        """
        observation_count, regime_count = log_emission.shape
        log_alpha = np.full((observation_count, regime_count), -np.inf)
        log_alpha[0] = log_initial + log_emission[0]

        for time_index in range(1, observation_count):
            for state_index in range(regime_count):
                log_alpha[time_index, state_index] = (
                    _logsumexp(
                        log_alpha[time_index - 1]
                        + log_transition[:, state_index]
                    )
                    + log_emission[time_index, state_index]
                )
        return log_alpha

    def _backward_pass(
        self, log_emission: np.ndarray, log_transition: np.ndarray
    ) -> np.ndarray:
        """
        Backward algorithm in log space.

        Description:
            Computes log-beta values: log P(o_{t+1}..o_T | state_t = j).

        Params:
            log_emission (np.ndarray): Log emission probabilities.
            log_transition (np.ndarray): Log transition matrix.

        Returns:
            np.ndarray: Shape (observation_count, regime_count) log-beta values.
        """
        observation_count, regime_count = log_emission.shape
        log_beta = np.full((observation_count, regime_count), -np.inf)
        log_beta[-1] = 0.0

        for time_index in range(observation_count - 2, -1, -1):
            for state_index in range(regime_count):
                log_beta[time_index, state_index] = _logsumexp(
                    log_transition[state_index, :]
                    + log_emission[time_index + 1]
                    + log_beta[time_index + 1]
                )
        return log_beta

    def _viterbi_decode(
        self,
        log_emission: np.ndarray,
        log_transition: np.ndarray,
        log_initial: np.ndarray,
    ) -> np.ndarray:
        """
        Viterbi decoding for the most likely state sequence.

        Description:
            Finds the single most probable sequence of hidden states given the
            observations, using dynamic programming in log space.

        Params:
            log_emission (np.ndarray): Log emission probabilities.
            log_transition (np.ndarray): Log transition matrix.
            log_initial (np.ndarray): Log initial state distribution.

        Returns:
            np.ndarray: 1-D integer array of decoded state indices.
        """
        observation_count, regime_count = log_emission.shape
        log_delta = np.full((observation_count, regime_count), -np.inf)
        backpointer = np.zeros((observation_count, regime_count), dtype=int)

        log_delta[0] = log_initial + log_emission[0]

        for time_index in range(1, observation_count):
            for state_index in range(regime_count):
                candidates = (
                    log_delta[time_index - 1]
                    + log_transition[:, state_index]
                )
                backpointer[time_index, state_index] = int(np.argmax(candidates))
                log_delta[time_index, state_index] = (
                    candidates[backpointer[time_index, state_index]]
                    + log_emission[time_index, state_index]
                )

        states = np.zeros(observation_count, dtype=int)
        states[-1] = int(np.argmax(log_delta[-1]))
        for time_index in range(observation_count - 2, -1, -1):
            states[time_index] = backpointer[
                time_index + 1, states[time_index + 1]
            ]
        return states


# ===========================================================================
# Tool: EstimateParametersTool (MCMC)
# ===========================================================================

class EstimateParametersTool(BaseTool):
    """
    Estimates model parameters via Metropolis-Hastings MCMC sampling.

    Description:
        Implements a random-walk Metropolis-Hastings sampler for Bayesian
        estimation of financial model parameters. Supports fitting GBM
        (drift + volatility from log-returns) and Ornstein-Uhlenbeck
        (mean-reversion speed, level, and volatility from observed values).
        Proposal scales are adapted from data statistics.

    Attributes:
        SUPPORTED_MODELS: Set of supported model type strings.

    Methods:
        name: Returns 'estimate_parameters'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for MCMC estimation parameters.
        execute: Runs the MCMC chain and returns posterior summaries.
    """

    SUPPORTED_MODELS = {"gbm_volatility", "ou_parameters"}

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'estimate_parameters'
        """
        return "estimate_parameters"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the MCMC parameter estimation capability.

        Params:
            None

        Returns:
            str: Description string.
        """
        return (
            "Estimates financial model parameters via Metropolis-Hastings MCMC. "
            "Supported models: gbm_volatility (drift and volatility from returns), "
            "ou_parameters (mean-reversion speed, level, and volatility from values)."
        )

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines the observed data, model type, chain settings, and time step.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "observed_data": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Array of observed values (returns for GBM, "
                        "sequential values for OU)"
                    ),
                },
                "model_type": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_MODELS),
                    "description": "The model whose parameters to estimate",
                },
                "chain_length": {
                    "type": "integer",
                    "description": "Total MCMC chain length (including burn-in)",
                    "default": 5000,
                },
                "burn_in_fraction": {
                    "type": "number",
                    "description": (
                        "Fraction of chain to discard as burn-in (0 to 1)"
                    ),
                    "default": 0.2,
                },
                "time_step_size": {
                    "type": "number",
                    "description": (
                        "Time between observations in years "
                        "(e.g. 1/252 for daily data)"
                    ),
                    "default": 0.003968,
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility",
                },
            },
            "required": ["observed_data", "model_type"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Run the MCMC chain and return posterior parameter summaries.

        Description:
            Dispatches to the model-specific MCMC implementation based on
            model_type, runs the chain, discards burn-in, and computes
            posterior mean, std, median, and 95% credible interval.

        Params:
            **kwargs (Any): Must include observed_data and model_type.

        Returns:
            dict: Dictionary with model_type, acceptance_rate, parameter_estimates,
                chain_length, burn_in, and effective_samples.

        Raises:
            ToolExecutionError: If model_type is unsupported or data is insufficient.
        """
        model_type = kwargs.get("model_type")
        if model_type not in self.SUPPORTED_MODELS:
            raise ToolExecutionError(
                f"Unsupported model_type: '{model_type}'. "
                f"Must be one of: {sorted(self.SUPPORTED_MODELS)}"
            )

        observed_data = np.array(kwargs["observed_data"], dtype=np.float64)
        chain_length = kwargs.get("chain_length", 5000)
        burn_in_fraction = kwargs.get("burn_in_fraction", 0.2)
        time_step_size = kwargs.get("time_step_size", 1.0 / 252.0)
        random_seed = kwargs.get("random_seed")

        if len(observed_data) < 20:
            raise ToolExecutionError(
                f"Insufficient data: {len(observed_data)} observations, "
                f"need at least 20 for reliable MCMC estimation."
            )

        random_generator = np.random.default_rng(random_seed)

        if model_type == "gbm_volatility":
            return self._estimate_gbm_volatility(
                observed_data, chain_length, burn_in_fraction,
                time_step_size, random_generator,
            )
        elif model_type == "ou_parameters":
            return self._estimate_ou_parameters(
                observed_data, chain_length, burn_in_fraction,
                time_step_size, random_generator,
            )

        raise ToolExecutionError(f"Unhandled model_type: '{model_type}'")

    def _estimate_gbm_volatility(
        self,
        returns: np.ndarray,
        chain_length: int,
        burn_in_fraction: float,
        time_step_size: float,
        random_generator: np.random.Generator,
    ) -> dict:
        """
        Estimate GBM drift and volatility from log-returns via Metropolis-Hastings.

        Description:
            Likelihood: returns ~ N((mu - sigma^2/2)*dt, sigma^2*dt).
            Priors: drift ~ N(0, 1), volatility ~ HalfNormal(scale=1).
            Proposal: independent Gaussian perturbation with data-adaptive scale.

        Params:
            returns (np.ndarray): Array of log-returns.
            chain_length (int): Total MCMC iterations.
            burn_in_fraction (float): Fraction to discard.
            time_step_size (float): Time between observations in years.
            random_generator (np.random.Generator): Seeded RNG.

        Returns:
            dict: MCMC result with parameter posterior summaries.
        """
        observation_count = len(returns)

        current_drift = float(np.mean(returns) / time_step_size)
        current_volatility = float(
            np.std(returns, ddof=1) / np.sqrt(time_step_size)
        )
        current_volatility = max(current_volatility, 1e-6)

        drift_proposal_std = abs(current_drift) * 0.1 + 0.01
        volatility_proposal_std = current_volatility * 0.05

        def log_posterior(drift: float, vol: float) -> float:
            if vol <= 0:
                return -np.inf
            adjusted_mean = (drift - vol ** 2 / 2.0) * time_step_size
            variance = vol ** 2 * time_step_size
            log_likelihood = (
                -observation_count / 2.0 * np.log(2.0 * np.pi * variance)
                - np.sum((returns - adjusted_mean) ** 2) / (2.0 * variance)
            )
            log_prior_drift = -0.5 * drift ** 2
            log_prior_vol = -0.5 * vol ** 2
            return log_likelihood + log_prior_drift + log_prior_vol

        chain_drift = np.zeros(chain_length)
        chain_volatility = np.zeros(chain_length)
        accepted_count = 0

        current_log_posterior = log_posterior(current_drift, current_volatility)

        for step in range(chain_length):
            proposed_drift = current_drift + random_generator.normal(
                0, drift_proposal_std
            )
            proposed_volatility = current_volatility + random_generator.normal(
                0, volatility_proposal_std
            )

            proposed_log_posterior = log_posterior(
                proposed_drift, proposed_volatility
            )
            log_acceptance = proposed_log_posterior - current_log_posterior

            if np.log(random_generator.uniform()) < log_acceptance:
                current_drift = proposed_drift
                current_volatility = proposed_volatility
                current_log_posterior = proposed_log_posterior
                accepted_count += 1

            chain_drift[step] = current_drift
            chain_volatility[step] = current_volatility

        burn_in = int(chain_length * burn_in_fraction)
        posterior_drift = chain_drift[burn_in:]
        posterior_volatility = chain_volatility[burn_in:]

        return {
            "model_type": "gbm_volatility",
            "acceptance_rate": float(accepted_count / chain_length),
            "parameter_estimates": {
                "drift": self._summarize_posterior(posterior_drift),
                "volatility": self._summarize_posterior(posterior_volatility),
            },
            "chain_length": chain_length,
            "burn_in": burn_in,
            "effective_samples": len(posterior_drift),
        }

    def _estimate_ou_parameters(
        self,
        observed_values: np.ndarray,
        chain_length: int,
        burn_in_fraction: float,
        time_step_size: float,
        random_generator: np.random.Generator,
    ) -> dict:
        """
        Estimate OU process parameters from sequential observations via MH-MCMC.

        Description:
            Likelihood uses the exact OU conditional distribution:
            X(t+dt)|X(t) ~ N(mu + (X(t)-mu)*exp(-theta*dt),
                              sigma^2/(2*theta)*(1-exp(-2*theta*dt))).
            Priors: theta ~ Exp(rate=0.5), mu ~ N(data_mean, 10*data_std),
            sigma ~ HalfNormal(scale=data_based).

        Params:
            observed_values (np.ndarray): Sequential observations.
            chain_length (int): Total MCMC iterations.
            burn_in_fraction (float): Fraction to discard.
            time_step_size (float): Time between observations in years.
            random_generator (np.random.Generator): Seeded RNG.

        Returns:
            dict: MCMC result with parameter posterior summaries.
        """
        data_mean = float(np.mean(observed_values))
        data_std = float(np.std(observed_values, ddof=1))
        data_std = max(data_std, 1e-6)
        increments = np.diff(observed_values)
        increment_std = float(np.std(increments, ddof=1))
        increment_std = max(increment_std, 1e-6)

        autocorrelation_lag1 = float(
            np.corrcoef(observed_values[:-1], observed_values[1:])[0, 1]
        )
        if 0.0 < autocorrelation_lag1 < 1.0:
            initial_speed = -np.log(autocorrelation_lag1) / time_step_size
        else:
            initial_speed = 1.0
        initial_level = data_mean
        initial_volatility = increment_std / np.sqrt(time_step_size)

        speed_proposal_std = max(initial_speed * 0.05, 0.01)
        level_proposal_std = data_std * 0.05
        volatility_proposal_std = initial_volatility * 0.05

        values_current = observed_values[:-1]
        values_next = observed_values[1:]

        def log_posterior(
            speed: float, level: float, vol: float
        ) -> float:
            if speed <= 0 or vol <= 0:
                return -np.inf

            decay = np.exp(-speed * time_step_size)
            conditional_mean = level + (values_current - level) * decay
            conditional_var = (
                vol ** 2 / (2.0 * speed) * (1.0 - decay ** 2)
            )
            if conditional_var <= 0:
                return -np.inf

            log_likelihood = np.sum(
                -0.5 * np.log(2.0 * np.pi * conditional_var)
                - 0.5 * (values_next - conditional_mean) ** 2 / conditional_var
            )

            log_prior_speed = -0.5 * speed
            log_prior_level = (
                -0.5 * ((level - data_mean) / (10.0 * data_std)) ** 2
            )
            log_prior_vol = -0.5 * (vol / initial_volatility) ** 2
            return log_likelihood + log_prior_speed + log_prior_level + log_prior_vol

        chain_speed = np.zeros(chain_length)
        chain_level = np.zeros(chain_length)
        chain_volatility = np.zeros(chain_length)
        accepted_count = 0

        current_speed = initial_speed
        current_level = initial_level
        current_vol = initial_volatility
        current_log_posterior = log_posterior(
            current_speed, current_level, current_vol
        )

        for step in range(chain_length):
            proposed_speed = current_speed + random_generator.normal(
                0, speed_proposal_std
            )
            proposed_level = current_level + random_generator.normal(
                0, level_proposal_std
            )
            proposed_vol = current_vol + random_generator.normal(
                0, volatility_proposal_std
            )

            proposed_log_posterior = log_posterior(
                proposed_speed, proposed_level, proposed_vol
            )
            log_acceptance = proposed_log_posterior - current_log_posterior

            if np.log(random_generator.uniform()) < log_acceptance:
                current_speed = proposed_speed
                current_level = proposed_level
                current_vol = proposed_vol
                current_log_posterior = proposed_log_posterior
                accepted_count += 1

            chain_speed[step] = current_speed
            chain_level[step] = current_level
            chain_volatility[step] = current_vol

        burn_in = int(chain_length * burn_in_fraction)
        posterior_speed = chain_speed[burn_in:]
        posterior_level = chain_level[burn_in:]
        posterior_volatility = chain_volatility[burn_in:]

        return {
            "model_type": "ou_parameters",
            "acceptance_rate": float(accepted_count / chain_length),
            "parameter_estimates": {
                "mean_reversion_speed": self._summarize_posterior(
                    posterior_speed
                ),
                "mean_reversion_level": self._summarize_posterior(
                    posterior_level
                ),
                "volatility": self._summarize_posterior(posterior_volatility),
            },
            "chain_length": chain_length,
            "burn_in": burn_in,
            "effective_samples": len(posterior_speed),
        }

    @staticmethod
    def _summarize_posterior(samples: np.ndarray) -> dict:
        """
        Compute summary statistics of a posterior sample array.

        Description:
            Returns mean, std, median, and 95% credible interval bounds.

        Params:
            samples (np.ndarray): 1-D array of posterior samples.

        Returns:
            dict: Summary with mean, std, median, percentile_2_5, percentile_97_5.
        """
        return {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples, ddof=1)),
            "median": float(np.median(samples)),
            "percentile_2_5": float(np.percentile(samples, 2.5)),
            "percentile_97_5": float(np.percentile(samples, 97.5)),
        }


# ===========================================================================
# Tool: SimulateProcessTool (SDEs)
# ===========================================================================

class SimulateProcessTool(BaseTool):
    """
    Simulates price/rate paths from stochastic differential equations.

    Description:
        Supports four SDE models: Geometric Brownian Motion (GBM),
        Ornstein-Uhlenbeck (OU) mean-reverting process, Heston stochastic
        volatility model, and Cox-Ingersoll-Ross (CIR) interest rate model.
        Returns simulated paths and terminal value statistics.

    Attributes:
        SUPPORTED_PROCESSES: Set of supported SDE process type strings.

    Methods:
        name: Returns 'simulate_process'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for SDE simulation parameters.
        execute: Simulates paths and returns path data with terminal statistics.
    """

    SUPPORTED_PROCESSES = {"gbm", "ornstein_uhlenbeck", "heston", "cir"}

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'simulate_process'
        """
        return "simulate_process"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the SDE simulation capability.

        Params:
            None

        Returns:
            str: Description string.
        """
        return (
            "Simulates price/rate paths from stochastic differential equations. "
            "Supported: gbm (Geometric Brownian Motion), ornstein_uhlenbeck "
            "(mean-reverting), heston (stochastic volatility), cir (interest rates)."
        )

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines the process type, initial conditions, time grid settings,
            and process-specific parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "process_type": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_PROCESSES),
                    "description": "The SDE model to simulate",
                },
                "initial_value": {
                    "type": "number",
                    "description": "Starting value S(0) or r(0)",
                },
                "time_horizon": {
                    "type": "number",
                    "description": "Total simulation time in years",
                    "default": 1.0,
                },
                "time_steps": {
                    "type": "integer",
                    "description": "Number of discrete time steps",
                    "default": 252,
                },
                "path_count": {
                    "type": "integer",
                    "description": "Number of independent paths to simulate",
                    "default": 100,
                },
                "drift": {
                    "type": "number",
                    "description": "Annualized drift (for GBM, Heston)",
                    "default": 0.05,
                },
                "volatility": {
                    "type": "number",
                    "description": "Annualized volatility (for GBM, OU, CIR)",
                    "default": 0.2,
                },
                "mean_reversion_speed": {
                    "type": "number",
                    "description": "Mean-reversion rate (for OU, CIR)",
                    "default": 1.0,
                },
                "mean_reversion_level": {
                    "type": "number",
                    "description": "Long-run equilibrium level (for OU)",
                },
                "long_run_rate": {
                    "type": "number",
                    "description": "Long-run equilibrium rate (for CIR)",
                },
                "initial_variance": {
                    "type": "number",
                    "description": "Starting variance v(0) (for Heston)",
                    "default": 0.04,
                },
                "variance_mean_reversion_speed": {
                    "type": "number",
                    "description": "Variance mean-reversion rate (for Heston)",
                    "default": 2.0,
                },
                "long_run_variance": {
                    "type": "number",
                    "description": "Long-run variance level (for Heston)",
                    "default": 0.04,
                },
                "volatility_of_variance": {
                    "type": "number",
                    "description": "Vol-of-vol (for Heston)",
                    "default": 0.3,
                },
                "correlation": {
                    "type": "number",
                    "description": (
                        "Correlation between price and variance noise (for Heston)"
                    ),
                    "default": -0.7,
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility",
                },
            },
            "required": ["process_type", "initial_value"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Simulate SDE paths and return path data with terminal statistics.

        Description:
            Dispatches to the appropriate simulation function based on
            process_type, generates paths, and packages the result with
            terminal distribution statistics and a time grid.

        Params:
            **kwargs (Any): Must include process_type and initial_value.

        Returns:
            dict: Dictionary with process_type, path_count, time_steps,
                time_horizon, terminal_statistics, paths, and time_grid.

        Raises:
            ToolExecutionError: If process_type is unsupported or parameters invalid.
        """
        process_type = kwargs.get("process_type")
        if process_type not in self.SUPPORTED_PROCESSES:
            raise ToolExecutionError(
                f"Unsupported process_type: '{process_type}'. "
                f"Must be one of: {sorted(self.SUPPORTED_PROCESSES)}"
            )

        initial_value = kwargs["initial_value"]
        time_horizon = kwargs.get("time_horizon", 1.0)
        time_steps = kwargs.get("time_steps", 252)
        path_count = kwargs.get("path_count", 100)
        random_seed = kwargs.get("random_seed")
        random_generator = np.random.default_rng(random_seed)

        if process_type == "gbm":
            if initial_value <= 0:
                raise ToolExecutionError(
                    "GBM requires positive initial_value"
                )
            paths = _simulate_gbm_paths(
                initial_value=initial_value,
                drift=kwargs.get("drift", 0.05),
                volatility=kwargs.get("volatility", 0.2),
                time_horizon=time_horizon,
                time_steps=time_steps,
                path_count=path_count,
                random_generator=random_generator,
            )
        elif process_type == "ornstein_uhlenbeck":
            mean_reversion_level = kwargs.get("mean_reversion_level")
            if mean_reversion_level is None:
                raise ToolExecutionError(
                    "ornstein_uhlenbeck requires mean_reversion_level"
                )
            paths = _simulate_ou_paths(
                initial_value=initial_value,
                mean_reversion_speed=kwargs.get("mean_reversion_speed", 1.0),
                mean_reversion_level=mean_reversion_level,
                volatility=kwargs.get("volatility", 0.2),
                time_horizon=time_horizon,
                time_steps=time_steps,
                path_count=path_count,
                random_generator=random_generator,
            )
        elif process_type == "heston":
            if initial_value <= 0:
                raise ToolExecutionError(
                    "Heston requires positive initial_value"
                )
            initial_variance = kwargs.get("initial_variance", 0.04)
            if initial_variance < 0:
                raise ToolExecutionError(
                    "Heston requires non-negative initial_variance"
                )
            price_paths, variance_paths = _simulate_heston_paths(
                initial_value=initial_value,
                drift=kwargs.get("drift", 0.05),
                initial_variance=initial_variance,
                variance_mean_reversion_speed=kwargs.get(
                    "variance_mean_reversion_speed", 2.0
                ),
                long_run_variance=kwargs.get("long_run_variance", 0.04),
                volatility_of_variance=kwargs.get(
                    "volatility_of_variance", 0.3
                ),
                correlation=kwargs.get("correlation", -0.7),
                time_horizon=time_horizon,
                time_steps=time_steps,
                path_count=path_count,
                random_generator=random_generator,
            )
            paths = price_paths
            terminal_variance = _compute_terminal_statistics(
                variance_paths[:, -1]
            )
        elif process_type == "cir":
            if initial_value < 0:
                raise ToolExecutionError(
                    "CIR requires non-negative initial_value"
                )
            long_run_rate = kwargs.get("long_run_rate")
            if long_run_rate is None:
                raise ToolExecutionError("CIR requires long_run_rate")
            paths = _simulate_cir_paths(
                initial_value=initial_value,
                mean_reversion_speed=kwargs.get("mean_reversion_speed", 1.0),
                long_run_rate=long_run_rate,
                volatility=kwargs.get("volatility", 0.2),
                time_horizon=time_horizon,
                time_steps=time_steps,
                path_count=path_count,
                random_generator=random_generator,
            )
        else:
            raise ToolExecutionError(
                f"Unhandled process_type: '{process_type}'"
            )

        terminal_values = paths[:, -1]
        time_grid = np.linspace(0, time_horizon, time_steps + 1)

        result = {
            "process_type": process_type,
            "path_count": path_count,
            "time_steps": time_steps,
            "time_horizon": time_horizon,
            "terminal_statistics": _compute_terminal_statistics(
                terminal_values
            ),
            "paths": paths.tolist(),
            "time_grid": time_grid.tolist(),
        }

        if process_type == "heston":
            result["terminal_variance_statistics"] = terminal_variance

        return result


# ===========================================================================
# Tool: RunMonteCarloTool (Monte Carlo Risk Analysis)
# ===========================================================================

class RunMonteCarloTool(BaseTool):
    """
    Runs Monte Carlo simulation to compute portfolio risk metrics.

    Description:
        Simulates many price paths using the specified SDE process, then
        computes risk metrics from the terminal distribution: Value at Risk
        (VaR), Conditional Value at Risk (CVaR / Expected Shortfall),
        probability of loss, and maximum drawdown distribution. VaR and CVaR
        are expressed as positive loss fractions relative to initial value.

    Attributes:
        SUPPORTED_PROCESSES: Set of supported SDE process type strings.

    Methods:
        name: Returns 'run_monte_carlo'.
        description: Returns the tool's purpose description.
        parameters_schema: Returns JSON schema for Monte Carlo parameters.
        execute: Runs simulation and returns risk metrics.
    """

    SUPPORTED_PROCESSES = {"gbm", "ornstein_uhlenbeck", "heston", "cir"}

    @property
    def name(self) -> str:
        """
        Unique name for this tool.

        Description:
            Returns the canonical name used for registry lookup.

        Params:
            None

        Returns:
            str: 'run_monte_carlo'
        """
        return "run_monte_carlo"

    @property
    def description(self) -> str:
        """
        Human-readable description of this tool.

        Description:
            Explains the Monte Carlo risk analysis capability.

        Params:
            None

        Returns:
            str: Description string.
        """
        return (
            "Runs Monte Carlo simulation for portfolio risk analysis. "
            "Computes VaR, CVaR (Expected Shortfall), probability of loss, "
            "and max drawdown distribution across thousands of simulated paths."
        )

    @property
    def parameters_schema(self) -> dict:
        """
        JSON Schema for the execute() parameters.

        Description:
            Defines the process type, initial conditions, simulation count,
            confidence levels for VaR/CVaR, and process-specific parameters.

        Params:
            None

        Returns:
            dict: JSON Schema dictionary.
        """
        return {
            "type": "object",
            "properties": {
                "process_type": {
                    "type": "string",
                    "enum": list(self.SUPPORTED_PROCESSES),
                    "description": "The SDE model for simulation",
                },
                "initial_value": {
                    "type": "number",
                    "description": "Starting portfolio/price value",
                },
                "time_horizon": {
                    "type": "number",
                    "description": "Risk horizon in years",
                    "default": 1.0,
                },
                "time_steps": {
                    "type": "integer",
                    "description": "Number of discrete time steps per path",
                    "default": 252,
                },
                "simulation_count": {
                    "type": "integer",
                    "description": "Number of Monte Carlo paths",
                    "default": 10000,
                },
                "confidence_levels": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": (
                        "Confidence levels for VaR/CVaR (e.g. [0.95, 0.99])"
                    ),
                    "default": [0.95, 0.99],
                },
                "drift": {
                    "type": "number",
                    "description": "Annualized drift (for GBM, Heston)",
                    "default": 0.05,
                },
                "volatility": {
                    "type": "number",
                    "description": "Annualized volatility (for GBM, OU, CIR)",
                    "default": 0.2,
                },
                "mean_reversion_speed": {
                    "type": "number",
                    "description": "Mean-reversion rate (for OU, CIR)",
                    "default": 1.0,
                },
                "mean_reversion_level": {
                    "type": "number",
                    "description": "Long-run equilibrium level (for OU)",
                },
                "long_run_rate": {
                    "type": "number",
                    "description": "Long-run equilibrium rate (for CIR)",
                },
                "initial_variance": {
                    "type": "number",
                    "description": "Starting variance v(0) (for Heston)",
                    "default": 0.04,
                },
                "variance_mean_reversion_speed": {
                    "type": "number",
                    "description": "Variance mean-reversion rate (for Heston)",
                    "default": 2.0,
                },
                "long_run_variance": {
                    "type": "number",
                    "description": "Long-run variance level (for Heston)",
                    "default": 0.04,
                },
                "volatility_of_variance": {
                    "type": "number",
                    "description": "Vol-of-vol (for Heston)",
                    "default": 0.3,
                },
                "correlation": {
                    "type": "number",
                    "description": (
                        "Correlation between price and variance noise "
                        "(for Heston)"
                    ),
                    "default": -0.7,
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility",
                },
            },
            "required": ["process_type", "initial_value"],
        }

    async def execute(self, **kwargs: Any) -> dict:
        """
        Run Monte Carlo simulation and return risk metrics.

        Description:
            Simulates many paths using the specified SDE, then computes
            VaR, CVaR, probability of loss, and max drawdown distribution
            from the simulated terminal values and path trajectories.

        Params:
            **kwargs (Any): Must include process_type and initial_value.

        Returns:
            dict: Dictionary with process_type, simulation_count, initial_value,
                time_horizon, terminal_distribution, value_at_risk,
                conditional_value_at_risk, probability_of_loss, and
                max_drawdown_statistics.

        Raises:
            ToolExecutionError: If process_type is unsupported or parameters invalid.
        """
        process_type = kwargs.get("process_type")
        if process_type not in self.SUPPORTED_PROCESSES:
            raise ToolExecutionError(
                f"Unsupported process_type: '{process_type}'. "
                f"Must be one of: {sorted(self.SUPPORTED_PROCESSES)}"
            )

        initial_value = kwargs["initial_value"]
        time_horizon = kwargs.get("time_horizon", 1.0)
        time_steps = kwargs.get("time_steps", 252)
        simulation_count = kwargs.get("simulation_count", 10000)
        confidence_levels = kwargs.get("confidence_levels", [0.95, 0.99])
        random_seed = kwargs.get("random_seed")
        random_generator = np.random.default_rng(random_seed)

        paths = self._generate_paths(
            process_type, initial_value, time_horizon, time_steps,
            simulation_count, random_generator, kwargs,
        )

        terminal_values = paths[:, -1]
        returns = (terminal_values - initial_value) / initial_value

        terminal_distribution = {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns, ddof=1)),
            "skewness": float(self._compute_skewness(returns)),
            "kurtosis": float(self._compute_kurtosis(returns)),
            "mean_terminal_value": float(np.mean(terminal_values)),
            "std_terminal_value": float(np.std(terminal_values, ddof=1)),
        }

        value_at_risk = {}
        conditional_value_at_risk = {}
        for confidence_level in confidence_levels:
            quantile_rank = (1.0 - confidence_level) * 100.0
            return_quantile = np.percentile(returns, quantile_rank)
            var_loss = -return_quantile
            value_at_risk[str(confidence_level)] = float(var_loss)

            tail_returns = returns[returns <= return_quantile]
            if len(tail_returns) > 0:
                cvar_loss = float(-np.mean(tail_returns))
            else:
                cvar_loss = float(var_loss)
            conditional_value_at_risk[str(confidence_level)] = cvar_loss

        probability_of_loss = float(np.mean(returns < 0))

        max_drawdowns = _compute_path_max_drawdowns(paths)
        max_drawdown_statistics = {
            "mean": float(np.mean(max_drawdowns)),
            "std": float(np.std(max_drawdowns, ddof=1)),
            "median": float(np.median(max_drawdowns)),
            "percentile_5": float(np.percentile(max_drawdowns, 5)),
            "percentile_95": float(np.percentile(max_drawdowns, 95)),
        }

        return {
            "process_type": process_type,
            "simulation_count": simulation_count,
            "initial_value": initial_value,
            "time_horizon": time_horizon,
            "terminal_distribution": terminal_distribution,
            "value_at_risk": value_at_risk,
            "conditional_value_at_risk": conditional_value_at_risk,
            "probability_of_loss": probability_of_loss,
            "max_drawdown_statistics": max_drawdown_statistics,
        }

    def _generate_paths(
        self,
        process_type: str,
        initial_value: float,
        time_horizon: float,
        time_steps: int,
        simulation_count: int,
        random_generator: np.random.Generator,
        kwargs: dict,
    ) -> np.ndarray:
        """
        Dispatch to the appropriate SDE simulation function.

        Description:
            Routes to the correct module-level simulation function based on
            process_type, extracting the relevant parameters from kwargs.

        Params:
            process_type (str): The SDE model identifier.
            initial_value (float): Starting value.
            time_horizon (float): Simulation time in years.
            time_steps (int): Number of time steps.
            simulation_count (int): Number of paths.
            random_generator (np.random.Generator): Seeded RNG.
            kwargs (dict): Full kwargs dict containing process-specific parameters.

        Returns:
            np.ndarray: Array of shape (simulation_count, time_steps + 1).

        Raises:
            ToolExecutionError: If required parameters are missing.
        """
        if process_type == "gbm":
            if initial_value <= 0:
                raise ToolExecutionError(
                    "GBM requires positive initial_value"
                )
            return _simulate_gbm_paths(
                initial_value, kwargs.get("drift", 0.05),
                kwargs.get("volatility", 0.2),
                time_horizon, time_steps, simulation_count, random_generator,
            )
        elif process_type == "ornstein_uhlenbeck":
            mean_reversion_level = kwargs.get("mean_reversion_level")
            if mean_reversion_level is None:
                raise ToolExecutionError(
                    "ornstein_uhlenbeck requires mean_reversion_level"
                )
            return _simulate_ou_paths(
                initial_value, kwargs.get("mean_reversion_speed", 1.0),
                mean_reversion_level, kwargs.get("volatility", 0.2),
                time_horizon, time_steps, simulation_count, random_generator,
            )
        elif process_type == "heston":
            if initial_value <= 0:
                raise ToolExecutionError(
                    "Heston requires positive initial_value"
                )
            price_paths, _ = _simulate_heston_paths(
                initial_value, kwargs.get("drift", 0.05),
                kwargs.get("initial_variance", 0.04),
                kwargs.get("variance_mean_reversion_speed", 2.0),
                kwargs.get("long_run_variance", 0.04),
                kwargs.get("volatility_of_variance", 0.3),
                kwargs.get("correlation", -0.7),
                time_horizon, time_steps, simulation_count, random_generator,
            )
            return price_paths
        elif process_type == "cir":
            if initial_value < 0:
                raise ToolExecutionError(
                    "CIR requires non-negative initial_value"
                )
            long_run_rate = kwargs.get("long_run_rate")
            if long_run_rate is None:
                raise ToolExecutionError("CIR requires long_run_rate")
            return _simulate_cir_paths(
                initial_value, kwargs.get("mean_reversion_speed", 1.0),
                long_run_rate, kwargs.get("volatility", 0.2),
                time_horizon, time_steps, simulation_count, random_generator,
            )

        raise ToolExecutionError(
            f"Unhandled process_type: '{process_type}'"
        )

    @staticmethod
    def _compute_skewness(values: np.ndarray) -> float:
        """
        Compute sample skewness of a 1-D array.

        Description:
            Uses the adjusted Fisher-Pearson formula for sample skewness.

        Params:
            values (np.ndarray): 1-D array of values.

        Returns:
            float: Sample skewness.
        """
        count = len(values)
        if count < 3:
            return 0.0
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std == 0:
            return 0.0
        return float(
            count / ((count - 1) * (count - 2))
            * np.sum(((values - mean) / std) ** 3)
        )

    @staticmethod
    def _compute_kurtosis(values: np.ndarray) -> float:
        """
        Compute sample excess kurtosis of a 1-D array.

        Description:
            Returns excess kurtosis (normal distribution = 0).

        Params:
            values (np.ndarray): 1-D array of values.

        Returns:
            float: Sample excess kurtosis.
        """
        count = len(values)
        if count < 4:
            return 0.0
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std == 0:
            return 0.0
        raw_kurtosis = float(np.mean(((values - mean) / std) ** 4))
        return raw_kurtosis - 3.0
