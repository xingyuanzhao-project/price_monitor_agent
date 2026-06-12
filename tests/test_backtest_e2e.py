"""E2E: Backtest -- all 4 tools with synthetic data and mathematical validation."""

import asyncio
import json
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.tools.backtest import (
    DetectRegimeTool,
    EstimateParametersTool,
    SimulateProcessTool,
    RunMonteCarloTool,
)

detect_regime_tool = DetectRegimeTool()
estimate_parameters_tool = EstimateParametersTool()
simulate_process_tool = SimulateProcessTool()
run_monte_carlo_tool = RunMonteCarloTool()


async def main():
    print("\n### Backtest -- all 4 tools with mathematical validation ###\n")

    # ==================================================================
    # 1/7  HMM Regime Detection on 2-regime synthetic data
    # ==================================================================
    print("=" * 60)
    print("detect_regime / HMM / 2-regime synthetic returns")
    print("=" * 60)

    random_generator = np.random.default_rng(42)
    true_transition = np.array([[0.95, 0.05], [0.10, 0.90]])
    true_means = [0.001, -0.002]
    true_stds = [0.008, 0.025]
    observation_count = 1000

    true_states = np.zeros(observation_count, dtype=int)
    current_state = 0
    for time_index in range(1, observation_count):
        if random_generator.uniform() < true_transition[current_state, 1 - current_state]:
            current_state = 1 - current_state
        true_states[time_index] = current_state

    synthetic_returns = np.array([
        random_generator.normal(true_means[state], true_stds[state])
        for state in true_states
    ])

    print(f"Input: {observation_count} returns from 2-regime model")
    print(f"  True regime 0: mean={true_means[0]}, std={true_stds[0]}")
    print(f"  True regime 1: mean={true_means[1]}, std={true_stds[1]}")
    print(f"  True transition: {true_transition.tolist()}")
    print(f"  Regime 0 fraction: {np.mean(true_states == 0):.3f}")
    print(f"  Regime 1 fraction: {np.mean(true_states == 1):.3f}")

    result = await detect_regime_tool.execute(
        returns=synthetic_returns.tolist(),
        regime_count=2,
    )

    print(f"\nResult:")
    print(f"  Iterations: {result['iterations_run']}")
    print(f"  Log-likelihood: {result['log_likelihood']:.2f}")
    for regime in result["regime_summaries"]:
        print(
            f"  Regime {regime['regime_index']}: "
            f"mean={regime['emission_mean']:.6f}, "
            f"std={regime['emission_std']:.6f}, "
            f"fraction={regime['stationary_fraction']:.3f}"
        )
    print(f"  Transition matrix: {json.dumps(result['transition_matrix'], indent=4)}")
    print(f"  Current regime probs: {result['current_regime_probabilities']}")

    sorted_regimes = sorted(result["regime_summaries"], key=lambda regime: regime["emission_mean"])
    mean_separation = sorted_regimes[1]["emission_mean"] - sorted_regimes[0]["emission_mean"]
    std_separation = sorted_regimes[0]["emission_std"] - sorted_regimes[1]["emission_std"]
    assert mean_separation > 0.001, (
        f"Regimes not well-separated by mean: separation={mean_separation:.6f}"
    )
    assert result["log_likelihood"] > -np.inf, "Log-likelihood is -inf (model did not converge)"
    assert np.isfinite(result["log_likelihood"]), "Log-likelihood is not finite"

    decoded_states = np.array(result["decoded_states"])
    regime_0_fraction = np.mean(decoded_states == 0)
    regime_1_fraction = np.mean(decoded_states == 1)
    assert regime_0_fraction > 0.1 and regime_1_fraction > 0.1, (
        f"One regime is degenerate: fractions={regime_0_fraction:.3f}, {regime_1_fraction:.3f}"
    )
    print("PASS\n")

    # ==================================================================
    # 2/7  MCMC: GBM volatility estimation
    # ==================================================================
    print("=" * 60)
    print("estimate_parameters / MCMC / GBM volatility")
    print("=" * 60)

    true_drift = 0.08
    true_volatility = 0.20
    time_step_size = 1.0 / 252.0
    random_generator_gbm = np.random.default_rng(123)
    gbm_returns = (
        (true_drift - 0.5 * true_volatility ** 2) * time_step_size
        + true_volatility * np.sqrt(time_step_size) * random_generator_gbm.standard_normal(500)
    )

    print(f"Input: 500 synthetic GBM returns")
    print(f"  True drift: {true_drift}")
    print(f"  True volatility: {true_volatility}")
    print(f"  Sample mean return: {np.mean(gbm_returns):.8f}")
    print(f"  Sample std return: {np.std(gbm_returns):.8f}")

    result = await estimate_parameters_tool.execute(
        observed_data=gbm_returns.tolist(),
        model_type="gbm_volatility",
        chain_length=10000,
        burn_in_fraction=0.3,
        time_step_size=time_step_size,
        random_seed=42,
    )

    print(f"\nResult:")
    print(f"  Acceptance rate: {result['acceptance_rate']:.4f}")
    print(f"  Effective samples: {result['effective_samples']}")
    drift_est = result["parameter_estimates"]["drift"]
    vol_est = result["parameter_estimates"]["volatility"]
    print(f"  Drift posterior:      mean={drift_est['mean']:.4f}, "
          f"std={drift_est['std']:.4f}, "
          f"95% CI=[{drift_est['percentile_2_5']:.4f}, {drift_est['percentile_97_5']:.4f}]")
    print(f"  Volatility posterior: mean={vol_est['mean']:.4f}, "
          f"std={vol_est['std']:.4f}, "
          f"95% CI=[{vol_est['percentile_2_5']:.4f}, {vol_est['percentile_97_5']:.4f}]")

    vol_relative_error = abs(vol_est["mean"] - true_volatility) / true_volatility
    print(f"  Volatility relative error: {vol_relative_error:.4f}")
    assert vol_relative_error < 0.30, (
        f"Posterior volatility mean {vol_est['mean']:.4f} is too far from "
        f"true {true_volatility} (error={vol_relative_error:.2%})"
    )
    assert 0.05 < result["acceptance_rate"] < 0.95, (
        f"Acceptance rate {result['acceptance_rate']:.4f} is outside healthy range"
    )
    print("PASS\n")

    # ==================================================================
    # 3/7  MCMC: OU parameter estimation
    # ==================================================================
    print("=" * 60)
    print("estimate_parameters / MCMC / OU parameters")
    print("=" * 60)

    true_ou_speed = 5.0
    true_ou_level = 50.0
    true_ou_vol = 3.0
    ou_dt = 1.0 / 252.0
    random_generator_ou = np.random.default_rng(456)

    ou_values = np.zeros(600)
    ou_values[0] = 48.0
    decay = np.exp(-true_ou_speed * ou_dt)
    cond_std = true_ou_vol * np.sqrt((1.0 - decay ** 2) / (2.0 * true_ou_speed))
    for step_index in range(1, 600):
        cond_mean = true_ou_level + (ou_values[step_index - 1] - true_ou_level) * decay
        ou_values[step_index] = cond_mean + cond_std * random_generator_ou.standard_normal()

    print(f"Input: 600 synthetic OU observations")
    print(f"  True speed: {true_ou_speed}, level: {true_ou_level}, vol: {true_ou_vol}")
    print(f"  Data mean: {np.mean(ou_values):.4f}, std: {np.std(ou_values):.4f}")

    result = await estimate_parameters_tool.execute(
        observed_data=ou_values.tolist(),
        model_type="ou_parameters",
        chain_length=10000,
        burn_in_fraction=0.3,
        time_step_size=ou_dt,
        random_seed=789,
    )

    print(f"\nResult:")
    print(f"  Acceptance rate: {result['acceptance_rate']:.4f}")
    speed_est = result["parameter_estimates"]["mean_reversion_speed"]
    level_est = result["parameter_estimates"]["mean_reversion_level"]
    vol_est_ou = result["parameter_estimates"]["volatility"]
    print(f"  Speed posterior:  mean={speed_est['mean']:.4f}, "
          f"95% CI=[{speed_est['percentile_2_5']:.4f}, {speed_est['percentile_97_5']:.4f}]")
    print(f"  Level posterior:  mean={level_est['mean']:.4f}, "
          f"95% CI=[{level_est['percentile_2_5']:.4f}, {level_est['percentile_97_5']:.4f}]")
    print(f"  Vol posterior:    mean={vol_est_ou['mean']:.4f}, "
          f"95% CI=[{vol_est_ou['percentile_2_5']:.4f}, {vol_est_ou['percentile_97_5']:.4f}]")

    level_error = abs(level_est["mean"] - true_ou_level) / true_ou_level
    print(f"  Level relative error: {level_error:.4f}")
    assert level_error < 0.15, (
        f"OU level estimate {level_est['mean']:.4f} too far from true {true_ou_level}"
    )
    assert 0.05 < result["acceptance_rate"] < 0.95, (
        f"OU MCMC acceptance rate {result['acceptance_rate']:.4f} outside healthy range"
    )
    print("PASS\n")

    # ==================================================================
    # 4/7  SDE: GBM path simulation
    # ==================================================================
    print("=" * 60)
    print("simulate_process / GBM / 1000 paths")
    print("=" * 60)

    result = await simulate_process_tool.execute(
        process_type="gbm",
        initial_value=100.0,
        drift=0.05,
        volatility=0.20,
        time_horizon=1.0,
        time_steps=252,
        path_count=1000,
        random_seed=42,
    )

    terminal_stats = result["terminal_statistics"]
    expected_mean_terminal = 100.0 * np.exp(0.05 * 1.0)
    mean_relative_error = abs(terminal_stats["mean"] - expected_mean_terminal) / expected_mean_terminal

    print(f"Result:")
    print(f"  Path count: {result['path_count']}, steps: {result['time_steps']}")
    print(f"  E[S(T)] theoretical: {expected_mean_terminal:.4f}")
    print(f"  E[S(T)] simulated:   {terminal_stats['mean']:.4f}")
    print(f"  Std[S(T)]:           {terminal_stats['std']:.4f}")
    print(f"  Min:                 {terminal_stats['min']:.4f}")
    print(f"  Max:                 {terminal_stats['max']:.4f}")
    print(f"  5th percentile:      {terminal_stats['percentile_5']:.4f}")
    print(f"  95th percentile:     {terminal_stats['percentile_95']:.4f}")
    print(f"  Mean relative error: {mean_relative_error:.4f}")

    assert mean_relative_error < 0.05, (
        f"GBM mean terminal {terminal_stats['mean']:.2f} too far from "
        f"expected {expected_mean_terminal:.2f} (error={mean_relative_error:.2%})"
    )
    assert terminal_stats["min"] > 0, "GBM produced non-positive prices"
    assert len(result["paths"]) == 1000, f"Expected 1000 paths, got {len(result['paths'])}"
    assert len(result["paths"][0]) == 253, f"Expected 253 points per path, got {len(result['paths'][0])}"
    print("PASS\n")

    # ==================================================================
    # 5/7  SDE: OU mean-reversion verification
    # ==================================================================
    print("=" * 60)
    print("simulate_process / Ornstein-Uhlenbeck / mean-reversion check")
    print("=" * 60)

    result = await simulate_process_tool.execute(
        process_type="ornstein_uhlenbeck",
        initial_value=80.0,
        mean_reversion_speed=5.0,
        mean_reversion_level=100.0,
        volatility=3.0,
        time_horizon=2.0,
        time_steps=504,
        path_count=500,
        random_seed=99,
    )

    terminal_stats = result["terminal_statistics"]
    mean_reversion_level = 100.0
    level_convergence_error = abs(terminal_stats["mean"] - mean_reversion_level) / mean_reversion_level

    print(f"Result (initial=80, target level=100, speed=5, T=2yr):")
    print(f"  E[X(T)] simulated: {terminal_stats['mean']:.4f}")
    print(f"  Std[X(T)]:         {terminal_stats['std']:.4f}")
    print(f"  Convergence error: {level_convergence_error:.4f}")

    assert level_convergence_error < 0.05, (
        f"OU terminal mean {terminal_stats['mean']:.2f} did not converge "
        f"to level {mean_reversion_level} (error={level_convergence_error:.2%})"
    )
    print("PASS\n")

    # ==================================================================
    # 6/7  SDE: Heston + CIR positivity checks
    # ==================================================================
    print("=" * 60)
    print("simulate_process / Heston + CIR / positivity verification")
    print("=" * 60)

    heston_result = await simulate_process_tool.execute(
        process_type="heston",
        initial_value=100.0,
        drift=0.05,
        initial_variance=0.04,
        variance_mean_reversion_speed=2.0,
        long_run_variance=0.04,
        volatility_of_variance=0.3,
        correlation=-0.7,
        time_horizon=1.0,
        time_steps=252,
        path_count=200,
        random_seed=55,
    )

    heston_paths = np.array(heston_result["paths"])
    heston_terminal = heston_result["terminal_statistics"]
    heston_all_positive = np.all(heston_paths > 0)

    print(f"Heston (S0=100, v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7):")
    print(f"  All prices positive: {heston_all_positive}")
    print(f"  Terminal mean: {heston_terminal['mean']:.4f}")
    print(f"  Terminal std:  {heston_terminal['std']:.4f}")
    print(f"  Terminal variance stats: {json.dumps(heston_result['terminal_variance_statistics'], indent=4)}")

    assert heston_all_positive, "Heston produced non-positive prices"
    assert heston_terminal["mean"] > 50, "Heston terminal mean unreasonably low"

    cir_result = await simulate_process_tool.execute(
        process_type="cir",
        initial_value=0.05,
        mean_reversion_speed=1.0,
        long_run_rate=0.04,
        volatility=0.1,
        time_horizon=5.0,
        time_steps=1260,
        path_count=200,
        random_seed=77,
    )

    cir_paths = np.array(cir_result["paths"])
    cir_terminal = cir_result["terminal_statistics"]
    cir_all_nonneg = np.all(cir_paths >= 0)
    cir_convergence = abs(cir_terminal["mean"] - 0.04) / 0.04

    print(f"\nCIR (r0=0.05, kappa=1, theta=0.04, sigma=0.1, T=5yr):")
    print(f"  All rates non-negative: {cir_all_nonneg}")
    print(f"  Terminal mean: {cir_terminal['mean']:.6f}")
    print(f"  Terminal std:  {cir_terminal['std']:.6f}")
    print(f"  Long-run convergence error: {cir_convergence:.4f}")

    assert cir_all_nonneg, "CIR produced negative rates"
    assert cir_convergence < 0.20, (
        f"CIR terminal mean {cir_terminal['mean']:.6f} did not converge "
        f"to long-run rate 0.04"
    )
    print("PASS\n")

    # ==================================================================
    # 7/7  Monte Carlo: VaR / CVaR / drawdown risk metrics
    # ==================================================================
    print("=" * 60)
    print("run_monte_carlo / GBM / VaR + CVaR + drawdown")
    print("=" * 60)

    result = await run_monte_carlo_tool.execute(
        process_type="gbm",
        initial_value=100.0,
        drift=0.05,
        volatility=0.25,
        time_horizon=1.0,
        time_steps=252,
        simulation_count=10000,
        confidence_levels=[0.90, 0.95, 0.99],
        random_seed=42,
    )

    print(f"Result (S0=100, mu=0.05, sigma=0.25, T=1yr, N=10000):")
    terminal_dist = result["terminal_distribution"]
    print(f"  Mean return:       {terminal_dist['mean_return']:.4f}")
    print(f"  Std return:        {terminal_dist['std_return']:.4f}")
    print(f"  Skewness:          {terminal_dist['skewness']:.4f}")
    print(f"  Excess kurtosis:   {terminal_dist['kurtosis']:.4f}")
    print(f"  Mean terminal:     {terminal_dist['mean_terminal_value']:.2f}")
    print(f"  Probability of loss: {result['probability_of_loss']:.4f}")

    print(f"\n  Value at Risk (positive = loss fraction):")
    for level, var_value in result["value_at_risk"].items():
        print(f"    VaR({level}): {var_value:.4f}")
    print(f"  Conditional VaR (Expected Shortfall):")
    for level, cvar_value in result["conditional_value_at_risk"].items():
        print(f"    CVaR({level}): {cvar_value:.4f}")

    drawdown = result["max_drawdown_statistics"]
    print(f"\n  Max drawdown distribution:")
    print(f"    Mean:     {drawdown['mean']:.4f}")
    print(f"    Std:      {drawdown['std']:.4f}")
    print(f"    Median:   {drawdown['median']:.4f}")
    print(f"    95th pct: {drawdown['percentile_95']:.4f}")

    var_90 = result["value_at_risk"]["0.9"]
    var_95 = result["value_at_risk"]["0.95"]
    var_99 = result["value_at_risk"]["0.99"]
    cvar_95 = result["conditional_value_at_risk"]["0.95"]
    cvar_99 = result["conditional_value_at_risk"]["0.99"]

    assert var_99 >= var_95 >= var_90, (
        f"VaR ordering violated: VaR(99)={var_99:.4f}, "
        f"VaR(95)={var_95:.4f}, VaR(90)={var_90:.4f}"
    )
    assert cvar_95 >= var_95, (
        f"CVaR(95)={cvar_95:.4f} should be >= VaR(95)={var_95:.4f}"
    )
    assert cvar_99 >= var_99, (
        f"CVaR(99)={cvar_99:.4f} should be >= VaR(99)={var_99:.4f}"
    )
    assert 0.0 < result["probability_of_loss"] < 1.0, (
        f"Probability of loss {result['probability_of_loss']:.4f} is degenerate"
    )
    assert drawdown["mean"] > 0, "Mean max drawdown should be positive"
    assert drawdown["percentile_95"] > drawdown["median"], (
        "95th pct drawdown should exceed median"
    )
    print("PASS\n")

    print("ALL BACKTEST TESTS PASSED (7 tests: HMM + 2×MCMC + 3×SDE + MC)")


asyncio.run(main())
