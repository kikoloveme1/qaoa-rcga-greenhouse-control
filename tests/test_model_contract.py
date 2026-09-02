import numpy as np
import pytest
from environment.greenhouse_model import GreenhouseConfig, GreenhouseEnv


def test_paper_response_equations_and_bounds():
    env = GreenhouseEnv()
    low, high = env.bounds()
    np.testing.assert_equal(low[:4], [15, 0, 300, 30])
    np.testing.assert_equal(high[:4], [38, 800, 1800, 95])
    assert env.temperature_response(30) == pytest.approx(np.exp(-0.5))
    assert env.light_response(120) == pytest.approx(0.5)
    assert env.co2_response(1000) == pytest.approx(0.5)
    assert env.co2_response(400) == pytest.approx(400/1400)


def test_negative_light_is_penalized_and_profit_is_not_penalized_twice():
    env = GreenhouseEnv()
    plan = np.tile([25., -1., 600., 70.], (24, 1))
    score, details = env.fitness(plan)
    assert details['total_penalty'] > 0
    assert details['total_penalty'] == pytest.approx(24*500/800**2)
    assert score == pytest.approx(details['profit_sgd'] - details['total_penalty'])
    assert details['profit_sgd'] == pytest.approx(
        details['revenue_sgd'] - details['total_energy']
        - details['total_water'] * env.config.water_cost_per_m3 * .001
        - env.config.fixed_cost_per_day)


def test_hourly_co2_balance_uses_ppm_injection_and_fractional_loss():
    env = GreenhouseEnv()
    old, injection, rate, lai = 900., 400., .2, 2.5
    expected = old + injection - env.ventilation_rate(25., 27.) * old - env.crop_uptake_co2(rate, lai)
    assert env.co2_balance(injection, old, 25., 27., rate, lai) == pytest.approx(expected)


def test_humidity_is_tracked_and_water_balance_is_auditable():
    env = GreenhouseEnv()
    _, d = env.fitness(np.tile([25., 100., 600., 70.], (24, 1)))
    previous = np.r_[env.config.RH_out, d['rh_realized'][:-1]]
    expected = np.clip(previous + env.config.humidity_tracking * (70 - previous)
                       + d['humidity_disturbance_hourly'], 0, 100)
    np.testing.assert_allclose(d['rh_realized'], expected)
    water_previous = np.r_[env.config.SWC_initial, d['soil_water_content_hourly'][:-1]]
    np.testing.assert_allclose(d['soil_water_content_hourly'], water_previous
                              + d['irrigation_swc_hourly'] - d['evapotranspiration_swc_hourly'])


def test_shape_and_nonfinite_controls_rejected():
    env = GreenhouseEnv()
    with pytest.raises(ValueError):
        env.fitness(np.full((24, 4), np.nan))
