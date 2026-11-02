import numpy as np
import pytest

from vsm.surface.ssvi import (
    GAMMA_MAX,
    SSVI,
    SliceData,
    butterfly_margins,
    fit_ssvi,
    phi_power,
)

TRUE = SSVI(
    rho=-0.3,
    eta=1.1,
    gamma=0.4,
    t_nodes=(0.05, 0.1, 0.25, 0.5, 1.0),
    theta_nodes=(0.012, 0.024, 0.06, 0.13, 0.27),
)


def data_from(surface: SSVI) -> list[SliceData]:
    slices = []
    for t in surface.t_nodes:
        k = np.linspace(-0.6, 0.6, 25) * np.sqrt(float(surface.theta(np.array(t)))) * 3
        slices.append(SliceData(t=t, k=k, w=surface.w(k, np.full(k.size, t)), weight=np.ones(25)))
    return slices


def test_theta_at_nodes_and_between() -> None:
    assert TRUE.theta(np.array(TRUE.t_nodes)) == pytest.approx(np.array(TRUE.theta_nodes))
    assert float(TRUE.theta(np.array(0.075))) == pytest.approx(0.018)


def test_theta_extrapolates_at_constant_atm_vol() -> None:
    assert float(TRUE.theta(np.array(0.01))) == pytest.approx(0.012 * 0.01 / 0.05)
    assert float(TRUE.theta(np.array(2.0))) == pytest.approx(0.27 * 2.0)


def test_atm_total_variance_is_theta() -> None:
    assert TRUE.w(np.zeros(5), np.array(TRUE.t_nodes)) == pytest.approx(
        np.array(TRUE.theta_nodes), rel=1e-12
    )


def test_slice_is_exactly_raw_svi() -> None:
    k = np.linspace(-1.5, 1.5, 41)
    for t in (0.07, 0.3, 1.0):
        slice_ = TRUE.slice_at(t)
        assert slice_.w(k) == pytest.approx(TRUE.w(k, np.full(k.size, t)), rel=1e-12)


def test_phi_is_the_power_law() -> None:
    theta = 0.09
    assert float(phi_power(theta, 1.1, 0.4)) == pytest.approx(1.1 * theta**-0.4 * 1.09**-0.6)


def test_calendar_condition_holds_by_construction() -> None:
    theta = np.logspace(-6, 2, 400)
    rho = 0.95
    gamma = 0.05
    phi = phi_power(theta, 1.0, gamma)
    derivative = np.gradient(theta * phi, theta)
    assert np.all(derivative[1:-1] >= 0.0)
    limit = (1.0 + np.sqrt(1.0 - rho * rho)) / (rho * rho) * phi
    assert np.all(derivative[1:-1] <= limit[1:-1])


def test_margins_detect_a_violation() -> None:
    assert min(butterfly_margins(-0.3, 1.0, 0.4)) > 0.0
    assert butterfly_margins(-0.3, 6.0, 0.4)[0] < 0.0


def test_fit_recovers_the_surface() -> None:
    fit = fit_ssvi(data_from(TRUE))
    assert fit.converged
    assert fit.ssvi.rho == pytest.approx(TRUE.rho, abs=1e-3)
    assert fit.ssvi.eta == pytest.approx(TRUE.eta, rel=2e-3)
    assert fit.ssvi.gamma == pytest.approx(TRUE.gamma, abs=2e-3)
    assert fit.ssvi.theta_nodes == pytest.approx(TRUE.theta_nodes, rel=1e-3)


def test_fit_keeps_theta_non_decreasing_and_conditions_true() -> None:
    noisy = [
        SliceData(s.t, s.k, s.w * (1.05 if i == 2 else 1.0), s.weight)
        for i, s in enumerate(data_from(TRUE))
    ]
    fit = fit_ssvi(noisy)
    assert np.all(np.diff(fit.ssvi.theta_nodes) >= 0.0)
    assert min(butterfly_margins(fit.ssvi.rho, fit.ssvi.eta, fit.ssvi.gamma)) >= 0.0


def test_gamma_stays_at_or_below_one_half() -> None:
    steep = SSVI(-0.2, 1.0, 0.8, TRUE.t_nodes, TRUE.theta_nodes)
    assert fit_ssvi(data_from(steep)).ssvi.gamma <= GAMMA_MAX + 1e-12


def test_prior_pulls_the_shape_parameters() -> None:
    prior = SSVI(0.2, 0.5, 0.2, TRUE.t_nodes, TRUE.theta_nodes)
    free = fit_ssvi(data_from(TRUE))
    held = fit_ssvi(data_from(TRUE), prior=prior, lam=1.0)
    assert abs(held.ssvi.rho - prior.rho) < abs(free.ssvi.rho - prior.rho)
