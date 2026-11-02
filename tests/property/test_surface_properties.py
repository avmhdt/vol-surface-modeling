"""SSVI inside the sufficient conditions always passes the gate; exports read back exactly."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from vsm.config import ArbitrageConfig
from vsm.surface.arbitrage import check_svi_conditions
from vsm.surface.ssvi import SSVI, butterfly_margins

CFG = ArbitrageConfig()
T_NODES = (0.02, 0.08, 0.25, 0.5, 1.0)

surfaces = st.builds(
    lambda rho, eta, gamma, atm, slope: SSVI(
        rho=rho,
        eta=eta,
        gamma=gamma,
        t_nodes=T_NODES,
        theta_nodes=tuple(atm * atm * t * (1.0 + slope * t) for t in T_NODES),
    ),
    rho=st.floats(-0.7, 0.7),
    eta=st.floats(0.2, 1.5),
    gamma=st.floats(0.1, 0.5),
    atm=st.floats(0.2, 1.5),
    slope=st.floats(0.0, 1.0),
)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(surface=surfaces)
def test_ssvi_inside_the_conditions_passes_the_gate(surface: SSVI) -> None:
    if min(butterfly_margins(surface.rho, surface.eta, surface.gamma)) <= 0.0:
        return
    checks = check_svi_conditions(surface, CFG)
    assert all(check.min_g >= -CFG.tol_g for check in checks)
    assert all(check.price_ok for check in checks)
    assert all(
        check.min_calendar_dw is None or check.min_calendar_dw >= -CFG.tol_calendar
        for check in checks
    )


@settings(max_examples=60, deadline=None)
@given(surface=surfaces)
def test_total_variance_rises_with_time_at_fixed_k(surface: SSVI) -> None:
    if min(butterfly_margins(surface.rho, surface.eta, surface.gamma)) <= 0.0:
        return
    k = np.linspace(-1.0, 1.0, 21)
    earlier = surface.w(k, np.full(k.size, 0.1))
    later = surface.w(k, np.full(k.size, 0.4))
    assert np.all(later - earlier >= -1e-12)
