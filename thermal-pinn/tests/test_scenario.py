import numpy as np
import pytest

from thermalpinn.scenario import SHOWCASE, Scenario, sample_scenarios


def test_flux_is_a_ramp_that_ends_with_the_braking():
    s = Scenario(abs_amp=0.0, dip_depth=0.0)
    assert s.q_true(0.0) == pytest.approx(s.q_peak)
    assert s.q_true(s.t_brake) == pytest.approx(0.0)
    assert s.q_true(s.t_brake + 1.0) == pytest.approx(0.0)
    t = np.linspace(0.0, s.t_brake, 200)
    assert np.all(np.diff(s.q_true(t)) < 0.0)


def test_the_pedal_release_is_a_real_dip_of_the_advertised_width():
    s = SHOWCASE
    baseline = Scenario(dip_depth=0.0, abs_amp=0.0)
    assert s.q_true(s.dip_t) < 0.75 * baseline.q_true(s.dip_t)
    # and it is compactly supported: nothing survives beyond half a width either side
    off = s.dip_t + 0.51 * s.dip_width
    assert s.q_dip(off) == pytest.approx(0.0)
    assert s.q_dip(s.dip_t - 0.51 * s.dip_width) == pytest.approx(0.0)


def test_the_abs_ripple_is_present_at_its_frequency():
    s = Scenario(dip_depth=0.0)
    t = np.arange(0.0, s.t_brake, 1e-3)
    ripple = s.q_true(t) - s.q_ramp(t)
    spectrum = np.abs(np.fft.rfft(ripple * np.hanning(t.size)))
    peak = np.fft.rfftfreq(t.size, 1e-3)[np.argmax(spectrum)]
    assert peak == pytest.approx(s.abs_freq, abs=0.5)


def test_flux_never_pumps_heat_back_into_the_pad():
    t = np.linspace(0.0, SHOWCASE.t_end, 20001)
    assert SHOWCASE.q_true(t).min() >= 0.0


def test_sensor_times_are_the_advertised_rate_over_the_whole_event():
    ts = SHOWCASE.sensor_times()
    assert ts[0] == 0.0 and ts[-1] == pytest.approx(SHOWCASE.t_end)
    assert np.allclose(np.diff(ts), 1.0 / SHOWCASE.f_sample)
    assert ts.size == 2001  # 100 Hz over 20 s, the PINN's data term


def test_sampling_is_reproducible_and_varied():
    a = sample_scenarios(np.random.default_rng(0), 24)
    b = sample_scenarios(np.random.default_rng(0), 24)
    assert [s.q_peak for s in a] == [s.q_peak for s in b]
    assert len({s.noise_seed for s in a}) == 24
    assert len({s.q_peak for s in a}) == 24
    for s in a:
        assert 0.0 < s.dip_t < s.t_brake  # the release always falls inside the braking
        assert s.t_brake < s.t_end
        assert s.q_true(np.linspace(0.0, s.t_end, 2001)).max() > 0.0
