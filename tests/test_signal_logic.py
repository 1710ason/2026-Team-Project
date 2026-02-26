import numpy as np

from src.dsp_utils import SignalProcessor
from src.physics import Magnetics
from src.config import HardwareConstants
from src.analyzer import TransformerCoreAnalyzer


def test_cut_to_integer_cycles_keeps_final_crossing_boundary():
    theta = np.linspace(0.0, 6.0 * np.pi, 1201)
    h = -np.sin(theta)
    b = np.cos(theta)
    t = np.linspace(0.0, 0.06, len(theta))

    h_cut, b_cut, t_cut = SignalProcessor.cut_to_integer_cycles(h, b, t)

    sign_changes = np.diff(np.signbit(h_cut))
    crossings = np.where(sign_changes)[0]
    pos_crossings = [idx for idx in crossings if h_cut[idx] < h_cut[idx + 1]]

    assert len(pos_crossings) >= 2
    assert pos_crossings[-1] == len(h_cut) - 2


def test_average_cycles_does_not_close_time_axis():
    theta = np.linspace(0.0, 6.0 * np.pi, 1201)
    h = -np.sin(theta)
    b = np.cos(theta)
    t = np.linspace(0.0, 0.06, len(theta))

    h_cut, b_cut, t_cut = SignalProcessor.cut_to_integer_cycles(h, b, t)
    t_rel = t_cut - t_cut[0]

    _, (_, _), args_stats = SignalProcessor.average_cycles(
        h_cut,
        b_cut,
        t_rel,
        points_per_cycle=400,
        close_aux=[True, False],
    )

    b_avg, _ = args_stats[0]
    t_avg, _ = args_stats[1]

    assert np.isclose(b_avg[-1], b_avg[0])
    assert t_avg[-1] > t_avg[0]
    assert np.all(np.diff(t_avg) > 0)


def test_core_loss_density_nonzero_for_valid_time_series():
    t = np.linspace(0.0, 0.02, 2000)
    omega = 2.0 * np.pi * 50.0
    h = np.sin(omega * t)
    b = np.sin(omega * t - np.pi / 6.0)

    loss = Magnetics.calculate_core_loss_density(h, b, t, frequency=50.0, density=7650.0)

    assert loss > 0.0


def test_drift_correction_aligns_endpoints():
    t = np.linspace(0.0, 1.0, 1000)
    signal_with_drift = np.sin(2.0 * np.pi * 5.0 * t) + 0.8 * t

    corrected = SignalProcessor.apply_drift_correction(signal_with_drift)

    assert np.isclose(corrected[0], corrected[-1], atol=1e-9)


def test_grouped_waveform_analysis_pools_files_and_keeps_time_monotonic():
    cfg = HardwareConstants.default_hwr90_32()
    analyzer = TransformerCoreAnalyzer(cfg)

    t = np.linspace(0.0, 0.06, 6000)
    omega = 2.0 * np.pi * 50.0

    # Build two synthetic "files" for the same condition.
    ch1_a = 0.4 * np.sin(omega * t)
    ch2_a = 0.25 * np.sin(omega * t + np.pi / 4.0)
    ch1_b = 0.4 * np.sin(omega * t + 0.02)
    ch2_b = 0.25 * np.sin(omega * t + np.pi / 4.0 + 0.02)

    result = analyzer.analyze_waveform_group(
        [(t, ch1_a, ch2_a), (t, ch1_b, ch2_b)],
        frequency=50,
    )

    assert result["num_files"] == 2
    assert result["num_cycles"] >= 2
    assert np.all(np.diff(result["time_avg"]) > 0)
