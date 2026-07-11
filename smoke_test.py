"""Smoke test for mimsim: exercise the full MIM chain and every feature."""

import numpy as np

from mimsim import (
    DemodResult,
    Network, Source, DirectionalCoupler, MatchingNetwork, Amplifier,
    PhaseShifter, Attenuator, IQMixer, Termination,
    series_L, shunt_C, TipSampleModel, Signal,
)


def build_chain(f0, tip):
    """Source -> coupler -> matching -> tip; reflected -> amp -> mixer.rf;
    forward tap -> phase shifter -> mixer.lo."""
    src = Source(f0, amplitude=1.0, name="src")
    cpl = DirectionalCoupler(coupling_db=10.0, directivity_db=30.0, name="cpl")
    match = MatchingNetwork([series_L(8e-9), shunt_C(0.5e-12)], name="match")
    amp = Amplifier(gain_db=30.0, name="amp")
    ps = PhaseShifter(voltage=0.0, name="lo_phase")
    mix = IQMixer(name="mixer")

    net = Network(f0=f0)
    net.connect(src.port(0), cpl.port("in"))
    net.connect(cpl.port("through"), match.port("in"))
    net.connect(match.port("out"), tip.port(0))
    net.connect(cpl.port("coupled"), ps.port("in"))
    net.connect(ps.port("out"), mix.port("lo"))
    net.connect(cpl.port("isolated"), amp.port("in"))
    net.connect(amp.port("out"), mix.port("rf"))
    return net, dict(src=src, cpl=cpl, match=match, amp=amp, ps=ps, mix=mix)


def test_basic_attenuator():
    """source -> attenuator -> matched load: output wave halves at 6 dB."""
    f0 = 1e9
    src = Source(f0, amplitude=1.0, name="src")
    att = Attenuator(voltage=6.0, name="att")  # 6 dB (1 dB/V default)
    load = Termination.matched()
    net = Network(f0=f0)
    net.connect(src.port(0), att.port("in"))
    net.connect(att.port("out"), load.port(0))
    res = net.run()
    v_in = res.incident(att.port("in")).env[0]
    v_out = res.reflected(att.port("out")).env[0]
    ratio_db = 20 * np.log10(abs(v_out) / abs(v_in))
    print(f"[attenuator] in={abs(v_in):.4f} out={abs(v_out):.4f} "
          f"ratio={ratio_db:.2f} dB (expect -6)")
    assert abs(ratio_db + 6.0) < 0.1, ratio_db


def test_response_curve():
    """Sweep sample resistivity and reproduce the canonical MIM response curve.

    The tip-sample model's dissipative peak must sit at rho*omega*eps ~ 1, and
    the full-chain demod, once the LO phase is calibrated, must show the same:
    MIM-Re peaked, MIM-Im monotonic.
    """
    f0 = 1e9
    model = TipSampleModel()
    rho_vals = np.logspace(-4, 4, 161)  # Ohm*m
    w, eps = 2 * np.pi * f0, 10 * 8.854e-12
    x = rho_vals * w * eps

    # (1) physics check: modelled tip-sample admittance dissipation peak
    Yts = np.array([model.Y_tipsample(f0, r, 10.0) for r in rho_vals])
    xpk_model = x[np.argmax(Yts.real)]
    assert 0.3 < xpk_model < 3.0, xpk_model

    # (2) full-chain check: encode the sweep on the slow-time axis
    def rho_of_t(t):
        return rho_vals[int(round(t))]

    tip = model.make_load(rho=rho_of_t, eps_r=10.0, name="tip")
    net, comp = build_chain(f0, tip)
    res = net.run(t_grid=np.arange(rho_vals.size))
    demod = comp["mix"].demodulate(res)

    # MIM measures *changes* in admittance: work with the differential contrast
    # (nulled against the most-conductive reference), then calibrate the LO phase
    # by aligning that contrast to the modelled delta-Y.
    dd = demod.complex - demod.complex[0]
    dY = Yts - Yts[0]
    aligned, phi = DemodResult(res.t, dd).align_to(dY)
    re, im = aligned.re, aligned.im
    xpk_chain = x[np.argmax(np.abs(re))]
    print(f"[response] model peak x={xpk_model:.3g}, chain peak x={xpk_chain:.3g} "
          f"(expect ~1), LO phase={np.degrees(phi):.1f} deg")
    assert 0.3 < xpk_chain < 3.0, xpk_chain
    return x, re, im


def test_voltage_control():
    """Changing the attenuator voltage changes the through signal."""
    f0 = 1e9
    src = Source(f0, amplitude=1.0)
    att = Attenuator(voltage=0.0, response=lambda v: 2.0 * v)  # 2 dB/V
    load = Termination.matched()
    net = Network(f0=f0)
    net.connect(src.port(0), att.port("in"))
    net.connect(att.port("out"), load.port(0))

    outs = []
    for v in [0.0, 5.0, 10.0]:
        att.voltage = v
        res = net.run()
        outs.append(abs(res.reflected(att.port("out")).env[0]))
    print(f"[voltage] out amplitudes vs V: {[f'{o:.4f}' for o in outs]}")
    assert outs[0] > outs[1] > outs[2]
    # 10 dB @ V=5 -> factor 10^(-0.5)
    assert abs(20 * np.log10(outs[1] / outs[0]) + 10.0) < 0.1


def test_modulation_sidebands():
    """Height-modulated tip -> line at f_mod in the demodulated (dMIM/dz) signal.

    The raw reflected carrier is huge and the tip-sample modulation rides on it
    at a very low level (aF on ~pF), exactly as in a real instrument -- which is
    why MIM demodulates.  After the IQ mixer the carrier is gone and the tip
    oscillation shows up as a clean line at f_mod in the baseband spectrum.
    """
    f0 = 1e9
    fmod = 1e3            # 1 kHz tip oscillation
    fs = 100e3           # slow-time sample rate
    t = np.arange(4096) / fs
    model = TipSampleModel()

    def z_of_t(tt):
        return 5e-9 * (1 + np.sin(2 * np.pi * fmod * tt))  # 0..10 nm gap

    tip = model.make_load(rho=1.0, eps_r=10.0, z=z_of_t, name="tip")
    net, comp = build_chain(f0, tip)
    res = net.run(t_grid=t)
    demod = comp["mix"].demodulate(res)

    # baseband spectrum of the demodulated contrast (remove the DC offset first)
    sig = Signal(demod.complex - demod.complex.mean(), f0=0.0, t=t)
    fb, S = sig.baseband_spectrum()
    mag = np.abs(S)
    i_mod = np.argmin(np.abs(fb - fmod))          # line at +f_mod
    i_ref = np.argmin(np.abs(fb - 2.5 * fmod))    # empty bin for comparison
    ratio = mag[i_mod] / max(mag[i_ref], 1e-30)
    print(f"[modulation] dMIM/dz line at {fb[i_mod]:.0f} Hz is "
          f"{20*np.log10(ratio):.0f} dB above a non-signal bin")
    assert ratio > 100  # clean modulation line well above the floor


def test_probe_time_and_freq():
    """Probe input & output of the matching network in time and frequency."""
    f0 = 1e9
    model = TipSampleModel()
    tip = model.make_load(rho=1.0, name="tip")
    net, comp = build_chain(f0, tip)
    res = net.run()
    vin = res.voltage(comp["match"].port("in"))
    vout = res.voltage(comp["match"].port("out"))
    t_fast, w_in = vin.waveform(n_periods=3)
    _, w_out = vout.waveform(n_periods=3)
    print(f"[probe] match.in |V|={abs(vin.env[0]):.4g}  "
          f"match.out |V|={abs(vout.env[0]):.4g}  "
          f"waveform samples={w_in.size}")
    assert w_in.size == w_out.size > 0


if __name__ == "__main__":
    test_basic_attenuator()
    test_voltage_control()
    test_response_curve()
    test_modulation_sidebands()
    test_probe_time_and_freq()
    print("\nAll smoke tests passed.")
