# mimsim — a microwave impedance microscopy (MIM) electronics simulator

A traveling-wave / S-parameter simulator for the full MIM signal chain, following
Barber, Ma & Shen, *Nat. Rev. Phys.* **4**, 61 (2022). It lets you **request the signal
at the input or output of any component**, in both the **time domain** (RF waveform) and
the **frequency domain** (carrier + sidebands), while controlling attenuators and phase
shifters by **voltage** and tuning the source **amplitude and frequency**.

## How it works

Every signal is a narrowband complex **baseband envelope** at the carrier `f0`; the real
RF voltage is `Re{env(t)·e^{j2πf0t}}`. Slow modulation (tip-height oscillation, drift,
gate sweeps) lives on a "slow-time" grid, so the GHz carrier and the kHz modulation are
handled on their two natural timescales.

The network is solved at the carrier frequency with normalized scattering waves. Each
port has an incoming wave `a` and outgoing wave `b`; per component `b = S·a + b_src`, and
ideal connections swap waves between ports. Writing the connection map as a permutation
`P` (`a = P·b`), the whole network is one linear solve:

```
(I − S·P)·b = b_src
```

so reflections, impedance mismatch, standing waves and carrier cancellation are all
physical. Time-varying control is handled by re-solving on the slow-time grid.

## Package layout

| file | contents |
|---|---|
| `mimsim/signal.py`     | `Signal` — complex envelope, `.waveform()` (time), `.spectrum()` (freq) |
| `mimsim/network.py`    | `Component`, `Port`, `Network` solver, `Result` (per-port signal extraction) |
| `mimsim/components.py` | `Source`, `Attenuator`, `PhaseShifter`, `DirectionalCoupler`, `MatchingNetwork`, `Amplifier`, `IQMixer`, `Termination`, `Load` + ABCD element helpers |
| `mimsim/matching.py`   | `resonant_match` — designs the paper's λ/4 resonant probe match; `mismatch_db` / `L_scale` model realistic residual mismatch and thermal detuning |
| `mimsim/sample.py`     | `TipSampleModel` — lumped tip-sample admittance / response curve |
| `MIMSimulationFull.ipynb` | worked demo of every feature (executed, with figures) |
| `smoke_test.py`        | fast self-check of all features |
| `build_notebook.py`    | regenerates and re-executes the demo notebook |

## Quick start

```python
import numpy as np
from mimsim import (Network, Source, DirectionalCoupler, resonant_match,
                    Amplifier, PhaseShifter, Attenuator, IQMixer,
                    TipSampleModel)

f0 = 1e9
src   = Source(f0, amplitude=1.0)
cpl   = DirectionalCoupler(coupling_db=10)
match = resonant_match(f0, C_probe=1e-12, Q=40)          # λ/4 resonant probe match
tip   = TipSampleModel().make_load(rho=1.0, eps_r=10.0)   # 1 Ohm*m sample
lo_ph = PhaseShifter(voltage=0.0)                         # LO phase (voltage-controlled)
amp   = Amplifier(gain_db=30)
mix   = IQMixer()

net = Network(f0=f0)
net.connect(src.port(0),          cpl.port("in"))
net.connect(cpl.port("through"),  match.port("in"))
net.connect(match.port("out"),    tip.port(0))
net.connect(cpl.port("coupled"),  lo_ph.port("in"))
net.connect(lo_ph.port("out"),    mix.port("lo"))
net.connect(cpl.port("isolated"), amp.port("in"))
net.connect(amp.port("out"),      mix.port("rf"))

res = net.run()                                   # solve the network

# probe any node, time & frequency domain
v = res.voltage(match.port("out"))
t_fast, waveform = v.waveform(n_periods=3)         # RF oscillation
freq, spectrum   = v.spectrum()                    # carrier + sidebands

# demodulated MIM channels
mim = mix.demodulate(res)
print("MIM-Re =", mim.re, "  MIM-Im =", mim.im)
```

### Voltage control

```python
att = Attenuator(voltage=3.0, response=lambda v: 2.0*v)     # 2 dB per volt
ps  = PhaseShifter(voltage=1.0, response=lambda v: np.deg2rad(36*v))  # 36 deg/V
```

### Modulation (any parameter can be a callable of slow time `t`)

```python
z = lambda t: 5e-9*(1 + np.sin(2*np.pi*1e3*t))   # 1 kHz tip-height oscillation
tip = TipSampleModel().make_load(rho=1.0, z=z)
res = net.run(t_grid=np.arange(4096)/100e3)      # solve over slow time
```

### Adding your own component

Subclass `Component`, set `n_ports` / `port_names`, and implement `s_matrix(f, t)`
(and `b_source(f, t)` for sources).

## Run it

```bash
python smoke_test.py          # self-check
python build_notebook.py      # (re)generate the executed demo notebook
```

Requires `numpy`, `scipy`, `matplotlib` (+ `nbformat`, `nbclient` to build the notebook).
