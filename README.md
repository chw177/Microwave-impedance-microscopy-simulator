# mimsim — a microwave impedance microscopy (MIM) electronics simulator

A traveling-wave / S-parameter simulator for the full MIM signal chain, following
Barber, Ma & Shen, *Nat. Rev. Phys.* **4**, 61 (2022). It lets you **request the signal
at the input or output of any component**, in both the **time domain** (RF waveform) and
the **frequency domain** (carrier + sidebands), while controlling attenuators and phase
shifters by **voltage** and tuning the source **amplitude and frequency** — and it ships
with a real-time interactive control panel.

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
| `mimsim/matching.py`   | `resonant_match` (per-frequency **designed** λ/4 match; `mismatch_db`/`L_scale` model residual mismatch and drift) and `lambda4_bridge` (a **fixed** physical λ/4 bridge + series coupling cap → a *comb* of match frequencies) |
| `mimsim/sample.py`     | `TipSampleModel` — lumped tip-sample admittance / response curve; `Q_probe` gives the probe a finite loss so a fixed bridge can match |
| `mim_gui.py`           | **standalone** real-time control panel — `python mim_gui.py` |
| `mim_gui_widgets.py`   | in-**notebook** control panel (ipywidgets) — `launch_widgets()` |
| `mim_gui_demo.ipynb`   | one-click notebook to open the panel |
| `MIMSimulationFull.ipynb` | worked demo of every feature (executed, with figures) |
| `smoke_test.py` · `build_notebook.py` | self-check · (re)generate the demo notebook |
| `docs/`                | tutorials (EN/中文, full + undergrad) and architecture diagrams as HTML + PDF |

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

### Impedance matching: designed vs a fixed physical bridge

```python
from mimsim import resonant_match, lambda4_bridge

# (a) a match "designed" at f0 — matched at any frequency you build it for
m = resonant_match(f0, C_probe=1e-12, Q=40)

# (b) a FIXED λ/4 bridge (~10 cm) + series coupling cap: not tunable, so matching
#     happens only at the line's resonances — a comb of ~equally-spaced frequencies
#     (~c/2·length apart). Sweep the source frequency to find the matched teeth.
bridge = lambda4_bridge(length=0.10, Z_line=150.0, C_series=0.12e-12)
tip = TipSampleModel(C_probe=1e-12, Q_probe=40).make_load(rho=1.0)  # finite-Q -> real |Γ| dips
```

### Adding your own component

Subclass `Component`, set `n_ports` / `port_names`, and implement `s_matrix(f, t)`
(and `b_source(f, t)` for sources).

## Interactive control panel

A real-time strip-chart: the signal streams against a wall-clock time axis and every
control is a live slider. As you drag, MIM-Re / MIM-Im and the microwave power **before
demodulation** update live; old points scroll out of the window and are discarded so
memory stays bounded.

```bash
python mim_gui.py            # standalone window (streams immediately)
```

Or inside a notebook (needs `%matplotlib widget` / ipympl):

```python
%matplotlib widget
from mim_gui_widgets import launch_widgets
panel = launch_widgets()     # press ▶ to stream
```

Sliders: source power, carrier frequency, cancellation attenuator A, cancellation phase,
reference LO phase, sample resistivity. Buttons: Pause/Resume · **Auto-null** the carrier ·
**Reset** · **Sweep ρ** (response curve) · **Sweep frequency** (finds the matched teeth of
the fixed bridge). Panels: I/Q plane, pre-demod power vs time, MIM-Re vs time, MIM-Im vs
time. Typical workflow: *sweep frequency → park on a matched tooth → tune A and phase.*

The base package (`mimsim/*.py`) is untouched by the GUIs; they only drive it.

## Documentation

`docs/` holds rendered tutorials and diagrams (open the PDFs):

- `tutorial.pdf` / `tutorial_cn.pdf` — full derivation (EN / 中文): scattering solve,
  component S-matrices, matching, tip-sample response, cancellation, modulation.
- `tutorial_intro.pdf` / `tutorial_intro_cn.pdf` — undergraduate-friendly guide (EN / 中文).
- `architecture.pdf` — code-structure and circuit-to-code diagrams.

## Run it

```bash
python smoke_test.py          # self-check
python mim_gui.py             # interactive panel
python build_notebook.py      # (re)generate the executed demo notebook
```

Requires `numpy`, `scipy`, `matplotlib`. The notebook GUI also needs `ipywidgets` +
`ipympl`; building the demo notebook / tutorials needs `nbformat`, `nbclient`.
