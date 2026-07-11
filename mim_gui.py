"""Standalone real-time MIM control panel built on top of the mimsim package.

A running strip-chart in a native matplotlib window: the signal streams against a
real time axis, and moving any slider changes the ongoing signal live. Old points
scroll out of the ~12 s window and are discarded (a bounded ring buffer), so
memory stays flat no matter how long it runs. Same feature set as the notebook
panel (mim_gui_widgets.py); the base package (mimsim/*.py) is untouched.

Run it:
    python mim_gui.py
(FuncAnimation timers are reliable in a native window, unlike in-notebook.)
Inside a notebook prefer the widget version:
    %matplotlib widget
    from mim_gui_widgets import launch_widgets; launch_widgets()
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation

from mimsim import (
    Network, Source, DirectionalCoupler, Amplifier, PhaseShifter, Attenuator,
    IQMixer, lambda4_bridge, TipSampleModel, EPS0,
)

WINDOW_S = 12.0    # seconds of history visible on the time axis
FPS = 15           # streaming sample/redraw rate
MAXLEN = int(WINDOW_S * FPS) + 30   # ring-buffer cap -> bounded memory


# ---------------------------------------------------------------------------
# the MIM chain (faithful three-arm reflectometer) built once; sliders mutate it
# ---------------------------------------------------------------------------
class MIMRig:
    def __init__(self, f0=1e9):
        self.f0 = f0
        # finite-Q probe so a *fixed* matching bridge produces real |Gamma| dips
        self.model = TipSampleModel(C_probe=1e-12, Q_probe=40.0)
        self.src = Source(f0, amplitude=1.0, name="src")
        self.cpl1 = DirectionalCoupler(3.0, 35.0, name="cpl1")   # source split
        self.cpl2 = DirectionalCoupler(3.0, 35.0, name="cpl2")   # ref / cancel split
        self.cplm = DirectionalCoupler(10.0, 35.0, name="cplm")  # excitation / reflection
        self.comb = DirectionalCoupler(3.0, 35.0, name="comb")   # cancellation combiner
        # FIXED lambda/4 bridge (10 cm) + series coupling cap -> a comb of matches
        # at ~equally-spaced frequencies; you sweep the source to find the teeth.
        self.match = lambda4_bridge(length=0.10, Z_line=150.0, C_series=0.12e-12, name="bridge")
        self.ref_ph = PhaseShifter(voltage=0.0, response=lambda v: np.deg2rad(v), name="ref_ph")
        self.can_att = Attenuator(voltage=6.0, response=lambda v: v, name="can_att")   # v = dB
        self.can_ph = PhaseShifter(voltage=0.0, response=lambda v: np.deg2rad(v), name="can_ph")
        self.amp = Amplifier(gain_db=30.0, name="amp")
        self.mix = IQMixer(name="mix")

        # a live tip whose resistivity is read from the slider each solve
        self._rho = 1.0
        self.tip = self.model.make_load(rho=lambda t: self._rho, eps_r=10.0, name="tip")

        net = Network(f0=f0)
        net.connect(self.src.port(0), self.cpl1.port("in"))
        net.connect(self.cpl1.port("through"), self.cplm.port("in"))
        net.connect(self.cpl1.port("coupled"), self.cpl2.port("in"))
        net.connect(self.cpl2.port("through"), self.ref_ph.port("in"))
        net.connect(self.ref_ph.port("out"), self.mix.port("lo"))
        net.connect(self.cpl2.port("coupled"), self.can_att.port("in"))
        net.connect(self.can_att.port("out"), self.can_ph.port("in"))
        net.connect(self.can_ph.port("out"), self.comb.port("isolated"))
        net.connect(self.cplm.port("through"), self.match.port("in"))
        net.connect(self.match.port("out"), self.tip.port(0))
        net.connect(self.cplm.port("isolated"), self.comb.port("in"))
        net.connect(self.comb.port("through"), self.amp.port("in"))
        net.connect(self.amp.port("out"), self.mix.port("rf"))
        self.net = net

    def measure(self, p_dbm, f_ghz, att_db, can_deg, lo_deg, log_rho):
        """Set all controls, solve, and return (MIM-Re, MIM-Im, pre-demod power dBm)."""
        self.src.set_power_dbm(p_dbm)
        self.can_att.voltage = att_db
        self.can_ph.voltage = can_deg
        self.ref_ph.voltage = lo_deg
        self._rho = 10.0 ** log_rho
        res = self.net.run(f=f_ghz * 1e9)

        c = self.mix.demodulate(res).complex[0]
        a_rf = res.incident(self.mix.port("rf")).env[0]      # wave into the mixer RF port
        p_watt = 0.5 * abs(a_rf) ** 2                        # pre-demod microwave power
        p_dbm_rf = 10.0 * np.log10(max(p_watt, 1e-30) / 1e-3)
        return c.real, c.imag, p_dbm_rf


# ---------------------------------------------------------------------------
# the interactive streaming panel
# ---------------------------------------------------------------------------
# (key, label, vmin, vmax, vinit)
CTRLS = [
    ("p_dbm",   "Source power (dBm)",       -30.0,  10.0,  -5.0),
    ("f_ghz",   "Carrier freq (GHz)",         0.30,  4.00,  1.05),
    ("att_db",  "Cancellation A (dB)",        0.0,   40.0,  40.0),
    ("can_deg", "Cancellation phase (deg)",  -180.0, 180.0, 0.0),
    ("lo_deg",  "Reference LO phase (deg)",  -180.0, 180.0,  0.0),
    ("log_rho", "Sample log10 rho",           -4.0,   4.0,   0.0),
]
TEAL, INDIGO, GREEN, TRAILC = "#0e7f8c", "#4b53cf", "#1f8f66", "#8a90e0"


class MIMPanel:
    def __init__(self, rig=None):
        self.rig = rig or MIMRig()
        self.tb = deque(maxlen=MAXLEN)
        self.reb = deque(maxlen=MAXLEN)
        self.imb = deque(maxlen=MAXLEN)
        self.pwb = deque(maxlen=MAXLEN)
        self.paused = False
        self.noise = True
        self.vclock = 0.0
        self._last_real = time.monotonic()

        self.fig = plt.figure(figsize=(12.4, 9.0))
        try:
            self.fig.canvas.manager.set_window_title("mimsim — live MIM control panel")
        except Exception:
            pass

        # ---- four live plots (top region) ----
        gs = self.fig.add_gridspec(2, 2, left=0.07, right=0.75, top=0.965, bottom=0.44,
                                   wspace=0.28, hspace=0.40)
        self.ax_iq = self.fig.add_subplot(gs[0, 0])
        self.ax_pw = self.fig.add_subplot(gs[0, 1])
        self.ax_re = self.fig.add_subplot(gs[1, 0])
        self.ax_im = self.fig.add_subplot(gs[1, 1])
        self.ax_iq.set(title="Demodulated output (I/Q)", xlabel="MIM-Re", ylabel="MIM-Im")
        self.ax_iq.axhline(0, color="0.85", lw=0.8); self.ax_iq.axvline(0, color="0.85", lw=0.8)
        self.ax_iq.grid(alpha=0.25)
        (self.trail_ln,) = self.ax_iq.plot([], [], "-", color=TRAILC, lw=1, alpha=0.6)
        (self.pt_ln,) = self.ax_iq.plot([], [], "o", color=TEAL, ms=10, mec="white", mew=1.2)
        self.ax_pw.set(title="Power before demodulation", ylabel="dBm", xlabel="time (s)")
        self.ax_pw.grid(alpha=0.25)
        (self.pw_ln,) = self.ax_pw.plot([], [], "-", color=GREEN, lw=1.8)
        self.ax_re.set(title="MIM-Re vs time", ylabel="MIM-Re", xlabel="time (s)")
        self.ax_re.grid(alpha=0.25)
        (self.re_ln,) = self.ax_re.plot([], [], "-", color=TEAL, lw=1.6)
        self.ax_im.set(title="MIM-Im vs time", ylabel="MIM-Im", xlabel="time (s)")
        self.ax_im.grid(alpha=0.25)
        (self.im_ln,) = self.ax_im.plot([], [], "-", color=INDIGO, lw=1.6)

        # ---- numeric readouts (top right) ----
        self.txt = self.fig.text(0.78, 0.93, "", family="monospace", fontsize=12, va="top")
        self.txt_pw = self.fig.text(0.78, 0.78, "", family="monospace", fontsize=15,
                                    color=GREEN, fontweight="bold", va="top")

        # ---- sliders (bottom left) ----
        self.sliders = {}
        for i, (key, label, lo, hi, init) in enumerate(CTRLS):
            ax = self.fig.add_axes([0.22, 0.355 - i * 0.055, 0.42, 0.03])
            s = Slider(ax, label, lo, hi, valinit=init, color=TEAL)
            s.label.set_fontsize(9.5)
            self.sliders[key] = s

        # ---- buttons (bottom middle-right) ----
        def button(x, y, w, h, label, cb, color="0.9"):
            b = Button(self.fig.add_axes([x, y, w, h]), label, color=color, hovercolor="0.8")
            b.label.set_fontsize(9.5); b.on_clicked(cb)
            return b
        self.btn_pause = button(0.70, 0.355, 0.11, 0.045, "Pause", self._toggle_pause, "#ffe6cc")
        self.btn_null = button(0.82, 0.355, 0.16, 0.045, "Auto-null", self._auto_null, "#d7f0e5")
        self.btn_clear = button(0.70, 0.295, 0.11, 0.045, "Reset", self._reset, "#f2d9d9")
        self.btn_sweep_rho = button(0.82, 0.295, 0.16, 0.045, "Sweep ρ", self._sweep_rho, "#e2eefb")
        self.btn_sweep_f = button(0.70, 0.235, 0.28, 0.045, "Sweep frequency", self._sweep_freq, "#e2eefb")
        self.chk_noise = CheckButtons(self.fig.add_axes([0.70, 0.16, 0.28, 0.06]),
                                      ["live noise"], [True])
        self.chk_noise.on_clicked(self._toggle_noise)

        self._sample(0.0); self._redraw(0.0)
        self.anim = FuncAnimation(self.fig, self._frame, interval=int(1000 / FPS),
                                  blit=False, cache_frame_data=False)

    # -- sampling ----------------------------------------------------------
    def _vals(self):
        return {k: self.sliders[k].val for k, *_ in CTRLS}

    def _measure(self):
        v = self._vals()
        return self.rig.measure(v["p_dbm"], v["f_ghz"], v["att_db"],
                                v["can_deg"], v["lo_deg"], v["log_rho"])

    def _sample(self, t):
        re, im, pw = self._measure()
        if self.noise:
            mag = max(np.hypot(re, im), 1e-9)
            re += np.random.randn() * 0.006 * mag
            im += np.random.randn() * 0.006 * mag
            pw += np.random.randn() * 0.06
        self.tb.append(t); self.reb.append(re); self.imb.append(im); self.pwb.append(pw)
        return re, im, pw

    # -- animation frame ---------------------------------------------------
    def _frame(self, _i):
        real = time.monotonic()
        if not self.paused:
            self.vclock += real - self._last_real
        self._last_real = real
        if self.paused:
            return self._artists()
        self._sample(self.vclock)
        self._redraw(self.vclock)
        return self._artists()

    def _artists(self):
        return self.trail_ln, self.pt_ln, self.pw_ln, self.re_ln, self.im_ln

    # -- drawing -----------------------------------------------------------
    def _redraw(self, now):
        t = np.fromiter(self.tb, float)
        re = np.fromiter(self.reb, float)
        im = np.fromiter(self.imb, float)
        pw = np.fromiter(self.pwb, float)
        cre, cim, cpw = re[-1], im[-1], pw[-1]

        self.txt.set_text(
            f"MIM-Re = {cre:+.4e}\n"
            f"MIM-Im = {cim:+.4e}\n"
            f"|MIM|  =  {np.hypot(cre, cim):.4e}\n"
            f"/_MIM  = {np.degrees(np.arctan2(cim, cre)):+7.1f} deg")
        self.txt_pw.set_text(f"pre-demod RF\n= {cpw:+.2f} dBm")

        self.trail_ln.set_data(re, im); self.pt_ln.set_data([cre], [cim])
        self._fit_iq(re, im)
        x0, x1 = now - WINDOW_S, now
        for ax, ln, y in ((self.ax_pw, self.pw_ln, pw), (self.ax_re, self.re_ln, re),
                          (self.ax_im, self.im_ln, im)):
            ln.set_data(t, y); ax.set_xlim(x0, x1); self._fit_y(ax, y)

    def _fit_iq(self, xs, ys):
        if len(xs) == 0:
            return
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        mx = max(x1 - x0, 1e-12) * 0.15; my = max(y1 - y0, 1e-12) * 0.15
        self.ax_iq.set_xlim(x0 - mx - 1e-12, x1 + mx + 1e-12)
        self.ax_iq.set_ylim(y0 - my - 1e-12, y1 + my + 1e-12)

    @staticmethod
    def _fit_y(ax, y):
        if len(y) == 0:
            return
        lo, hi = y.min(), y.max()
        pad = max(0.1 * (hi - lo), abs(hi) * 0.05, 1e-9)
        ax.set_ylim(lo - pad, hi + pad)

    # -- controls ----------------------------------------------------------
    def _toggle_pause(self, _evt):
        self.paused = not self.paused
        self.btn_pause.label.set_text("Resume" if self.paused else "Pause")
        self.fig.canvas.draw_idle()

    def _toggle_noise(self, _label):
        self.noise = self.chk_noise.get_status()[0]

    def _reset(self, _evt):
        """Full reset: clear the data, restart the time axis at 0, restore the
        sliders to their defaults, and resume streaming."""
        for b in (self.tb, self.reb, self.imb, self.pwb):
            b.clear()
        self.vclock = 0.0
        self._last_real = time.monotonic()
        for s in self.sliders.values():
            s.reset()                       # back to valinit
        if self.paused:
            self.paused = False
            self.btn_pause.label.set_text("Pause")
        self._sample(0.0); self._redraw(0.0)
        self.fig.canvas.draw_idle()

    def _auto_null(self, _evt):
        v = self._vals()
        def power(a, ph):
            return self.rig.measure(v["p_dbm"], v["f_ghz"], a, ph, v["lo_deg"], v["log_rho"])[2]
        best, ba = 1e9, (v["att_db"], v["can_deg"])
        for a in np.linspace(0, 25, 26):
            for ph in np.linspace(-180, 180, 49):
                p = power(a, ph)
                if p < best:
                    best, ba = p, (a, ph)
        try:
            from scipy.optimize import minimize
            r = minimize(lambda x: power(max(x[0], 0.0), x[1]), list(ba),
                         method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3})
            if power(max(r.x[0], 0.0), r.x[1]) < best:
                ba = (max(r.x[0], 0.0), r.x[1])
        except Exception:
            pass
        self.sliders["att_db"].set_val(float(np.clip(ba[0], 0, 40)))
        self.sliders["can_deg"].set_val(float(np.clip(((ba[1] + 180) % 360) - 180, -180, 180)))

    # -- sweeps (open a separate figure window) ----------------------------
    def _sweep_rho(self, _evt):
        v = self._vals()
        rhos = np.logspace(-4, 4, 161)
        w, eps = 2 * np.pi * self.rig.f0, 10 * EPS0
        x = rhos * w * eps
        d = np.array([complex(*self.rig.measure(v["p_dbm"], v["f_ghz"], v["att_db"],
                              v["can_deg"], v["lo_deg"], np.log10(r))[:2]) for r in rhos])
        Yts = np.array([self.rig.model.Y_tipsample(self.rig.f0, r, 10.0) for r in rhos])
        dd = d - d[0]
        dd = dd * np.exp(-1j * np.angle(np.sum(dd * np.conj(Yts - Yts[0]))))
        norm = np.abs(dd).max() or 1.0
        fig, ax = plt.subplots(figsize=(6.6, 3.6), num="Sweep ρ → response curve")
        ax.semilogx(x, dd.real / norm, color=TEAL, lw=2, label="MIM-Re (loss)")
        ax.semilogx(x, dd.imag / norm, color=INDIGO, lw=2, label="MIM-Im (capacitance)")
        ax.axvline(1.0, ls="--", c="0.5", lw=1)
        ax.set(xlabel=r"$\rho\,\omega\,\varepsilon$  (conductive $\leftarrow$ | $\rightarrow$ insulating)",
               ylabel="MIM (norm.)", title="Response curve (swept sample resistivity)")
        ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9)
        fig.tight_layout(); fig.show()

    def _sweep_freq(self, _evt):
        v = self._vals()
        fs = np.linspace(CTRLS[1][2], CTRLS[1][3], 221)
        p = np.array([self.rig.measure(v["p_dbm"], f, v["att_db"], v["can_deg"],
                                       v["lo_deg"], v["log_rho"])[2] for f in fs])
        fig, ax = plt.subplots(figsize=(6.6, 3.6), num="Sweep frequency")
        ax.plot(fs, p, color=GREEN, lw=2)
        ax.axvline(v["f_ghz"], ls="--", c="0.5", lw=1, label="current f")
        ax.set(xlabel="carrier frequency (GHz)", ylabel="pre-demod power (dBm)",
               title="Frequency sweep (match / cancellation vs frequency)")
        ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9)
        fig.tight_layout(); fig.show()


def launch(rig=None):
    """Open the interactive panel (blocks until the window is closed)."""
    panel = MIMPanel(rig)
    plt.show()
    return panel


if __name__ == "__main__":
    launch()
