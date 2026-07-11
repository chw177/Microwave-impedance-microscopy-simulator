"""In-notebook ipywidgets control panel for MIM, built on top of mimsim.

A real-time strip-chart: the signal streams continuously against a time axis,
and moving any slider changes the ongoing signal live. Points that scroll out of
the visible window are dropped automatically (a fixed-length ring buffer), so
memory stays bounded no matter how long it runs. The base mimsim package is
untouched; this only drives the verified rig from mim_gui.py.

The stream is driven by an ipywidgets Play widget (frontend-timed, so it runs
reliably in JupyterLab, classic Notebook and VS Code). Best plots need the
interactive backend:
    %matplotlib widget
    from mim_gui_widgets import launch_widgets
    panel = launch_widgets()
Then press the Play (>) button to start streaming.
"""

from __future__ import annotations

import io
from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display, Image as IPImage

from mim_gui import MIMRig   # reuse the verified three-arm rig
from mimsim import EPS0

WINDOW_S = 12.0    # seconds of history visible on the time axis
FPS = 12           # streaming sample/redraw rate
MAXLEN = int(WINDOW_S * FPS) + 24   # ring-buffer cap -> bounded memory


class WidgetPanel:
    def __init__(self, rig=None):
        self.rig = rig or MIMRig()
        # bounded ring buffers: old points fall off automatically
        self.tb = deque(maxlen=MAXLEN)
        self.reb = deque(maxlen=MAXLEN)
        self.imb = deque(maxlen=MAXLEN)
        self.pwb = deque(maxlen=MAXLEN)
        self._t = 0.0            # monotonic stream time (s), independent of Play's counter
        self._last_tick = 0      # last Play value seen (to detect Stop/rewind)

        # ---- sliders ----
        sl = dict(continuous_update=True, readout_format=".1f",
                  layout=W.Layout(width="420px"), style={"description_width": "170px"})
        self.s_power = W.FloatSlider(description="Source power (dBm)", min=-30, max=10, value=-5, step=0.5, **sl)
        self.s_freq = W.FloatSlider(description="Carrier freq (GHz)", min=0.30, max=4.00, value=1.05, step=0.005,
                                    continuous_update=True, readout_format=".3f",
                                    layout=W.Layout(width="420px"), style={"description_width": "170px"})
        self.s_att = W.FloatSlider(description="Cancellation A (dB)", min=0, max=40, value=40, step=0.25, **sl)
        self.s_canph = W.FloatSlider(description="Cancellation phase (deg)", min=-180, max=180, value=0, step=1, **sl)
        self.s_loph = W.FloatSlider(description="Reference LO phase (deg)", min=-180, max=180, value=0, step=1, **sl)
        self.s_rho = W.FloatSlider(description="Sample log10 rho", min=-4, max=4, value=0, step=0.05, **sl)
        self._sliders = [self.s_power, self.s_freq, self.s_att, self.s_canph, self.s_loph, self.s_rho]

        # ---- readouts ----
        self.readout = W.HTML()
        self.meter = W.FloatProgress(min=-70, max=20, value=0, description="pre-demod",
                                     bar_style="info", orientation="horizontal",
                                     layout=W.Layout(width="420px"), style={"description_width": "170px"})
        self.meter_lbl = W.HTML()

        # ---- stream engine (Play) + controls ----
        self.play = W.Play(value=0, min=0, max=2_000_000_000, step=1, interval=int(1000 / FPS),
                           description="stream")
        self.cb_noise = W.Checkbox(value=True, description="live noise", indent=False,
                                   layout=W.Layout(width="120px"))
        self.btn_null = W.Button(description="Auto-null cancellation", button_style="success",
                                 layout=W.Layout(width="205px"))
        self.btn_clear = W.Button(description="Reset", button_style="warning", layout=W.Layout(width="90px"))
        self.btn_sweep_rho = W.Button(description="Sweep ρ → response curve", button_style="info",
                                      layout=W.Layout(width="205px"))
        self.btn_sweep_f = W.Button(description="Sweep frequency", button_style="info",
                                    layout=W.Layout(width="150px"))
        self.play.observe(self._step, names="value")
        self.btn_null.on_click(self._auto_null)
        self.btn_clear.on_click(self._clear)
        self.btn_sweep_rho.on_click(self._sweep_rho)
        self.btn_sweep_f.on_click(self._sweep_freq)
        self.sweep_out = W.Output()

        # ---- figure ----
        self._use_canvas = "ipympl" in matplotlib.get_backend().lower()
        plt.ioff()
        self.fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.3))
        self.fig.subplots_adjust(left=0.10, right=0.98, bottom=0.08, top=0.94, wspace=0.28, hspace=0.40)
        self.ax_iq, self.ax_pw = axs[0, 0], axs[0, 1]
        self.ax_re, self.ax_im = axs[1, 0], axs[1, 1]
        self.ax_iq.set(title="Demodulated output (I/Q)", xlabel="MIM-Re", ylabel="MIM-Im")
        self.ax_iq.axhline(0, color="0.85", lw=0.8); self.ax_iq.axvline(0, color="0.85", lw=0.8)
        self.ax_iq.grid(alpha=0.25)
        (self.trail_ln,) = self.ax_iq.plot([], [], "-", color="#8a90e0", lw=1, alpha=0.6)
        (self.pt_ln,) = self.ax_iq.plot([], [], "o", color="#0e7f8c", ms=10, mec="white", mew=1.2)
        self.ax_pw.set(title="Power before demodulation", ylabel="dBm", xlabel="time (s)")
        self.ax_pw.grid(alpha=0.25)
        (self.pw_ln,) = self.ax_pw.plot([], [], "-", color="#1f8f66", lw=1.8)
        self.ax_re.set(title="MIM-Re vs time", ylabel="MIM-Re", xlabel="time (s)")
        self.ax_re.grid(alpha=0.25)
        (self.re_ln,) = self.ax_re.plot([], [], "-", color="#0e7f8c", lw=1.6)
        self.ax_im.set(title="MIM-Im vs time", ylabel="MIM-Im", xlabel="time (s)")
        self.ax_im.grid(alpha=0.25)
        (self.im_ln,) = self.ax_im.plot([], [], "-", color="#4b53cf", lw=1.6)
        if self._use_canvas:
            self.fig.canvas.header_visible = False
            self.fig.canvas.footer_visible = False
            self.plot_widget = self.fig.canvas
        else:
            self.plot_out = W.Output()
            self.plot_widget = self.plot_out

        for s in self._sliders:
            s.observe(self._peek, names="value")

        self.ui = W.VBox([
            W.HBox([W.HTML("<b style='font-size:15px'>mimsim &mdash; live MIM control panel</b> "
                           "&nbsp; press&nbsp;&#9654;&nbsp;to stream &rarr;"), self.play, self.cb_noise]),
            W.HBox([W.VBox(self._sliders),
                    W.VBox([self.readout, self.meter, self.meter_lbl,
                            W.HBox([self.btn_null, self.btn_clear]),
                            W.HBox([self.btn_sweep_rho, self.btn_sweep_f])])]),
            self.plot_widget,
            self.sweep_out,
        ])

        self._sample(0.0); self._redraw(0.0); self._paint()

    # -- sampling ----------------------------------------------------------
    def _measure(self):
        return self.rig.measure(self.s_power.value, self.s_freq.value, self.s_att.value,
                                self.s_canph.value, self.s_loph.value, self.s_rho.value)

    def _sample(self, t):
        re, im, pw = self._measure()
        if self.cb_noise.value:                     # a touch of instrument noise -> looks alive
            mag = max(np.hypot(re, im), 1e-9)
            re += np.random.randn() * 0.006 * mag
            im += np.random.randn() * 0.006 * mag
            pw += np.random.randn() * 0.06
        self.tb.append(t); self.reb.append(re); self.imb.append(im); self.pwb.append(pw)
        return re, im, pw

    # -- drivers -----------------------------------------------------------
    def _step(self, change):
        """One streaming frame, driven by the Play widget's tick.

        Pressing Stop rewinds Play's counter to 0; when that happens we clear the
        buffers and restart the timeline so old data doesn't linger or draw stray
        lines. Pause holds the counter, so resume continues seamlessly.
        """
        tick = change["new"]
        if tick <= self._last_tick:          # Stop/rewind -> fresh run
            for b in (self.tb, self.reb, self.imb, self.pwb):
                b.clear()
            self._t = 0.0
        else:
            self._t += 1.0 / FPS
        self._last_tick = tick
        self._sample(self._t)
        self._redraw(self._t)
        self._paint()

    def _peek(self, _change):
        """Instant feedback while dragging: update readouts + current point (no append)."""
        re, im, pw = self._measure()
        self._readout(re, im, pw)
        self.pt_ln.set_data([re], [im])
        self._paint()

    def _paint(self):
        if self._use_canvas:
            self.fig.canvas.draw_idle()
        else:
            with self.plot_out:
                self.plot_out.clear_output(wait=True)
                display(self.fig)

    # -- drawing -----------------------------------------------------------
    def _readout(self, re, im, pw):
        mag, ang = np.hypot(re, im), np.degrees(np.arctan2(im, re))
        self.readout.value = (
            "<pre style='font-size:13px;line-height:1.5;margin:0'>"
            f"MIM-Re = {re:+.4e}\n"
            f"MIM-Im = {im:+.4e}\n"
            f"|MIM|  =  {mag:.4e}\n"
            f"/_MIM  = {ang:+7.1f} deg</pre>")
        self.meter.value = float(np.clip(pw, self.meter.min, self.meter.max))
        self.meter_lbl.value = (f"<b style='color:#1f8f66;font-size:15px'>"
                                f"pre-demod RF = {pw:+.2f} dBm</b>")

    def _redraw(self, now):
        t = np.fromiter(self.tb, float)
        re = np.fromiter(self.reb, float)
        im = np.fromiter(self.imb, float)
        pw = np.fromiter(self.pwb, float)
        self._readout(re[-1], im[-1], pw[-1])
        self.trail_ln.set_data(re, im); self.pt_ln.set_data([re[-1]], [im[-1]])
        self._fit_iq(re, im)
        x0, x1 = now - WINDOW_S, now
        for ax, ln, y in ((self.ax_pw, self.pw_ln, pw), (self.ax_re, self.re_ln, re), (self.ax_im, self.im_ln, im)):
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

    # -- buttons -----------------------------------------------------------
    def _clear(self, _btn):
        """Full reset: clear the data, restart the time axis at 0, and restore
        the sliders to their defaults."""
        for b in (self.tb, self.reb, self.imb, self.pwb):
            b.clear()
        self._t = 0.0                       # restart the stream time axis
        defaults = dict(s_power=-5, s_freq=1.05, s_att=40, s_canph=0, s_loph=0, s_rho=0)
        for name, val in defaults.items():
            getattr(self, name).value = val
        self._sample(0.0); self._redraw(0.0); self._paint()

    def _auto_null(self, _btn):
        """Find the cancellation A and phase that minimize the pre-demod power."""
        def power(a, ph):
            return self.rig.measure(self.s_power.value, self.s_freq.value, a, ph,
                                    self.s_loph.value, self.s_rho.value)[2]
        best, ba = 1e9, (self.s_att.value, self.s_canph.value)
        for a in np.linspace(0, 25, 26):
            for ph in np.linspace(-180, 180, 49):
                p = power(a, ph)
                if p < best:
                    best, ba = p, (a, ph)
        try:
            from scipy.optimize import minimize
            res = minimize(lambda x: power(max(x[0], 0.0), x[1]), list(ba),
                           method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3})
            if power(max(res.x[0], 0.0), res.x[1]) < best:
                ba = (max(res.x[0], 0.0), res.x[1])
        except Exception:
            pass
        self.s_att.value = float(np.clip(ba[0], self.s_att.min, self.s_att.max))
        self.s_canph.value = float(np.clip(((ba[1] + 180) % 360) - 180, -180, 180))

    # -- sweeps (static plot into an Output; do not disturb the live stream) --
    def _show_sweep(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        with self.sweep_out:
            self.sweep_out.clear_output(wait=True)
            display(IPImage(data=buf.getvalue()))

    def _sweep_rho(self, _btn):
        rhos = np.logspace(-4, 4, 161)
        w, eps = 2 * np.pi * self.rig.f0, 10 * EPS0
        x = rhos * w * eps
        v = (self.s_power.value, self.s_freq.value, self.s_att.value,
             self.s_canph.value, self.s_loph.value)
        d = np.array([complex(*self.rig.measure(*v, np.log10(r))[:2]) for r in rhos])
        Yts = np.array([self.rig.model.Y_tipsample(self.rig.f0, r, 10.0) for r in rhos])
        dd = d - d[0]
        dd = dd * np.exp(-1j * np.angle(np.sum(dd * np.conj(Yts - Yts[0]))))
        norm = np.abs(dd).max() or 1.0
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.semilogx(x, dd.real / norm, color="#0e7f8c", lw=2, label="MIM-Re (loss)")
        ax.semilogx(x, dd.imag / norm, color="#4b53cf", lw=2, label="MIM-Im (capacitance)")
        ax.axvline(1.0, ls="--", c="0.5", lw=1)
        ax.set(xlabel=r"$\rho\,\omega\,\varepsilon$  (conductive $\leftarrow$ | $\rightarrow$ insulating)",
               ylabel="MIM (norm.)", title="Response curve  (swept sample resistivity)")
        ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9)
        self._show_sweep(fig)

    def _sweep_freq(self, _btn):
        fs = np.linspace(self.s_freq.min, self.s_freq.max, 221)
        rest = (self.s_att.value, self.s_canph.value, self.s_loph.value, self.s_rho.value)
        p = np.array([self.rig.measure(self.s_power.value, f, *rest)[2] for f in fs])
        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        ax.plot(fs, p, color="#1f8f66", lw=2)
        ax.axvline(self.s_freq.value, ls="--", c="0.5", lw=1, label="current f")
        ax.set(xlabel="carrier frequency (GHz)", ylabel="pre-demod power (dBm)",
               title="Frequency sweep  (match / cancellation vs frequency)")
        ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9)
        self._show_sweep(fig)


def launch_widgets(rig=None):
    """Build and display the panel in a notebook. Returns the WidgetPanel."""
    panel = WidgetPanel(rig)
    display(panel.ui)
    return panel
