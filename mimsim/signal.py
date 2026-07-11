"""Signal representation for the MIM simulator.

All microwave signals in the simulator are narrowband: a carrier at ``f0``
(typically ~0.1-10 GHz) with a slowly varying complex envelope.  We therefore
store every node signal as a *complex baseband envelope* sampled on a "slow"
time grid (the modulation / dynamics timescale, e.g. tip-height oscillation at
kHz).  The physical real-valued RF voltage at any instant is reconstructed as

    v(t) = Re{ env(t) * exp(j 2 pi f0 t) }.

This two-timescale approach avoids sampling a GHz carrier over millisecond
windows (which would need ~1e12 samples), while still giving both the
time-domain oscillation and the frequency-domain spectrum (carrier + sidebands).
"""

from __future__ import annotations

import numpy as np


class Signal:
    """A narrowband signal: complex envelope on a slow-time grid at carrier f0.

    Parameters
    ----------
    envelope : complex or array_like
        Complex envelope value(s).  A scalar (or length-1 array) means a static
        (CW) signal.  An array means a time-varying / modulated signal sampled
        on ``t``.
    f0 : float
        Carrier frequency in Hz.
    t : array_like, optional
        Slow-time grid (seconds), same length as ``envelope``.  Defaults to a
        single point at t=0 for static signals, or ``arange(len)`` scaled by
        ``dt`` if ``dt`` is given.
    dt : float, optional
        Slow-time sample spacing, used to build ``t`` when ``t`` is not given.
    Z0 : float
        Reference impedance the wave amplitude is normalized to (Ohms).
    label : str
        Human-readable name of the probed node (for plotting).
    """

    def __init__(self, envelope, f0, t=None, dt=None, Z0=50.0, label=""):
        env = np.atleast_1d(np.asarray(envelope, dtype=complex))
        if t is None:
            if dt is not None:
                t = np.arange(env.size) * dt
            else:
                t = np.zeros(env.size) if env.size == 1 else np.arange(env.size)
        t = np.asarray(t, dtype=float)
        if t.shape != env.shape:
            raise ValueError("t and envelope must have the same shape")
        self.env = env
        self.f0 = float(f0)
        self.t = t
        self.Z0 = float(Z0)
        self.label = label

    # -- basic descriptors ---------------------------------------------------
    @property
    def is_static(self) -> bool:
        return self.env.size == 1

    @property
    def dt(self):
        return self.t[1] - self.t[0] if self.t.size > 1 else None

    @property
    def magnitude(self):
        """Envelope magnitude (peak RF amplitude) vs slow time."""
        return np.abs(self.env)

    @property
    def phase(self):
        """Envelope phase (rad) vs slow time."""
        return np.angle(self.env)

    @property
    def power(self):
        """Average power into Z0 for a sinusoid of this envelope: |env|^2 / (2 Z0)."""
        return np.abs(self.env) ** 2 / (2.0 * self.Z0)

    def power_dbm(self):
        return 10.0 * np.log10(np.maximum(self.power, 1e-30) / 1e-3)

    # -- time-domain reconstruction -----------------------------------------
    def waveform(self, n_periods=5, samples_per_period=64, at=0.0):
        """Reconstruct the real RF voltage waveform over a short window.

        The window is a few carrier periods long so the microwave oscillation
        is visible.  For a modulated signal, the envelope is taken at slow time
        ``at`` (held constant across the short window).

        Returns
        -------
        (t_fast, v) : tuple of ndarray
            ``t_fast`` in seconds, ``v`` the real voltage Re{env e^{j w t}}.
        """
        env_at = self._env_at(at)
        T = 1.0 / self.f0
        n = int(n_periods * samples_per_period)
        t_fast = np.linspace(0.0, n_periods * T, n, endpoint=False)
        v = np.real(env_at * np.exp(1j * 2 * np.pi * self.f0 * t_fast))
        return t_fast, v

    def _env_at(self, at):
        """Envelope value at slow time ``at`` (nearest sample, or the scalar)."""
        if self.is_static:
            return self.env[0]
        idx = int(np.argmin(np.abs(self.t - at)))
        return self.env[idx]

    def envelope_series(self):
        """Return (t, env): the complex envelope vs slow time (modulation view)."""
        return self.t, self.env

    # -- frequency-domain reconstruction ------------------------------------
    def spectrum(self, window="hann", pad=1):
        """Narrowband RF voltage spectrum around the carrier.

        The envelope is Fourier transformed on the slow-time grid to obtain the
        modulation (baseband) spectrum, which is then placed on the carrier at
        +/- f0.  This shows the carrier line plus any modulation sidebands
        (e.g. from tip-height modulation) at the correct RF frequencies without
        having to sample the GHz carrier directly.

        Returns
        -------
        (freq, V) : tuple of ndarray
            ``freq`` in Hz (positive RF frequencies around f0), ``V`` the complex
            voltage spectral component (arbitrary units, peak-amplitude scaled).
        """
        if self.is_static or self.t.size < 2:
            # Pure CW tone: a single line at f0.
            return np.array([self.f0]), np.array([self.env[0]], dtype=complex)

        env = self.env.copy()
        N = env.size
        w = _get_window(window, N)
        cg = np.mean(w)  # coherent gain, to keep line amplitudes calibrated
        Nfft = int(N * pad)
        fb = np.fft.fftshift(np.fft.fftfreq(Nfft, d=self.dt))
        E = np.fft.fftshift(np.fft.fft(env * w, n=Nfft)) / (N * cg)
        # Baseband offset fb maps to RF frequency f0 + fb.
        freq = self.f0 + fb
        return freq, E

    def baseband_spectrum(self, window="hann", pad=1):
        """Modulation spectrum of the envelope, centered at 0 Hz (no carrier)."""
        if self.is_static or self.t.size < 2:
            return np.array([0.0]), np.array([self.env[0]], dtype=complex)
        env = self.env.copy()
        N = env.size
        w = _get_window(window, N)
        cg = np.mean(w)
        Nfft = int(N * pad)
        fb = np.fft.fftshift(np.fft.fftfreq(Nfft, d=self.dt))
        E = np.fft.fftshift(np.fft.fft(env * w, n=Nfft)) / (N * cg)
        return fb, E

    def __repr__(self):
        kind = "CW" if self.is_static else f"modulated[{self.env.size}]"
        return (f"Signal(label={self.label!r}, f0={self.f0:.4g} Hz, {kind}, "
                f"|env0|={np.abs(self.env.flat[0]):.4g})")


def _get_window(name, N):
    if name is None or name == "rect" or name == "boxcar":
        return np.ones(N)
    if name == "hann":
        return np.hanning(N)
    if name == "hamming":
        return np.hamming(N)
    if name == "blackman":
        return np.blackman(N)
    raise ValueError(f"unknown window {name!r}")
