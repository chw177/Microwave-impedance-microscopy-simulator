"""Concrete microwave components for the MIM signal chain.

Every component exposes an S-matrix at a given carrier frequency and slow time.
Voltage-controlled parts (attenuator, phase shifter) map an applied control
voltage through a user-definable response curve; both the voltage and any other
parameter may be a constant or a callable ``param(t)`` for time modulation.
"""

from __future__ import annotations

import numpy as np

from .network import Component


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def db_to_lin_amp(db):
    """Convert a dB value to a linear amplitude ratio (20*log10)."""
    return 10.0 ** (db / 20.0)


def abcd_to_s(A, B, C, D, Z0):
    """Convert a 2-port ABCD matrix to a scattering matrix at impedance Z0."""
    denom = A + B / Z0 + C * Z0 + D
    S11 = (A + B / Z0 - C * Z0 - D) / denom
    S12 = 2.0 * (A * D - B * C) / denom
    S21 = 2.0 / denom
    S22 = (-A + B / Z0 - C * Z0 + D) / denom
    return np.array([[S11, S12], [S21, S22]], dtype=complex)


# ---------------------------------------------------------------------------
# source
# ---------------------------------------------------------------------------
class Source(Component):
    """Microwave source (1-port) with tunable amplitude, phase and frequency.

    The source emits an outgoing wave ``bs = amplitude * exp(j*phase)`` and
    presents an internal reflection ``gamma_s`` (0 for an ideal matched source).
    ``amplitude`` and ``phase`` may be callables of slow time for modulation.
    ``f0`` sets the carrier frequency for the whole network.
    """

    n_ports = 1

    def __init__(self, f0, amplitude=1.0, phase=0.0, gamma_s=0.0, name="source"):
        super().__init__(name)
        self.f0 = float(f0)
        self.amplitude = amplitude
        self.phase = phase
        self.gamma_s = gamma_s

    def set_power_dbm(self, dbm):
        """Set the emitted wave amplitude from an available power in dBm."""
        p_watt = 1e-3 * 10.0 ** (dbm / 10.0)
        self.amplitude = np.sqrt(2.0 * p_watt)  # peak-amplitude wave power = |b|^2/2
        return self

    def s_matrix(self, f, t):
        return np.array([[self._eval(self.gamma_s, t)]], dtype=complex)

    def b_source(self, f, t):
        amp = self._eval(self.amplitude, t)
        ph = self._eval(self.phase, t)
        return np.array([amp * np.exp(1j * ph)], dtype=complex)


# ---------------------------------------------------------------------------
# voltage-controlled attenuator and phase shifter
# ---------------------------------------------------------------------------
class Attenuator(Component):
    """Voltage-controlled 2-port attenuator (matched, reciprocal).

    The attenuation in dB is ``insertion_db + response(voltage)``.  By default
    ``response`` is 1 dB per volt; pass any callable to model a real device.
    """

    n_ports = 2
    port_names = ["in", "out"]

    def __init__(self, voltage=0.0, response=None, insertion_db=0.0, name="attenuator"):
        super().__init__(name)
        self.voltage = voltage
        self.response = response if response is not None else (lambda v: v)
        self.insertion_db = insertion_db

    def attenuation_db(self, t=0.0):
        v = self._eval(self.voltage, t)
        return self.insertion_db + self.response(v)

    def s_matrix(self, f, t):
        tr = db_to_lin_amp(-self.attenuation_db(t))
        return np.array([[0.0, tr], [tr, 0.0]], dtype=complex)


class PhaseShifter(Component):
    """Voltage-controlled 2-port phase shifter (matched, reciprocal).

    The phase shift in radians is ``response(voltage)``.  Default response is
    1 rad per volt.  An optional insertion loss (dB) can be included.
    """

    n_ports = 2
    port_names = ["in", "out"]

    def __init__(self, voltage=0.0, response=None, insertion_db=0.0, name="phase_shifter"):
        super().__init__(name)
        self.voltage = voltage
        self.response = response if response is not None else (lambda v: v)
        self.insertion_db = insertion_db

    def phase_rad(self, t=0.0):
        v = self._eval(self.voltage, t)
        return self.response(v)

    def s_matrix(self, f, t):
        loss = db_to_lin_amp(-self.insertion_db)
        tr = loss * np.exp(1j * self.phase_rad(t))
        return np.array([[0.0, tr], [tr, 0.0]], dtype=complex)


# ---------------------------------------------------------------------------
# directional coupler
# ---------------------------------------------------------------------------
class DirectionalCoupler(Component):
    """Ideal-ish 4-port directional coupler.

    Ports: 0 = input, 1 = through, 2 = coupled, 3 = isolated.
    ``coupling_db`` sets the coupled-port level; ``directivity_db`` sets the
    finite isolation-port leakage.  Coupled port carries the usual +90 deg (j).
    """

    n_ports = 4
    port_names = ["in", "through", "coupled", "isolated"]

    def __init__(self, coupling_db=20.0, directivity_db=np.inf,
                 insertion_db=0.0, name="coupler"):
        super().__init__(name)
        self.coupling_db = coupling_db
        self.directivity_db = directivity_db
        self.insertion_db = insertion_db

    def s_matrix(self, f, t):
        k = db_to_lin_amp(-self.coupling_db)          # coupled amplitude
        thru = db_to_lin_amp(-self.insertion_db) * np.sqrt(max(1.0 - k * k, 0.0))
        c = 1j * k                                    # coupled port (90 deg)
        iso = c * db_to_lin_amp(-self.directivity_db)  # leakage to isolated port
        S = np.array([
            [0.0, thru, c,   iso],
            [thru, 0.0, iso, c  ],
            [c,   iso, 0.0, thru],
            [iso, c,   thru, 0.0],
        ], dtype=complex)
        return S


# ---------------------------------------------------------------------------
# amplifier
# ---------------------------------------------------------------------------
class Amplifier(Component):
    """Unilateral 2-port amplifier (matched, infinite reverse isolation).

    Ports: 0 = in, 1 = out.  Gain set in dB with an optional phase.
    """

    n_ports = 2
    port_names = ["in", "out"]

    def __init__(self, gain_db=20.0, phase=0.0, name="amplifier"):
        super().__init__(name)
        self.gain_db = gain_db
        self.phase = phase

    def s_matrix(self, f, t):
        g = db_to_lin_amp(self._eval(self.gain_db, t)) * np.exp(1j * self._eval(self.phase, t))
        return np.array([[0.0, 0.0], [g, 0.0]], dtype=complex)


# ---------------------------------------------------------------------------
# terminations and loads
# ---------------------------------------------------------------------------
class Termination(Component):
    """1-port termination defined by a reflection coefficient."""

    n_ports = 1

    def __init__(self, gamma=0.0, name="termination"):
        super().__init__(name)
        self.gamma = gamma

    @classmethod
    def matched(cls, name="match"):
        return cls(0.0, name)

    @classmethod
    def open(cls, name="open"):
        return cls(1.0, name)

    @classmethod
    def short(cls, name="short"):
        return cls(-1.0, name)

    def s_matrix(self, f, t):
        return np.array([[self._eval(self.gamma, t)]], dtype=complex)


class Load(Component):
    """1-port load defined by an impedance or admittance (possibly time-varying).

    Provide exactly one of ``Z`` (ohms) or ``Y`` (siemens); either may be a
    constant, a callable ``p(t)``, or a callable ``p(f, t)``.
    """

    n_ports = 1

    def __init__(self, Z=None, Y=None, Z0=50.0, name="load"):
        super().__init__(name)
        if (Z is None) == (Y is None):
            raise ValueError("provide exactly one of Z or Y")
        self.Z = Z
        self.Y = Y
        self.Z0 = float(Z0)

    def _admittance(self, f, t):
        if self.Y is not None:
            return _eval_ft(self.Y, f, t)
        Z = _eval_ft(self.Z, f, t)
        return 1.0 / Z

    def s_matrix(self, f, t):
        Y = self._admittance(f, t)
        Y0 = 1.0 / self.Z0
        gamma = (Y0 - Y) / (Y0 + Y)
        return np.array([[gamma]], dtype=complex)


def _eval_ft(param, f, t):
    """Evaluate a parameter that may be constant, p(t), or p(f, t)."""
    if callable(param):
        try:
            return param(f, t)
        except TypeError:
            return param(t)
    return param


# ---------------------------------------------------------------------------
# impedance-matching network (2-port ABCD cascade)
# ---------------------------------------------------------------------------
class MatchingNetwork(Component):
    """2-port network built from a cascade of ABCD elements.

    ``elements`` is a list of callables ``elem(f) -> 2x2 ABCD ndarray``.  Use the
    ``series_*`` / ``shunt_*`` / ``tline`` factory functions below to build them.
    Ports: 0 = in (line side), 1 = out (probe side).
    """

    n_ports = 2
    port_names = ["in", "out"]

    def __init__(self, elements, Z0=50.0, name="matching"):
        super().__init__(name)
        self.elements = list(elements)
        self.Z0 = float(Z0)

    def abcd(self, f):
        M = np.eye(2, dtype=complex)
        for elem in self.elements:
            M = M @ elem(f)
        return M

    def s_matrix(self, f, t):
        M = self.abcd(f)
        return abcd_to_s(M[0, 0], M[0, 1], M[1, 0], M[1, 1], self.Z0)


def series_Z(Zfunc):
    """Series impedance element. ``Zfunc`` is Z(f) in ohms (or a constant)."""
    zf = Zfunc if callable(Zfunc) else (lambda f: Zfunc)
    return lambda f: np.array([[1.0, zf(f)], [0.0, 1.0]], dtype=complex)


def shunt_Y(Yfunc):
    """Shunt admittance element. ``Yfunc`` is Y(f) in siemens (or a constant)."""
    yf = Yfunc if callable(Yfunc) else (lambda f: Yfunc)
    return lambda f: np.array([[1.0, 0.0], [yf(f), 1.0]], dtype=complex)


def series_L(L):
    return series_Z(lambda f: 1j * 2 * np.pi * f * L)


def series_C(C):
    return series_Z(lambda f: 1.0 / (1j * 2 * np.pi * f * C))


def series_R(R):
    return series_Z(lambda f: R)


def shunt_L(L):
    return shunt_Y(lambda f: 1.0 / (1j * 2 * np.pi * f * L))


def shunt_C(C):
    return shunt_Y(lambda f: 1j * 2 * np.pi * f * C)


def shunt_R(R):
    return shunt_Y(lambda f: 1.0 / R)


def tline(Z0_line, length, eps_eff=1.0, loss_np_per_m=0.0):
    """Lossy transmission-line element of physical ``length`` (m)."""
    c = 299792458.0

    def elem(f):
        beta = 2 * np.pi * f * np.sqrt(eps_eff) / c
        gamma = loss_np_per_m + 1j * beta
        gl = gamma * length
        A = np.cosh(gl)
        B = Z0_line * np.sinh(gl)
        C = np.sinh(gl) / Z0_line
        D = np.cosh(gl)
        return np.array([[A, B], [C, D]], dtype=complex)

    return elem


# ---------------------------------------------------------------------------
# IQ mixer / demodulator (detector)
# ---------------------------------------------------------------------------
class IQMixer(Component):
    """IQ demodulator: mixes the RF port against the LO port to give I and Q.

    In the RF network both ports are matched sinks (they absorb).  The actual
    demodulation is done after the solve by :meth:`demodulate`, using the
    incident waves at the RF and LO ports.  This is exactly the in-phase /
    quadrature detection that yields the MIM-Re and MIM-Im channels.

    Ports: 0 = rf, 1 = lo.
    """

    n_ports = 2
    port_names = ["rf", "lo"]

    def __init__(self, conv_gain=1.0, name="iq_mixer"):
        super().__init__(name)
        self.conv_gain = conv_gain

    def s_matrix(self, f, t):
        # both ports matched: absorb everything, no coupling
        return np.zeros((2, 2), dtype=complex)

    def demodulate(self, result, normalize_lo=True):
        """Compute the baseband I (MIM-Re) and Q (MIM-Im) channels.

        Returns a :class:`DemodResult`.
        """
        rf = result.incident(self.port("rf"))
        lo = result.incident(self.port("lo"))
        ref = lo.env
        if normalize_lo:
            ref = ref / np.maximum(np.abs(lo.env), 1e-30)
        base = self.conv_gain * rf.env * np.conj(ref)
        return DemodResult(result.t, base, f0=result.f0)


class DemodResult:
    """Baseband output of an IQ mixer: the MIM-Re / MIM-Im channels vs slow time."""

    def __init__(self, t, complex_baseband, f0=None):
        self.t = np.asarray(t, dtype=float)
        self.complex = np.atleast_1d(np.asarray(complex_baseband, dtype=complex))
        self.f0 = f0

    @property
    def I(self):
        """In-phase channel (MIM-Re)."""
        return np.real(self.complex)

    @property
    def Q(self):
        """Quadrature channel (MIM-Im)."""
        return np.imag(self.complex)

    # convenient aliases matching the paper's channel names
    re = I
    im = Q

    @property
    def magnitude(self):
        return np.abs(self.complex)

    @property
    def phase(self):
        return np.angle(self.complex)

    def rotate(self, phi):
        """Return a copy rotated by ``phi`` radians in the I/Q plane.

        Rotating corresponds to adjusting the LO (reference-arm) phase so the
        dissipative response lands in MIM-Re and the reactive response in MIM-Im.
        """
        return DemodResult(self.t, self.complex * np.exp(1j * phi), f0=self.f0)

    def align_to(self, reference_complex):
        """Rotate so this signal aligns with a known complex reference.

        ``reference_complex`` is a signal whose real part should map to MIM-Re
        (e.g. the modelled tip-sample admittance change delta-Y).  Returns a
        rotated copy plus the applied phase.
        """
        ref = np.atleast_1d(np.asarray(reference_complex, dtype=complex))
        phi = -np.angle(np.sum(self.complex * np.conj(ref)))
        return self.rotate(phi), phi

    def __repr__(self):
        return (f"DemodResult(N={self.complex.size}, "
                f"MIM-Re={self.I.flat[0]:.4g}, MIM-Im={self.Q.flat[0]:.4g})")
