"""Microwave network core: components, ports and the traveling-wave solver.

The network is solved in the frequency domain at a single carrier frequency f0
using normalized scattering waves.  Every port carries an incoming wave ``a``
and an outgoing wave ``b`` (normalized to the reference impedance Z0), related
per component by its scattering matrix::

    b_comp = S_comp @ a_comp + b_source_comp     # b_source is nonzero for sources

Ideal connections between ports simply swap outgoing and incoming waves::

    a_i = b_partner(i)

Collecting every port into global vectors and writing the connection map as a
matrix P (a_i = sum_j P[i,j] b_j), the whole linear network is::

    b = S_global @ (P @ b) + b_src   =>   (I - S_global @ P) b = b_src

which is a single linear solve.  Unconnected ports are treated as matched
terminations (they absorb: a_i = 0, i.e. row i of P is zero).

Time-varying control (voltage-driven attenuators / phase shifters, modulated
tip-sample load, ...) is handled by re-solving on a slow-time grid; each
component's ``s_matrix``/``b_source`` receive the slow time ``t``.
"""

from __future__ import annotations

import numpy as np

from .signal import Signal


class Port:
    """A reference to one port of a component."""

    __slots__ = ("component", "index")

    def __init__(self, component, index):
        self.component = component
        self.index = index

    def __repr__(self):
        return f"Port({self.component.name}[{self.index}])"


class Component:
    """Base class for all N-port microwave components.

    Subclasses implement :meth:`s_matrix` (and optionally :meth:`b_source`).
    Both receive the carrier frequency ``f`` (Hz) and slow time ``t`` (s) so a
    component can be dispersive and/or time-modulated.
    """

    n_ports = 1
    #: optional list of port names, e.g. ["in", "out"]
    port_names: list | None = None

    def __init__(self, name):
        self.name = name

    # -- interface to override ----------------------------------------------
    def s_matrix(self, f, t):
        raise NotImplementedError

    def b_source(self, f, t):
        return np.zeros(self.n_ports, dtype=complex)

    # -- port access --------------------------------------------------------
    def port(self, key=0):
        """Return a Port by integer index or by name."""
        if isinstance(key, str):
            if not self.port_names or key not in self.port_names:
                raise KeyError(f"{self.name} has no port named {key!r}")
            idx = self.port_names.index(key)
        else:
            idx = int(key)
        if not (0 <= idx < self.n_ports):
            raise IndexError(f"{self.name}: port {idx} out of range")
        return Port(self, idx)

    def __getitem__(self, key):
        return self.port(key)

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r})"

    # -- helper for time-varying parameters ---------------------------------
    @staticmethod
    def _eval(param, t):
        """Evaluate a parameter that may be a constant or a callable of time."""
        return param(t) if callable(param) else param


class SolveResult:
    """Per-port wave amplitudes from a single-time-point solve."""

    def __init__(self, f, t, Z0, a, b, gid, ports_of):
        self.f = f
        self.t = t
        self.Z0 = Z0
        self._a = a          # global incoming-wave vector
        self._b = b          # global outgoing-wave vector
        self._gid = gid      # (component, index) -> global id
        self._ports_of = ports_of

    def _key(self, port):
        return (id(port.component), port.index)

    def a(self, port):
        return self._a[self._gid[self._key(port)]]

    def b(self, port):
        return self._b[self._gid[self._key(port)]]

    def voltage(self, port):
        i = self._gid[self._key(port)]
        return np.sqrt(self.Z0) * (self._a[i] + self._b[i])

    def current(self, port):
        i = self._gid[self._key(port)]
        return (self._a[i] - self._b[i]) / np.sqrt(self.Z0)


class Result:
    """Solution over a slow-time grid, with per-port :class:`Signal` extraction."""

    def __init__(self, f0, t_grid, Z0, a_hist, b_hist, gid):
        self.f0 = f0
        self.t = np.asarray(t_grid, dtype=float)
        self.Z0 = Z0
        self._a = a_hist          # shape (Ntime, Mports) complex
        self._b = b_hist
        self._gid = gid

    def _idx(self, port):
        return self._gid[(id(port.component), port.index)]

    def _wrap(self, arr, port, label):
        return Signal(arr, f0=self.f0, t=self.t, Z0=self.Z0, label=label)

    def incident(self, port):
        """Incoming (incident) wave signal at ``port``."""
        i = self._idx(port)
        return self._wrap(self._a[:, i], port, f"{port.component.name}[{port.index}].a")

    def reflected(self, port):
        """Outgoing (reflected/emerging) wave signal at ``port``."""
        i = self._idx(port)
        return self._wrap(self._b[:, i], port, f"{port.component.name}[{port.index}].b")

    def voltage(self, port):
        """Total node voltage signal at ``port``: sqrt(Z0)*(a+b)."""
        i = self._idx(port)
        v = np.sqrt(self.Z0) * (self._a[:, i] + self._b[:, i])
        return self._wrap(v, port, f"{port.component.name}[{port.index}].V")

    def current(self, port):
        i = self._idx(port)
        cur = (self._a[:, i] - self._b[:, i]) / np.sqrt(self.Z0)
        return self._wrap(cur, port, f"{port.component.name}[{port.index}].I")

    def probe(self, port):
        """Return a dict of all wave signals at a port: {a, b, V, I}."""
        return {
            "a": self.incident(port),
            "b": self.reflected(port),
            "V": self.voltage(port),
            "I": self.current(port),
        }


class Network:
    """Container of components + connections with the traveling-wave solver."""

    def __init__(self, Z0=50.0, f0=None):
        self.Z0 = float(Z0)
        self.f0 = f0
        self.components: list[Component] = []
        self.connections: list[tuple[Port, Port]] = []
        self._detectors = []  # components with post-solve processing (e.g. mixers)

    # -- construction -------------------------------------------------------
    def add(self, *components):
        for c in components:
            if c not in self.components:
                self.components.append(c)
        return components[0] if len(components) == 1 else components

    def connect(self, port_a, port_b):
        port_a = self._as_port(port_a)
        port_b = self._as_port(port_b)
        for p in (port_a, port_b):
            if p.component not in self.components:
                self.add(p.component)
        self.connections.append((port_a, port_b))

    @staticmethod
    def _as_port(x):
        if isinstance(x, Port):
            return x
        if isinstance(x, Component):
            return x.port(0)
        raise TypeError(f"expected Port or Component, got {type(x)}")

    def register_detector(self, comp):
        """Register a component whose output is computed after the RF solve."""
        if comp not in self._detectors:
            self._detectors.append(comp)

    # -- indexing -----------------------------------------------------------
    def _build_index(self):
        gid = {}
        start = {}
        m = 0
        for c in self.components:
            start[id(c)] = m
            for k in range(c.n_ports):
                gid[(id(c), k)] = m + k
            m += c.n_ports
        return gid, start, m

    def _build_permutation(self, gid, m):
        # partner[i] = global id of the port wired to port i, or -1 if matched.
        partner = -np.ones(m, dtype=int)
        for pa, pb in self.connections:
            i = gid[(id(pa.component), pa.index)]
            j = gid[(id(pb.component), pb.index)]
            if partner[i] != -1 or partner[j] != -1:
                raise ValueError("a port is connected more than once")
            partner[i] = j
            partner[j] = i
        P = np.zeros((m, m), dtype=complex)
        for i in range(m):
            if partner[i] >= 0:
                P[i, partner[i]] = 1.0
        return P

    def _carrier_freq(self, f):
        if f is not None:
            return float(f)
        if self.f0 is not None:
            return float(self.f0)
        # look for a component exposing an f0 attribute (a source)
        for c in self.components:
            if getattr(c, "f0", None) is not None:
                return float(c.f0)
        raise ValueError("no carrier frequency: set Network.f0 or a Source.f0, "
                         "or pass f= to solve/run")

    # -- solving ------------------------------------------------------------
    def _solve_one(self, f, t, gid, m, P):
        S = np.zeros((m, m), dtype=complex)
        bsrc = np.zeros(m, dtype=complex)
        for c in self.components:
            s0 = gid[(id(c), 0)]
            n = c.n_ports
            S[s0:s0 + n, s0:s0 + n] = c.s_matrix(f, t)
            bsrc[s0:s0 + n] = c.b_source(f, t)
        A = np.eye(m, dtype=complex) - S @ P
        b = np.linalg.solve(A, bsrc)
        a = P @ b
        return a, b

    def solve(self, t=0.0, f=None):
        """Solve the network at a single slow-time point. Returns SolveResult."""
        f = self._carrier_freq(f)
        gid, start, m = self._build_index()
        P = self._build_permutation(gid, m)
        a, b = self._solve_one(f, t, gid, m, P)
        return SolveResult(f, t, self.Z0, a, b, gid, self.components)

    def run(self, t_grid=None, f=None):
        """Solve over a slow-time grid (or a single point).  Returns Result.

        Parameters
        ----------
        t_grid : array_like or None
            Slow-time samples (seconds).  ``None`` gives a single static solve.
        f : float or None
            Carrier frequency; defaults to the network / source f0.
        """
        f = self._carrier_freq(f)
        gid, start, m = self._build_index()
        P = self._build_permutation(gid, m)

        if t_grid is None:
            t_grid = np.array([0.0])
        else:
            t_grid = np.atleast_1d(np.asarray(t_grid, dtype=float))

        Nt = t_grid.size
        a_hist = np.empty((Nt, m), dtype=complex)
        b_hist = np.empty((Nt, m), dtype=complex)
        for k, tk in enumerate(t_grid):
            a, b = self._solve_one(f, tk, gid, m, P)
            a_hist[k] = a
            b_hist[k] = b

        self.f0 = f
        return Result(f, t_grid, self.Z0, a_hist, b_hist, gid)

    # -- frequency sweep ----------------------------------------------------
    def sweep_frequency(self, freqs, port, quantity="b", t=0.0):
        """Sweep carrier frequency and read one scalar quantity at a port.

        Useful for matching-network |S| curves.  Returns (freqs, values).
        """
        gid, start, m = self._build_index()
        P = self._build_permutation(gid, m)
        vals = np.empty(len(freqs), dtype=complex)
        idx = gid[(id(port.component), port.index)]
        for k, fk in enumerate(freqs):
            a, b = self._solve_one(fk, t, gid, m, P)
            if quantity == "a":
                vals[k] = a[idx]
            elif quantity == "b":
                vals[k] = b[idx]
            elif quantity == "V":
                vals[k] = np.sqrt(self.Z0) * (a[idx] + b[idx])
            else:
                raise ValueError("quantity must be 'a', 'b' or 'V'")
        return np.asarray(freqs, dtype=float), vals
