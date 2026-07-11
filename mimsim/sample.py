"""Tip-sample interaction model.

The tip-sample admittance is modeled with the lumped-element picture of Fig. 1a
of Barber, Ma & Shen, Nat. Rev. Phys. 4, 61 (2022): a tip-sample coupling
capacitance ``C_coupling`` (the "interface" capacitance) in series with the
sample, which is itself a parallel R-C to ground.  The probe adds a large self
capacitance ``C_probe`` (~1 pF) in parallel, so the tip-sample term is a small
perturbation, exactly as in the paper.

Writing the sample as R || C with ``R = rho/g`` and ``C = eps*g`` for a geometric
length scale ``g``, the product ``R*C = rho*eps`` is geometry independent, so the
dissipative (MIM-Re) response peaks at ``rho*omega*eps ~ 1`` and the reactive
(MIM-Im) response steps up toward the conductive side -- reproducing the
canonical MIM response curve.
"""

from __future__ import annotations

import numpy as np

from .components import Load

EPS0 = 8.8541878128e-12  # vacuum permittivity, F/m


class TipSampleModel:
    """Lumped-element tip-sample admittance model.

    Parameters
    ----------
    C_probe : float
        Probe self-capacitance (F), dominates the overall probe admittance.
    C_coupling : float
        Tip-sample interface coupling capacitance (F) at contact/closest approach.
    g : float
        Geometric length scale (m) setting sample R and C: R = rho/g, C = eps*g.
        The response-curve *shape* is independent of g (only magnitude scales).
        The dissipative (MIM-Re) peak sits at ``rho*omega*eps = 1/(1 + C_coupling/(eps*g))``,
        so a small interface capacitance (C_coupling << eps*g) centers it at ~1,
        matching the paper's "contrast window centred on sigma ~ omega*eps".
    z0 : float
        Reference tip-sample gap parameter (m) for height dependence of the
        coupling capacitance: C_coupling(z) = C_coupling * z0 / (z0 + z).
    """

    def __init__(self, C_probe=1e-12, C_coupling=1e-18, g=5e-7, z0=10e-9, Q_probe=np.inf):
        self.C_probe = float(C_probe)
        self.C_coupling = float(C_coupling)
        self.g = float(g)
        self.z0 = float(z0)
        # finite probe quality factor -> a small self-loss G = w*C_probe/Q_probe.
        # Needed for a fixed matching bridge to produce real |Gamma| dips; the
        # default (inf) keeps the probe lossless for the idealized response curve.
        self.Q_probe = float(Q_probe)

    # -- admittances --------------------------------------------------------
    def coupling_cap(self, z=0.0):
        """Tip-sample coupling capacitance at gap ``z`` (m)."""
        return self.C_coupling * self.z0 / (self.z0 + z)

    def Y_tipsample(self, f, rho, eps_r=10.0, z=0.0):
        """Tip-sample admittance (S) for sample resistivity ``rho`` (Ohm*m).

        ``eps_r`` is the sample relative permittivity, ``z`` the tip-sample gap.
        """
        w = 2 * np.pi * f
        eps = eps_r * EPS0
        R = rho / self.g
        C = eps * self.g
        Cc = self.coupling_cap(z)
        # sample = R in parallel with C
        Z_sample = R / (1.0 + 1j * w * R * C)
        Z_ts = 1.0 / (1j * w * Cc) + Z_sample
        return 1.0 / Z_ts

    def Y_total(self, f, rho, eps_r=10.0, z=0.0):
        """Total probe admittance: self-capacitance (+ its loss) plus tip-sample term."""
        w = 2 * np.pi * f
        G_probe = w * self.C_probe / self.Q_probe if np.isfinite(self.Q_probe) else 0.0
        return 1j * w * self.C_probe + G_probe + self.Y_tipsample(f, rho, eps_r, z)

    # -- component factory --------------------------------------------------
    def make_load(self, rho, eps_r=10.0, z=0.0, name="tip_sample"):
        """Build a 1-port :class:`Load` for this tip-sample state.

        ``rho``, ``eps_r`` and ``z`` may each be constants or callables of slow
        time ``t`` (e.g. a height-modulated ``z(t)`` for dMIM/dz measurements).
        """
        model = self

        def Y_of(f, t):
            r = rho(t) if callable(rho) else rho
            e = eps_r(t) if callable(eps_r) else eps_r
            zz = z(t) if callable(z) else z
            return model.Y_total(f, r, e, zz)

        return Load(Y=Y_of, name=name)


def response_curve(model, f, rho_array, eps_r=10.0, z=0.0):
    """Convenience: tip-sample admittance vs resistivity (standalone, no network).

    Returns ``(rho_omega_eps, Y_ts)`` where the x-axis is the dimensionless
    ``rho*omega*eps`` used in the paper's response curve.
    """
    w = 2 * np.pi * f
    eps = eps_r * EPS0
    rho_array = np.asarray(rho_array, dtype=float)
    Y = np.array([model.Y_tipsample(f, r, eps_r, z) for r in rho_array], dtype=complex)
    return rho_array * w * eps, Y
