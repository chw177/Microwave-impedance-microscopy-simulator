"""Impedance-matching network design for the MIM probe.

The MIM probe is dominated by a self-capacitance ``C_probe`` (~1 pF), with a small
loss giving a finite quality factor ``Q``.  A lossless reactive probe reflects
everything (|Gamma| = 1); the matching network transforms the probe admittance
close to the line impedance ``Z0`` so that a small tip-sample admittance change
produces a large change in the reflection coefficient -- i.e. it maximizes the
measurement sensitivity ``dGamma/dY`` (paper Box 1).

``resonant_match`` builds the paper's canonical scheme: a shunt inductor that
resonates out ``C_probe`` at ``f0`` (turning the probe into a real resistance
``R_loss = Q/(omega*C_probe)``), followed by a quarter-wave transformer of
impedance ``sqrt(Z0 * R_loss)`` that steps ``R_loss`` down to ``Z0``.
"""

from __future__ import annotations

import numpy as np

from .components import MatchingNetwork, shunt_L, tline

C_LIGHT = 299792458.0


def resonant_match(f0, C_probe=1e-12, Q=40.0, Z0=50.0, eps_eff=1.0,
                   L_scale=1.0, Zqw_scale=1.0, mismatch_db=None, name="match"):
    """Design a quarter-wave resonant match for a capacitive MIM probe.

    A perfectly tuned match (all scales = 1) gives |Gamma| -> 0 at ``f0``.  Real
    matches are never perfect: component tolerances, thermal drift and imperfect
    knowledge of the probe leave a residual mismatch.  Two knobs model this:

    - ``L_scale`` detunes the resonator (shunt L off by this factor), which shifts
      the |Gamma| dip *in frequency* -- the drift the paper warns about.
    - ``Zqw_scale`` scales the transformer impedance, leaving a residual mismatch
      *at* ``f0`` (the transformed resistance becomes ``Z0 * Zqw_scale**2``).

    As a convenience, ``mismatch_db`` sets ``Zqw_scale`` automatically so that
    ``|Gamma|`` at ``f0`` is approximately that value (e.g. ``mismatch_db=-25``).

    Parameters
    ----------
    f0 : float
        Match (resonant) frequency in Hz.
    C_probe : float
        Probe self-capacitance in F.
    Q : float
        Assumed probe/resonator quality factor; sets the resonator loss
        resistance ``R_loss = Q/(omega*C_probe)`` that the transformer matches.
    Z0 : float
        Line impedance in ohms.
    eps_eff : float
        Effective permittivity of the quarter-wave line (sets its length).
    L_scale, Zqw_scale : float
        Fractional error on the shunt inductor and transformer impedance
        (1.0 = ideal).
    mismatch_db : float or None
        If given, overrides ``Zqw_scale`` to target this |Gamma| (dB) at ``f0``.

    Returns
    -------
    MatchingNetwork
        2-port network; connect its ``"in"`` port to the line side and its
        ``"out"`` port to the probe.
    """
    w = 2 * np.pi * f0
    R_loss = Q / (w * C_probe)                  # resonator loss resistance
    if mismatch_db is not None:
        # |Gamma| = (r-1)/(r+1) with r = Zqw_scale**2 (transformed R / Z0)
        g = 10.0 ** (mismatch_db / 20.0)
        Zqw_scale = np.sqrt((1.0 + g) / (1.0 - g))
    L = L_scale / (w ** 2 * C_probe)            # shunt L resonates C_probe at f0
    Z_qw = Zqw_scale * np.sqrt(Z0 * R_loss)     # quarter-wave transformer impedance
    length = C_LIGHT / (4.0 * f0 * np.sqrt(eps_eff))  # physical lambda/4 length
    # cascade order in (line) -> out (probe): transformer, then shunt L across probe
    net = MatchingNetwork([tline(Z_qw, length, eps_eff), shunt_L(L)], Z0=Z0, name=name)
    net.design = dict(L=L, R_loss=R_loss, Z_qw=Z_qw, length=length, f0=f0, Q=Q,
                      L_scale=L_scale, Zqw_scale=Zqw_scale)
    return net
