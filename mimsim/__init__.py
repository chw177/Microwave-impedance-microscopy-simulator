"""mimsim: a traveling-wave / S-parameter simulator for microwave impedance
microscopy (MIM) electronics.

Model the full MIM signal chain (source, directional couplers, voltage-controlled
attenuators and phase shifters, impedance-matching network, tip-sample load,
amplifier, IQ mixer) as a scattering network solved at the carrier frequency,
and probe the time-domain and frequency-domain signal at the input or output of
any component.
"""

from .signal import Signal
from .network import Network, Component, Port, Result, SolveResult
from .components import (
    Source,
    Attenuator,
    PhaseShifter,
    DirectionalCoupler,
    Amplifier,
    Termination,
    Load,
    MatchingNetwork,
    IQMixer,
    DemodResult,
    series_Z, shunt_Y, series_L, series_C, series_R,
    shunt_L, shunt_C, shunt_R, tline,
    abcd_to_s, db_to_lin_amp,
)
from .sample import TipSampleModel, response_curve, EPS0
from .matching import resonant_match, lambda4_bridge

__version__ = "0.1.0"

__all__ = [
    "Signal", "Network", "Component", "Port", "Result", "SolveResult",
    "Source", "Attenuator", "PhaseShifter", "DirectionalCoupler", "Amplifier",
    "Termination", "Load", "MatchingNetwork", "IQMixer", "DemodResult",
    "series_Z", "shunt_Y", "series_L", "series_C", "series_R",
    "shunt_L", "shunt_C", "shunt_R", "tline", "abcd_to_s", "db_to_lin_amp",
    "TipSampleModel", "response_curve", "EPS0", "resonant_match", "lambda4_bridge",
]
