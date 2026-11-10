"""Cleaveform - split a 3D model into printable chunks that cleave back together.

Phase 1 is a deterministic, offline, chunk-mode pipeline: STL in, one STL per
part plus a text assembly guide out. No machine learning, no network calls, no
solver dependencies.
"""

__version__ = "0.1.0"

from .config import DEFAULT_CONFIG, Config

__all__ = ["Config", "DEFAULT_CONFIG", "__version__"]
