# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

"""Twincher: trainable diffeomorphic representations for solving inverse problems."""

from importlib.metadata import PackageNotFoundError, version as _version
from typing import TYPE_CHECKING

try:
    __version__ = _version("twincher")
except PackageNotFoundError: # sources used without installation
    __version__ = "unknown"

from .shuttle import info, Shuttle, load_metadata
from .registry import register, print_reg
from .solver import solver
from .verifier import Verifier
from .noise_check import noise_check
from .monitor import monitor
from .time_monitor import TimeMonitor
from .tw_util import estimate_J

# Learner requires PyTorch, which takes a few seconds to import. It is therefore imported
# on first access (PEP 562), so that "import twincher" stays fast when only the
# computational core (Shuttle) is used.
if TYPE_CHECKING:
    from .learner import Learner

def __getattr__(name):
    if name == "Learner":
        from .learner import Learner
        return Learner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | {"Learner"})

__all__ = [
    "info",
    "Shuttle",
    "load_metadata",
    "register",
    "print_reg",
    "Learner",
    "solver",
    "Verifier",
    "noise_check",
    "monitor",
    "TimeMonitor",
    "estimate_J",
]
