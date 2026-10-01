# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

COMPONENTS = {
    "arch_defaults" : {},
    "loss_terms" : {},
    "solvers" : {},
    "monitors" : {},
}

# The components of the architectures included in this package are registered on first
# use of the registry rather than at "import twincher", because they import PyTorch.
_builtins_registered = False

def _register_builtins():
    global _builtins_registered
    if _builtins_registered:
        return
    _builtins_registered = True # set before registering, since register() calls this function
    try:
        from . import hs_twincher
        hs_twincher.register_components(register)
    except BaseException:
        _builtins_registered = False
        raise

def register(category, name, ref):
    _register_builtins()
    if category not in COMPONENTS:
        raise RuntimeError(f"unknown component category {category}")
    if name in COMPONENTS[category]:
        raise RuntimeError(f"component {name} has already been registered")
    COMPONENTS[category][name] = ref

def components(category):
    """Return the dictionary {name: component} of the registered components of a category."""
    _register_builtins()
    return COMPONENTS[category]

def print_reg():
    _register_builtins()
    print(COMPONENTS)
