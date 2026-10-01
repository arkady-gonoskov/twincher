# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

from . import registry

def monitor(*args, **kwargs):
    if len(args) > 0:
        monitor_name = args[0]
        args = args[1:]
    elif "name" in kwargs:
        monitor_name = kwargs["name"]
        del kwargs["name"]
    else:
        raise ValueError("Missing name")

    if monitor_name not in registry.components("monitors"):
        raise RuntimeError(f"Unknown monitor name {monitor_name}")
    else:
        return registry.components("monitors")[monitor_name](*args, **kwargs)