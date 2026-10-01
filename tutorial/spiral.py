# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

from pathlib import Path
import math
import twincher

# Define stencil
class Spiral: 
    def __init__(self, n_turns = 1.0):
        self.n_p, self.n_y = 1, 2
        self.n_turns = n_turns
        self.name = f"spiral_{n_turns:3.1f}"
    def __call__(self, p, y):
        r = 1/(0.5*p[0] + 1.5)
        alpha = 0.25*math.pi + self.n_turns*math.pi*(p[0] + 1)
        y[0] = r*math.cos(alpha)
        y[1] = r*math.sin(alpha)
stencil = Spiral(n_turns=1.3)

# Create working dir
script_path = Path(__file__).resolve()
dir = Path(__file__).resolve().parent / f"output_{stencil.name}"
dir.mkdir(exist_ok=True)

# Create learner
learner = twincher.Learner(arch="hs", n_s=16, n_l=64, dir=dir)

# Generate data and initiate learner
learner.generate_data(stencil=stencil, n_grid=128)

# Prepare state output tool
fig_stride = 200
from output_tools_p1_y2 import StateMonitor
monitor = StateMonitor(stencil=stencil, dir=dir)

# Run optimization
n_iter = 5001
for iter in range(n_iter):
    learner.step()
    if iter % fig_stride == 0:
        monitor.make_plot(file_name= dir / f"{iter // fig_stride}.png", iter=iter, shuttle=learner.SH)
    print(f"\r{iter}/{n_iter}: {learner.state}", end='')

# Save result
learner.SH.save(str(dir / "final.twc"))

# Make parity plot
from output_tools_p1_y2 import make_parity_plot
make_parity_plot(dir, stencil, file_name="spiral_parity_plot-light.svg")

# Plot noise tolerance result
twincher.noise_check(
    file_name=str(dir / "final.twc"), 
    stencil=stencil, 
    output_file_name=f"{stencil.name}_noise_tol.png", 
    ax_ylim = 0.2,
    noise_max=0.1,
) 