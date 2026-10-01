# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

from pathlib import Path
import twincher
import numpy as np

# Define stencil
class DoubleGaussian:
    def __init__(self, n_y: int):
        self.n_p = 2
        self.n_y = n_y
        self.x = np.linspace(-1, 1, n_y)
        self.name = f"double_gaussian"
    def __call__(self, p, y):
        y[:]  = 0.5*np.exp(-4*(self.x - p[0])**2)
        y[:] += 0.25*np.exp(-9*(self.x - p[1])**2)
    def batch_call(self, p, y):
        backend = type(p).__module__.split(".")[0]
        if backend == "numpy":
            y[:, :]  = 0.5*np.exp(-4*(self.x[None, :] - p[:, 0, None])**2)
            y[:, :] += 0.25*np.exp(-9*(self.x[None, :] - p[:, 1, None])**2)
        elif backend == "torch":
            import torch
            if not hasattr(self, "x_torch") or self.x_torch.device != y.device or self.x_torch.dtype != y.dtype:
                self.x_torch = torch.tensor(self.x, dtype=y.dtype, device=y.device)
            y[:, :]  = 0.5*torch.exp(-4*(self.x_torch[None, :] - p[:, 0, None])**2)
            y[:, :] += 0.25*torch.exp(-9*(self.x_torch[None, :] - p[:, 1, None])**2)
    def compute(self, p, y, dy_dp, dp_eval = 1e-4):
        self.batch_call(p, y)
        for i_p in range(self.n_p):
            p[:, i_p] += dp_eval
            self.batch_call(p, dy_dp[:, :, i_p])
            dy_dp[:, :, i_p] -= y[:]
            dy_dp[:, :, i_p] /= dp_eval
            p[:, i_p] -= dp_eval

# Create working dir
stencil = DoubleGaussian(n_y=32)
n_p, n_y = stencil.n_p, stencil.n_y
script_path = Path(__file__).resolve()
dir = Path(__file__).resolve().parent / f"output_{stencil.name}"
dir.mkdir(exist_ok=True)

# Plot example
from output_tools_p2 import plot_example
plot_example(stencil=stencil, file_name=dir / "example.png")

# Training takes a while: it is skipped if a trained twincher has already been saved
# (delete the file final.twc to train again)
if not (dir / "final.twc").exists():
    # Create learner, generate data, initiate
    learner = twincher.Learner(arch="hs", n_s=2*n_y, n_l=64, dir=dir)
    learner.set_stencil(stencil=stencil)
    #learner.generate_data(stencil=stencil, n_grid=32)
    learner.init()

    # Prepare state output tool
    fig_stride = 200
    monitor = twincher.monitor(
        name="hs_monitor_p2",
        SH=learner.SH,
        stencil=stencil, 
        n_g=32, dir=dir, color_scheme="dark",
    )

    # Run optimization
    for iter in range(13001):
        learner.step()
        print(f"\r{iter}: {learner.loss_detailed}", end="")
        if iter % 200 == 0:
            print("\n", learner.state)
        if iter % fig_stride == 0:
            monitor.draw(learner.SH, file_name=f"{(iter//fig_stride):03d}.png")

    learner.SH.save(str(dir) + f"/final.twc")

# Test solution
solver = twincher.solver(file_name=str(dir) + "/final.twc", n_c = 5*1024)

rng = np.random.default_rng(42)
p_true = rng.uniform(-1, 1, size=(solver.n_c, n_p))
y_true = np.empty((solver.n_c, n_y), dtype=np.float64)
stencil.batch_call(p_true, y_true)
solver.start_solution(y_true)

for _ in range(6):
    stencil.compute(solver.p, solver.y, solver.dy_dp)
    solver.step(y=solver.y, dy_dp=solver.dy_dp)

p_sol = solver.p.detach().cpu().numpy()
diffs = np.linalg.norm(p_true - p_sol, axis=1)
print("\nMax difference in p:", np.max(diffs))
print("Average difference in p:", np.mean(diffs))
