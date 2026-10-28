"""Does Adam's momentum drive the explosions? (hypothesis and tests)

Hypothesis: deep inside the region where det J > det_margin almost everywhere in
[-R, R]^n_p, the det loss gives almost no gradient, and over many iterations Adam (beta_1
0.99) accumulates momentum from the trimmer alone; when the state point leaves the
region, that momentum carries it far beyond the admissible determinants before the det
loss can turn it around: an explosion. A lower learning rate (as after a rollback) has
less inertia per distance.

1. Replays (identical configuration and seeds, so the runs repeat the original ones up
   to the explosion; the checkpoint of the stride before the explosion is kept as
   pre_explosion_<iter>.pt):
   - replay_l384: y0_full.py l384 (ChartY0, collapse at a fixed R 0.86 at iteration
     10000);
   - replay_y2: train_charts_fb.py Y2 (explosion at the step to a larger R at iteration
     25000).
2. From each pre-explosion state, without rollback, 4000-5000 iterations of:
   - ctrl: unchanged (beta_1 0.99): the control, should explode as in the original;
   - b09: beta_1 0.9;
   - reset: momentum reset when more than 0.1 % of a batch has det < 0 (parameters back
     to those before the previous update, Adam's first moment zeroed, second kept, lr x
     0.5 and then x 1.1 per iteration back to nominal): the first version of the reset
     (chart_training.FirstReset), revised afterwards (option momentum_reset of the hs
     loss).

Usage:
    nohup python -u adam_momentum.py replay_l384 replay_y2 > output_momentum/a.log 2>&1 &
    nohup python -u adam_momentum.py l384_ctrl l384_b09 l384_reset > ... &
    python adam_momentum.py --summary
"""

import json
import sys
from pathlib import Path

import torch

from chart_training import summary_table, train
from train_charts_fb import RUNS as CHARTS
from y0_full import RUNS as Y0

HERE = Path(__file__).resolve().parent
OUT = HERE / "output_momentum"
REPLAYS = {
    "replay_l384": {**Y0["l384"], "max_iters": 10500},
    "replay_y2": {**CHARTS["Y2"], "max_iters": 25200},
}
VARIANTS = {
    "ctrl": {},
    "b09": {"adam_betas": [0.9, 0.999]},
    "reset": {
        "momentum_reset": {"neg_frac": 1e-3, "lr_factor": 0.5, "lr_up": 1.1},
        "first_reset": True,
    },
}
HORIZON = {"l384": 4000, "y2": 5000}


def runs():
    out = dict(REPLAYS)
    for case in ("l384", "y2"):
        for name, extra in VARIANTS.items():
            out[f"{case}_{name}"] = (case, extra)
    return out


def start_variant(case, extra, run_dir):
    """Configuration of a variant; copies the pre-explosion state as its checkpoint
    (without the best state of the replay, which no state of the variant would beat, so
    that best.twc would never be written)."""
    replay = OUT / f"replay_{case}"
    pre = sorted(replay.glob("pre_explosion_*.pt"))
    if not pre:
        raise SystemExit(f"no pre_explosion_*.pt in {replay}: run replay_{case} first")
    it = int(pre[0].stem.split("_")[-1])
    start = it - REPLAYS[f"replay_{case}"]["stride"]
    extra, base = dict(extra), REPLAYS[f"replay_{case}"]
    if "momentum_reset" in extra:  # an option of the hs loss
        extra["loss_params"] = {**base["loss_params"],
                                "momentum_reset": extra.pop("momentum_reset")}  # fmt: skip
    cfg = {**base, "rollback": None, **extra,
           "max_iters": start + HORIZON[case], "start_state": pre[0].name}  # fmt: skip
    run_dir.mkdir(parents=True, exist_ok=True)
    if (
        not (run_dir / "checkpoint.pt").exists()
        and not (run_dir / "train.json").exists()
    ):
        state = torch.load(pre[0], map_location="cpu")
        state["best"] = None  # the variant keeps its own best state, not the replay's
        torch.save(state, run_dir / "checkpoint.pt")
    return cfg


def main():
    all_runs = runs()
    if "--summary" in sys.argv:
        res = {}
        for name in all_runs:
            f = OUT / name / "result.json"
            res[name] = json.loads(f.read_text()) if f.exists() else None
        print("\n".join(summary_table(res)))
        return
    for name in sys.argv[1:]:
        run_dir = OUT / name
        if name in REPLAYS:
            cfg = REPLAYS[name]
        else:
            cfg = start_variant(*all_runs[name], run_dir)
        train(cfg, run_dir)


if __name__ == "__main__":
    main()
