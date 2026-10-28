"""Insert zero-twist layers into a twincher (.twc), keeping the map.

A layer is inserted after every `every`-th existing layer (e.g. every 2: n_l 256 ->
384). The .twc file stores the wiring (pairs of components per twinch), so the existing
layers keep theirs; the new layers get the round-robin pairs of their new index (as
twincher.h would assign them) and parameters as for a cold start (c0, c1, w random) with
a3 = 0, i.e. twist t = 0: they are identities until trained. The metadata is copied.

Usage:
    python twc_extend.py IN.twc OUT.twc [--every 2] [--seed 1]
"""

import argparse
import json
import struct

import numpy as np
import torch
import twincher
from twincher.shuttle import _read_header

MAGIC = b"TWINCHER"


def read_twc(file):
    """(metadata, dict of the core fields) of a .twc file."""
    metadata, offset = _read_header(file)
    with open(file, "rb") as f:
        raw = f.read()[offset:]
    tw_type, n_s, n_l, n_t, fp_size = struct.unpack_from("<5i", raw, 0)
    fp = {4: np.float32, 8: np.float64}[fp_size]
    pos = 20
    twv = np.frombuffer(raw, fp, 3, pos)
    pos += 3 * fp_size
    a = np.frombuffer(raw, fp, 4 * n_t, pos)
    pos += 4 * n_t * fp_size
    i0 = np.frombuffer(raw, np.int32, n_t, pos)
    pos += 4 * n_t
    i1 = np.frombuffer(raw, np.int32, n_t, pos)
    pos += 4 * n_t
    l_start = np.frombuffer(raw, np.int32, n_l + 1, pos)
    core = {"tw_type": tw_type, "n_s": n_s, "n_l": n_l, "fp": fp, "twv": twv, "a": a,
            "i0": i0, "i1": i1, "l_start": l_start}  # fmt: skip
    return metadata, core


def write_twc(file, metadata, core):
    meta = json.dumps(metadata, indent=4).encode("utf-8")
    n_t = len(core["i0"])
    with open(file, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", len(meta)))
        f.write(meta)
        fp_size = np.dtype(core["fp"]).itemsize
        f.write(
            struct.pack("<5i", core["tw_type"], core["n_s"], core["n_l"], n_t, fp_size)
        )
        fields = (("twv", core["fp"]), ("a", core["fp"]), ("i0", np.int32),
                  ("i1", np.int32), ("l_start", np.int32))  # fmt: skip
        f.writelines(
            np.ascontiguousarray(core[k], dtype=t).tobytes() for k, t in fields
        )


def round_robin(n_s, il):
    """Pairs (i0, i1) of layer il as in twincher.h."""
    stage, h = il % (n_s - 1), n_s // 2
    i0 = [stage] + [(h - it + stage) % (n_s - 1) for it in range(1, h)]
    i1 = [n_s - 1] + [(h + it - 1 + stage) % (n_s - 1) for it in range(1, h)]
    return i0, i1


def extend(core, every, seed):
    n_s, h = core["n_s"], core["n_s"] // 2
    rng = np.random.default_rng(seed)
    a_old = core["a"].reshape(-1, 4)
    layers = []  # (a (h, 4), i0, i1)
    for il in range(core["n_l"]):
        lo, hi = core["l_start"][il], core["l_start"][il + 1]
        layers.append((a_old[lo:hi], core["i0"][lo:hi], core["i1"][lo:hi]))
        if (il + 1) % every == 0:
            i0, i1 = round_robin(n_s, len(layers))
            a = np.zeros((h, 4))
            a[:, :2] = (2 * rng.random((h, 2)) - 1) / np.sqrt(3)
            a[:, 2] = -0.333 + (0.5 + 0.333) * rng.random(h)
            layers.append((a, np.array(i0), np.array(i1)))
    sizes = [len(layer[1]) for layer in layers]
    return {**core, "n_l": len(layers),
            "a": np.concatenate([layer[0] for layer in layers]).ravel(),
            "i0": np.concatenate([layer[1] for layer in layers]),
            "i1": np.concatenate([layer[2] for layer in layers]),
            "l_start": np.concatenate([[0], np.cumsum(sizes)])}  # fmt: skip


def max_difference(file_a, file_b, n_y, n_p, n=1024, seed=0):
    """max |s_a - s_b| and |Q_a - Q_b| / max |Q| of the two twinchers for random inputs
    (CPU, double precision: the identity layers are exact up to rounding)."""
    out = []
    for file in (file_a, file_b):
        sh = twincher.Shuttle(data_type="torch_cpu")
        sh.load(str(file), n_c=n, n_p=n_p, n_m=n_p)
        g = torch.Generator().manual_seed(seed)
        sh.s.fill_(0)
        sh.s[:n_y].copy_(0.8 * (2 * torch.rand((n_y, n), generator=g) - 1))
        sh.Q.fill_(0)
        sh.Q[:n_y].copy_(torch.randn((n_y, n_p, n), generator=g, dtype=torch.float64))
        sh.forward_query()
        out.append((sh.s.clone(), sh.Q.clone()))
    return ((out[0][0] - out[1][0]).abs().max().item(),
            ((out[0][1] - out[1][1]).abs().max() / out[0][1].abs().max()).item())  # fmt: skip


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    metadata, core = read_twc(args.src)
    new = extend(core, args.every, args.seed)
    write_twc(args.dst, metadata, new)
    d_s, d_q = max_difference(args.src, args.dst, metadata["n_y"], metadata["n_p"])
    print(f"{args.src}: n_l {core['n_l']} -> {new['n_l']}; max |ds| {d_s:.2e}, "
          f"max |dQ| / max |Q| {d_q:.2e}")  # fmt: skip


if __name__ == "__main__":
    main()
