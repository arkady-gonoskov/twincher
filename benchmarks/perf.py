# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

import numpy as np
import twincher
import time

def print_type(arr):
    module_name = arr.__class__.__module__.split(".")[0]
    print(module_name)

def set_rnd(arr, rng_seed=42):
    module_name = arr.__class__.__module__.split(".")[0]
    if module_name == "numpy":
        rng = np.random.default_rng(rng_seed)
        arr[:] = rng.uniform(-1, 1, size=arr.shape)
    elif module_name == "cupy":
        import cupy as cp
        rng = cp.random.default_rng(rng_seed)
        arr[:] = rng.random(arr.shape, dtype=arr.dtype)*2 - 1
    elif module_name == "torch":
        import torch  
        generator = torch.Generator(device=arr.device)
        generator.manual_seed(rng_seed)
        arr[:] = torch.rand(arr.shape, dtype=arr.dtype, device=arr.device, generator=generator)*2 - 1
    else:
        raise TypeError(f"Unsupported array type: {type(arr)}")

def to_np(arr):
    backend = type(arr).__module__.split(".")[0]
    if backend == "numpy":
        return arr
    elif backend == "cupy":
        import cupy as cp
        return cp.asnumpy(arr)
    elif backend == "torch":
        # return arr.detach().cpu().numpy()
        return arr.cpu().numpy()
    else:
        raise TypeError(f"Unsupported backend: {backend}")

def copy(arr):
    backend = type(arr).__module__.split(".")[0]
    if backend == "numpy":
        return arr.copy()
    elif backend == "cupy":
        return arr.copy()
    elif backend == "torch":
        return arr.clone()
    else:
        raise TypeError(f"Unsupported backend: {backend}")


def inference(dtype): #forward_backward
    n_iter = 5
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    SH = twincher.Shuttle(data_type=dtype, n_s=1024, n_l=1024, n_c=1024, n_p=4, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    if(dtype == "np"):
        SH.core.n_cores = 6
        print(f"Using {SH.core.n_cores} cores.")
    if(dtype == "cp"):
        SH.core.n_threads = 64
        print(f"Using {SH.core.n_threads} threads.")
    print(f"inference: n_s={n_s}, n_l={n_l}, n_c={n_c}, n_p={n_p}, n_m={n_m}, n_t={n_t}")
    SH.set_rnd_t(rng_seed=rng_seed)
    set_rnd(SH.s, rng_seed=rng_seed)
    start = time.perf_counter()
    perf_best = 1e+10
    perf_av = 0
    for i_iter in range(n_iter):
        start = time.perf_counter()
        SH.forward_inference()
        SH.backward_inference()
        end = time.perf_counter()
        perf = 1e+9*(end - start)/(n_t*n_c)
        perf_best = min(perf_best, perf)
        perf_av += perf/n_iter
        print("\rSH.s[0, 0]", SH.s[0, 0], end = "")
        print("\r                                    ", end = "")
        if(i_iter%1 == 0): print(f"\r {i_iter}/{n_iter}", end = "")

    print("\r         ", end = "\r")
    print(f"best: {perf_best:.6f} ns per operation")
    print(f"av  : {perf_av:.6f} ns per operation") 

def compute_V(dtype):
    n_iter = 5
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    SH = twincher.Shuttle(data_type=dtype, n_s=1024, n_l=1024, n_c=1024, n_p=4, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    print(f"compute_V: n_s={n_s}, n_l={n_l}, n_c={n_c}, n_p={n_p}, n_m={n_m}, n_t={n_t}")
    SH.set_rnd_t(rng_seed=rng_seed)
    set_rnd(SH.s, rng_seed=rng_seed)
    start = time.perf_counter()
    perf_best = 1e+10
    perf_av = 0
    for i_iter in range(n_iter):
        start = time.perf_counter()
        SH.compute_V()
        end = time.perf_counter()
        perf = 1e+9*(end - start)/(n_t*n_c)
        perf_best = min(perf_best, perf)
        perf_av += perf/n_iter
        print("\rSH.V[0, 0, 0]", SH.V[0, 0, 0], end = "")
        print("\r                                    ", end = "")
        if(i_iter%1 == 0): print(f"\r {i_iter}/{n_iter}", end = "")

    print("\r         ", end = "\r")
    print(f"best: {perf_best:.6f} ns per operation")
    print(f"av  : {perf_av:.6f} ns per operation") 

def query(dtype):
    n_iter = 5
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    SH = twincher.Shuttle(data_type=dtype, n_s=1024, n_l=1024, n_c=1024, n_p=4, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    print(f"query: n_s={n_s}, n_l={n_l}, n_c={n_c}, n_p={n_p}, n_m={n_m}, n_t={n_t}")
    if(dtype == "np"):
        SH.core.n_cores = 6
        print(f"Using {SH.core.n_cores} cores.")
    if(dtype == "cp"):
        SH.core.n_threads = 64
        print(f"Using {SH.core.n_threads} threads.")
    SH.set_rnd_t(rng_seed=rng_seed)
    set_rnd(SH.s)
    set_rnd(SH.Q)
    B_out = copy(SH.B)
    b_out = copy(SH.b)
    set_rnd(B_out)
    set_rnd(b_out)
    SH.B[:] = B_out
    SH.b[:] = b_out
    start = time.perf_counter()
    perf_best = 1e+10
    perf_av = 0
    for i_iter in range(n_iter):
        elapsed = 0
        start = time.perf_counter()
        SH.forward_query()
        SH.B[:] = B_out
        SH.b[:] = b_out
        SH.backward_query()
        elapsed += time.perf_counter() - start
        perf = 1e+9*elapsed/(n_t*n_c)
        perf_best = min(perf_best, perf)
        perf_av += perf/n_iter
        print("\rSH.g[0]", SH.g[0], end = "")
        print("\r                                        ", end = "")
        if(i_iter%1 == 0): print(f"\r {i_iter}/{n_iter}", end = "")

    print("\r         ", end = "\r")
    print(f"best: {perf_best:.6f} ns per operation")
    print(f"av  : {perf_av:.6f} ns per operation") 

def inference_mini(dtype): #forward_backward
    n_iter = 1
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    SH = twincher.Shuttle(data_type=dtype, n_s=1024, n_l=4, n_c=1024, n_p=4, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    print(f"inference: n_s={n_s}, n_l={n_l}, n_c={n_c}, n_p={n_p}, n_m={n_m}, n_t={n_t}")
    SH.set_rnd_t(rng_seed=rng_seed)
    set_rnd(SH.s, rng_seed=rng_seed)
    start = time.perf_counter()
    perf_best = 1e+10
    perf_av = 0
    for i_iter in range(n_iter):
        start = time.perf_counter()
        SH.forward_inference()
        SH.backward_inference()
        end = time.perf_counter()
        perf = 1e+9*(end - start)/(n_t*n_c)
        perf_best = min(perf_best, perf)
        perf_av += perf/n_iter
        print("\rSH.s[0, 0]", SH.s[0, 0], end = "")
        print("\r                                    ", end = "")
        if(i_iter%1 == 0): print(f"\r {i_iter}/{n_iter}", end = "")

    print("\r            ", end = "\r")
    print(f"best: {perf_best:.6f} ns per operation")
    print(f"av  : {perf_av:.6f} ns per operation") 

def query_mini(dtype):
    n_iter = 5
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    SH = twincher.Shuttle(data_type=dtype, n_s=1024, n_l=4, n_c=1024, n_p=4, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    print(f"query: n_s={n_s}, n_l={n_l}, n_c={n_c}, n_p={n_p}, n_m={n_m}, n_t={n_t}")
    SH.set_rnd_t(rng_seed=rng_seed)
    set_rnd(SH.s)
    set_rnd(SH.Q)
    B_out = copy(SH.B)
    b_out = copy(SH.b)
    set_rnd(B_out)
    set_rnd(b_out)
    SH.B[:] = B_out
    SH.b[:] = b_out
    start = time.perf_counter()
    perf_best = 1e+10
    perf_av = 0
    for i_iter in range(n_iter):
        elapsed = 0
        start = time.perf_counter()
        SH.forward_query()
        SH.B[:] = B_out
        SH.b[:] = b_out
        SH.backward_query()
        elapsed += time.perf_counter() - start
        perf = 1e+9*elapsed/(n_t*n_c)
        perf_best = min(perf_best, perf)
        perf_av += perf/n_iter
        print("\rSH.g[0]", SH.g[0], end = "")
        print("\r                                    ", end = "")
        if(i_iter%1 == 0): print(f"\r {i_iter}/{n_iter}", end = "")

    print("\r         ", end = "\r")
    print(f"best: {perf_best:.6f} ns per operation")
    print(f"av  : {perf_av:.6f} ns per operation") 

# inference(dtype="np")
# inference(dtype="cp")

# compute_V(dtype="np")
# compute_V(dtype="cp")

# query(dtype="np")
#query(dtype="cp")

import torch
torch.cuda.set_per_process_memory_fraction(0.92)   # per device, call once at startup
query(dtype="torch")

# inference_mini(dtype="cp")
# query_mini(dtype="cp")