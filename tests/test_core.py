# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

"""
Unit tests of the computational core (Shuttle).

The CPU backend is verified against finite-difference estimates, and the GPU backend is
verified against the CPU backend. The printed intermediate values are shown by pytest for
failed tests (and for all tests with the option -rP).
"""

import numpy as np
import pytest
import twincher
from helpers import set_rnd, to_np, copy, check

# ======================================== CPU ========================================

def test_cpu_save_load(tmp_path):
    rng_seed = 100
    SH = twincher.Shuttle(data_type="np", n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    SH.set_rnd_t(rng_seed=rng_seed)
    SH.core.set_tw_config(0.02, 0.2, 1.5)
    print(SH.a[2])
    file_name = str(tmp_path / "cpu_save_load.tw")
    SH.tw_save(file_name)
    rng = np.random.default_rng(42)
    SH.s[:] = rng.uniform(-1, 1, size=SH.s.shape)
    s_in_0 = SH.s.copy()
    SH.forward_inference()
    s_out_0 = SH.s.copy()

    SH1 = twincher.Shuttle(data_type="np")
    SH1.tw_load(file_name, n_c=n_c, n_p=n_p, n_m=n_m)
    print(SH1.a[2])
    SH1.s[:] = s_in_0
    SH1.forward_inference()
    s_out_1 = SH1.s.copy()

    check(np.max(np.abs(s_out_0 - s_out_1)), 1e-8)

def test_cpu_inference():
    rng_seed = 42
    SH = twincher.Shuttle(data_type="np", n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    SH.set_rnd_t()
    set_rnd(SH.s)
    s0 = SH.s.copy()
    print("s_in_0=", SH.s)
    SH.forward_inference()
    print("s_out =", SH.s)
    SH.backward_inference()
    print("s_in  =", SH.s)
    check(np.max(np.abs(s0 - SH.s)), 1e-8)

def test_cpu_compute_V():
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    SH = twincher.Shuttle(data_type="np", n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    SH.set_rnd_t()
    SH.s[:] = rng.uniform(-1, 1, size=SH.s.shape)
    s_in_0 = SH.s.copy()
    SH.forward_inference()
    s_out_0 = SH.s.copy()
    V_est = np.empty((n_p, n_s, n_c), dtype=np.float64)
    d = 1e-8
    for i_s in range(n_s):
        SH.s[:] = s_in_0.copy()
        for i_c in range(n_c):
            SH.s[i_s, i_c] += d
        SH.forward_inference()
        for i_p in range(n_p):
            V_est[i_p, i_s, :] = (SH.s[i_p, :] - s_out_0[i_p, :])/d
    print("V_est=", V_est)
    SH.s[:] = s_in_0.copy()
    SH.compute_V()
    V_cpu = SH.V.copy()
    print("V_cpu=", V_cpu)
    check(np.max(np.abs(V_est - V_cpu)), 1e-6)

def test_cpu_Q():
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    SH = twincher.Shuttle(data_type="np", n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    SH.set_rnd_t()
    SH.s[:] = rng.uniform(-1, 1, size=SH.s.shape)
    s_in_0 = SH.s.copy()
    SH.forward_inference()
    s_out_0 = SH.s.copy()
    Q_in = SH.Q.copy()
    Q_in[:] = rng.uniform(-1, 1, size=Q_in.shape)
    Q_out_est = SH.Q.copy()
    d = 1e-8
    for i_m in range(n_m):
        SH.s[:] = s_in_0
        for i_s in range(n_s):
            SH.s[i_s, :] += Q_in[i_s, i_m, :]*d
        SH.forward_inference()
        for i_s in range(n_s):
            Q_out_est[i_s, i_m, :] = (SH.s[i_s, :] - s_out_0[i_s, :])/d
    print("Q_out_est=", Q_out_est)
    SH.s[:] = s_in_0
    SH.Q[:]= Q_in
    SH.forward_query()
    Q_out_cpu = SH.Q.copy()
    print("Q_out_cpu=", Q_out_cpu)
    check(np.max(np.abs(Q_out_est - Q_out_cpu)), 1e-6)

def test_cpu_backward_B():
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    SH = twincher.Shuttle(data_type="np", n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    SH.set_rnd_t()
    SH.s[:] = rng.uniform(-1, 1, size=SH.s.shape)
    s_in = SH.s.copy()
    Q_in = SH.Q.copy()
    Q_in[:] = rng.uniform(-1, 1, size=Q_in.shape)
    B_out = SH.B.copy()
    B_out[:] = rng.uniform(-1, 1, size=B_out.shape)

    d = 1e-8
    B_in_est = SH.B.copy()

    SH.s[:] = s_in
    SH.Q[:] = Q_in
    SH.forward_query()
    Q_out_0 = SH.Q.copy()

    for i_s in range(n_s):
        for i_m in range(n_m):
            SH.s[:] = s_in
            SH.Q[:] = Q_in
            SH.Q[i_s, i_m, :] += d
            SH.forward_query()
            B_in_est[i_s, i_m, :] = 0
            for i_s_ in range(n_s):
                for i_m_ in range(n_m):
                    B_in_est[i_s, i_m, :] += B_out[i_s_, i_m_, :]*(SH.Q[i_s_, i_m_, :] - Q_out_0[i_s_, i_m_, :])/d

    print("B_in_est=", B_in_est)
    SH.s[:] = s_in
    SH.Q[:] = Q_in
    SH.forward_query()
    SH.B[:] = B_out
    SH.backward_query()
    B_in_cpu = SH.B.copy()
    print("B_in_cpu=", B_in_cpu)
    check(np.max(np.abs(B_in_est - B_in_cpu)), 1e-6)

def test_cpu_b():
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    SH = twincher.Shuttle(data_type="np", n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    SH.core.sga = 0 # set s-gate loss amplitude to zero, to not affect b_out -> b_in
    SH.set_rnd_t()
    SH.s[:] = rng.uniform(-1, 1, size=SH.s.shape)
    s_in_0 = SH.s.copy()
    Q_in = SH.Q.copy()
    Q_in[:] = rng.uniform(-1, 1, size=Q_in.shape)
    B_out = SH.B.copy()
    B_out[:] = rng.uniform(-1, 1, size=B_out.shape)
    b_out = SH.b.copy()
    b_out[:] = rng.uniform(-1, 1, size=b_out.shape)

    b_in_est = b_out.copy()

    d = 1e-8
    SH.s[:] = s_in_0
    SH.forward_inference()
    s_out_0 = SH.s.copy()

    for i_s in range(n_s):
        SH.s[:] = s_in_0
        SH.s[i_s, :] += d
        SH.forward_inference()
        b_in_est[i_s, :] = 0
        for i_s_ in range(n_s):
            b_in_est[i_s, :] += b_out[i_s_]*(SH.s[i_s_, :] - s_out_0[i_s_, :])/d
    print("b_in_est=", b_in_est)

    SH.s[:] = s_in_0
    SH.Q[:] = Q_in
    SH.forward_query()
    SH.B[:] = B_out
    SH.b[:] = b_out
    SH.backward_query()
    b_in_cpu = SH.b.copy()
    print("b_in_cpu=", b_in_cpu)

    check(np.max(np.abs(b_in_est - b_in_cpu)), 1e-6)

def test_cpu_g():
    rng_seed = 42
    rng = np.random.default_rng(rng_seed)
    SH = twincher.Shuttle(data_type="np", n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    SH.core.sga = 0
    SH.set_rnd_t()

    s_in = SH.s.copy()
    s_in[:] = rng.uniform(-1, 1, size=s_in.shape)
    Q_in = SH.Q.copy()
    Q_in[:] = rng.uniform(-1, 1, size=Q_in.shape)
    B_out = SH.B.copy()
    B_out[:] = rng.uniform(-1, 1, size=B_out.shape)
    b_out = SH.b.copy()
    b_out[:] = rng.uniform(-1, 1, size=b_out.shape)

    g_est = SH.g.copy()
    d = 1e-8

    SH.s[:] = s_in
    SH.Q[:] = Q_in
    SH.forward_query()
    s_out_0 = SH.s.copy()
    Q_out_0 = SH.Q.copy()

    inv_n_c = 1/n_c

    for i_a in range(4*n_t):
        SH.a[i_a] += d
        SH.tw_update()
        SH.s[:] = s_in
        SH.Q[:] = Q_in
        SH.forward_query()
        g_est[i_a] = 0
        for i_s in range(n_s):
            for i_c in range(n_c):
                g_est[i_a] += b_out[i_s, i_c]*(SH.s[i_s, i_c] - s_out_0[i_s, i_c])/d
                for i_m in range(n_m):
                    g_est[i_a] += B_out[i_s, i_m, i_c]*(SH.Q[i_s, i_m, i_c] - Q_out_0[i_s, i_m, i_c])/d
        g_est[i_a] *= inv_n_c
        SH.a[i_a] -= d
    print("g_est=", g_est)

    SH.tw_update()
    SH.s[:] = s_in
    SH.Q[:] = Q_in
    SH.forward_query()
    SH.B[:] = B_out
    SH.b[:] = b_out
    SH.backward_query()
    g_cpu = SH.g.copy()
    print("g_cpu=", g_cpu)

    check(np.max(np.abs(g_est - g_cpu)), 1e-6)

# ======================================== GPU ========================================
# Each test runs for both GPU data types (see the fixture gpu_data_type in conftest.py).

@pytest.mark.gpu
def test_gpu_save_load(gpu_data_type, tmp_path):
    rng_seed = 42
    SH = twincher.Shuttle(data_type=gpu_data_type, n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    SH.set_rnd_t(rng_seed=rng_seed)
    SH.core.set_tw_config(0.02, 0.2, 1.5)
    print(SH.a[2])
    file_name = str(tmp_path / "gpu_save_load.tw")
    SH.tw_save(file_name)
    set_rnd(SH.s)
    s_in_0 = copy(SH.s)
    SH.forward_inference()
    s_out_0 = copy(SH.s)

    SH1 = twincher.Shuttle(data_type=gpu_data_type)
    SH1.tw_load(file_name, n_c=n_c, n_p=n_p, n_m=n_m)
    print(SH1.a[2])
    SH1.s[:] = s_in_0
    SH1.forward_inference()
    s_out_1 = copy(SH1.s)
    check(np.max(np.abs(to_np(s_out_0) - to_np(s_out_1))), 1e-8)

@pytest.mark.gpu
def test_gpu_inference(gpu_data_type):
    rng_seed = 42
    SH = twincher.Shuttle(data_type=gpu_data_type, n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    SH.set_rnd_t()
    set_rnd(SH.s)
    s0 = copy(SH.s)
    print(s0)
    SH.forward_inference()
    print(SH.s)
    SH.backward_inference()
    print(SH.s)
    check(np.max(np.abs(to_np(s0) - to_np(SH.s))), 1e-6)

@pytest.mark.gpu
def test_gpu_compute_V(gpu_data_type):
    rng_seed = 42
    SH = twincher.Shuttle(data_type=gpu_data_type, n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    SH.set_rnd_t()
    set_rnd(SH.s)
    S = twincher.Shuttle(data_type="np", n_s=n_s, n_l=n_l, n_c=n_c, n_p=n_p, n_m=n_m, rng_seed=rng_seed)
    S.set_rnd_t()
    S.s[:] = to_np(SH.s)
    SH.compute_V()
    S.compute_V()
    print("CPU: V=", S.V)
    print("GPU: V=", SH.V)
    check(np.max(np.abs(to_np(S.V) - to_np(SH.V))), 1e-6)

@pytest.mark.gpu
def test_gpu_Q(gpu_data_type):
    rng_seed = 42
    SH = twincher.Shuttle(data_type=gpu_data_type, n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    SH.set_rnd_t()
    S = twincher.Shuttle(data_type="np", n_s=n_s, n_l=n_l, n_c=n_c, n_p=n_p, n_m=n_m, rng_seed=rng_seed)
    S.set_rnd_t()

    set_rnd(SH.s)
    set_rnd(SH.Q)

    S.s[:] = to_np(SH.s)
    S.Q[:] = to_np(SH.Q)

    S.forward_query()
    SH.forward_query()

    print("CPU: Q=", S.Q)
    print("GPU: Q=", SH.Q)
    check(np.max(np.abs(to_np(S.Q) - to_np(SH.Q))), 1e-6)

@pytest.mark.gpu
def test_gpu_B_b_g(gpu_data_type):
    rng_seed = 42

    print("--- gpu computations ---")
    SH = twincher.Shuttle(data_type=gpu_data_type, n_s=4, n_l=3, n_c=2, n_p=3, n_m=4, rng_seed=rng_seed)
    n_s, n_l, n_c, n_p, n_m, n_t = SH.n_s, SH.n_l, SH.n_c, SH.n_p, SH.n_m, SH.n_t
    SH.set_rnd_t()

    s_in = copy(SH.s)
    Q_in = copy(SH.Q)
    B_out = copy(SH.B)
    b_out = copy(SH.b)
    for data in (s_in, Q_in, B_out, b_out): set_rnd(data)

    SH.s[:] = s_in
    SH.Q[:] = Q_in
    SH.forward_query()
    SH.B[:] = B_out
    SH.b[:] = b_out
    SH.backward_query()
    B_in_gpu = copy(SH.B)
    b_in_gpu = copy(SH.b)
    g_gpu = copy(SH.g)
    print("g_gpu=", g_gpu)

    print("--- cpu computations ---")
    S = twincher.Shuttle(data_type="np", n_s=n_s, n_l=n_l, n_c=n_c, n_p=n_p, n_m=n_m, rng_seed=rng_seed)
    S.set_rnd_t()

    S.s[:] = to_np(s_in)
    S.Q[:] = to_np(Q_in)
    S.forward_query()
    S.B[:] = to_np(B_out)
    S.b[:] = to_np(b_out)
    S.backward_query()
    B_in_cpu = S.B.copy()
    b_in_cpu = S.b.copy()
    g_cpu = S.g.copy()
    print("g_cpu=", g_cpu)

    err_B_in = np.max(np.abs(to_np(B_in_cpu) - to_np(B_in_gpu)))
    err_b_in = np.max(np.abs(to_np(b_in_cpu) - to_np(b_in_gpu)))
    err_g =    np.max(np.abs(to_np(g_cpu) -    to_np(g_gpu)))

    print("err_B_in=", err_B_in)
    print("err_b_in=", err_b_in)
    print("err_g=", err_g)

    check(max(err_B_in, err_b_in, err_g), 1e-5)
