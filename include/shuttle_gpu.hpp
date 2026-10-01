// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 Arkady Gonoskov

#pragma once
#include "twinch.h"
#include "twincher.h"
#include "shuttle_util.h"

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>
#include <stdexcept>

struct matrix_row_reducer{
    float *d_mat; // pointer to source matrix in row-first format
    float *d_vec; // pointer to output vector
    int n_rows, n_cols;
    // internal data:
    int *d_begin = nullptr, *d_end = nullptr;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes;
    matrix_row_reducer(float *d_mat, float *d_vec, int n_rows, int n_cols, cudaStream_t stream);
    ~matrix_row_reducer();
    void run(cudaStream_t stream);
    matrix_row_reducer(const matrix_row_reducer&) = delete;
    matrix_row_reducer& operator=(const matrix_row_reducer&) = delete;
};

struct shuttle_gpu{
    std::unique_ptr<twincher<float>> TW; // twincher model
    int tw_type; // tw_type, copied from TW
    int n_s; // size of s, copied from TW
    int n_l; // number of layers, copied from TW
    int n_t; // number of twincher, copied from TW

    int n_c; // number of cases in the batch 
    int n_p; // maximal number of p vector components
    int n_m; // number of objective modes
    int n_m_max; // n_m at construction, which sets the size of Q and B

    // device memory allocated by shuttle_gpu (freed in the destructor):
    int *d_l_start = nullptr, *d_i0 = nullptr, *d_i1 = nullptr; // twincher structure, copied from TW
    twinch<float> *d_tw = nullptr; // array of twinches
    float *d_A = nullptr; // n_s*n_c
    float *d_dLc_da = nullptr; //4*n_s_2*n_c
    float *d_dLl_da = nullptr; // local buffer 4*n_s_2
    std::unique_ptr<matrix_row_reducer> dLc_da_to_dLl_da;

    // device memory bound by init() (owned by the caller):
    float *d_s = nullptr; // n_s*n_c running field of s vectors [is*n_c + ic]
    float *d_a = nullptr;  // 4*n_t, model parameters, aka 'param'
    float *d_V = nullptr; // n_p*n_s*n_c, dr_ds matrix
    float *d_Q = nullptr; // n_s*n_m*n_c // Q = ds_dm
    float *d_B = nullptr; // n_s*n_m*n_c
    float *d_dL_ds = nullptr; //n_s*n_c
    float *d_dL_da = nullptr; //n_t*4

    bool initialized = false; // set by init()
    inline void check_initialized() const {
        if(!initialized) throw std::logic_error("shuttle_gpu: init() must be called before this operation");
    }

    int n_threads = 64; // number of threads for running CUDA kernels
    // CUDA stream for all device operations (0: legacy default stream); set to the stream
    // used by the framework that operates the data arrays (e.g. PyTorch's current stream)
    std::uintptr_t cuda_stream_ptr = 0;
    inline cudaStream_t stream() const { return reinterpret_cast<cudaStream_t>(cuda_stream_ptr); }

    float inv_n_c;

    // gate regularization for s (ought to prevent s from going far from [-1, 1] interval):
    float sga = 1.0; // amplitude
    float sgt = 1.3; // threshold 

    shuttle_gpu(int n_s, int n_l, int n_c, int n_p, int n_m, int tw_type = 1, int rng_seed = 42);

    shuttle_gpu(std::string file_name, int n_c, int n_p, int n_m, int offset = 0);

    void init(
        std::uintptr_t s_ptr, 
        std::uintptr_t a_ptr,
        std::uintptr_t V_ptr,
        std::uintptr_t Q_ptr,
        std::uintptr_t B_ptr,
        std::uintptr_t b_ptr,
        std::uintptr_t g_ptr
    );

    ~shuttle_gpu();
    // not copyable, since it owns device memory
    shuttle_gpu(const shuttle_gpu&) = delete;
    shuttle_gpu& operator=(const shuttle_gpu&) = delete;
    void tw_update();
    
    template<twincher_mode Mode>
    void forward();

    template<twincher_mode Mode>
    void backward();

    void compute_V();

    void set_rnd_t(int rng_seed = 42);
    
    void tw_save(std::string file_name, bool append = false);
    
    void set_tw_config(double wl, double ws, double wu);

    // changes the number of objective modes used in computations (Q and B keep their size)
    void set_n_m(int n_m_new);

    void copy_a_to_host();
    void copy_a_to_device();
};

bool gpu_available();

