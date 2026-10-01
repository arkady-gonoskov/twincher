// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 Arkady Gonoskov

#include "shuttle_gpu.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <cub/cub.cuh>

void check_cuda(cudaError_t error){
    if (error != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
}

// frees device memory without throwing (for destructors); errors at this point (for example,
// during interpreter shutdown after the CUDA context is destroyed) cannot be handled anyway
void free_device(void *d_ptr) noexcept {
    if(d_ptr != nullptr) cudaFree(d_ptr);
}

// checks for errors in the launch of a kernel (e.g. invalid configuration); errors during
// the execution of kernels are reported by later synchronizing CUDA calls
void check_launch(){
    check_cuda(cudaGetLastError());
}

// allocates device memory and copies host data to it (stream-ordered)
template<class T>
T* make_d_copy(const T* h_ptr, size_t size, cudaStream_t stream)
{
    T* d_ptr = nullptr;
    check_cuda(cudaMalloc(&d_ptr, size * sizeof(T)));
    cudaError_t error = cudaMemcpyAsync(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice, stream);
    if(error != cudaSuccess){
        free_device(d_ptr);
        check_cuda(error);
    }
    return d_ptr;
}

__global__ void tw_set_i_kernel(twinch<float> *d_tw, int *d_i0, int *d_i1, int n_t){
    int it = blockIdx.x * blockDim.x + threadIdx.x;
    if(it < n_t){
        d_tw[it].i0 = d_i0[it];
        d_tw[it].i1 = d_i1[it];
    }
}

__global__ void tw_update_kernel(twinch<float> *d_tw, float *d_a, int n_t, twinch_config<float> tw_config){
    int it = blockIdx.x * blockDim.x + threadIdx.x;
    if(it < n_t) twinch_update<float>(&d_tw[it], &d_a[4*it], tw_config);
}

shuttle_gpu::shuttle_gpu(int n_s, int n_l, int n_c, int n_p, int n_m, int tw_type, int rng_seed):
n_s(n_s), n_l(n_l), n_c(n_c), n_p(n_p), n_m(n_m), tw_type(tw_type)
{
    check_batch_sizes(n_s, n_c, n_p, n_m);
    n_m_max = n_m;
    TW = std::make_unique<twincher<float>>(n_s, n_l, tw_type, rng_seed);
    n_t = TW->n_t;
}

shuttle_gpu::shuttle_gpu(std::string file_name, int n_c, int n_p, int n_m, int offset):
n_c(n_c), n_p(n_p), n_m(n_m)
{
    TW = std::make_unique<twincher<float>>();
    TW->load(file_name, offset);
    tw_type = TW->tw_type;
    n_s = TW->n_s;
    n_l = TW->n_l;
    n_t = TW->n_t;
    check_batch_sizes(n_s, n_c, n_p, n_m);
    n_m_max = n_m;
}

void shuttle_gpu::set_n_m(int n_m_new){
    if(n_m_new < 0 || n_m_new > n_m_max) throw std::invalid_argument(
        "n_m must be in [0, " + std::to_string(n_m_max) + "], got " + std::to_string(n_m_new));
    n_m = n_m_new;
}

void shuttle_gpu::init(
    std::uintptr_t s_ptr, 
    std::uintptr_t a_ptr,
    std::uintptr_t V_ptr,
    std::uintptr_t Q_ptr,
    std::uintptr_t B_ptr,
    std::uintptr_t b_ptr,
    std::uintptr_t g_ptr
){
    if(initialized) throw std::logic_error("shuttle_gpu::init: already initialized");
    // allocate all data:
    d_a = reinterpret_cast<float*>(a_ptr);
    d_s = reinterpret_cast<float*>(s_ptr);
    d_V = reinterpret_cast<float*>(V_ptr);
    d_Q = reinterpret_cast<float*>(Q_ptr);
    d_B = reinterpret_cast<float*>(B_ptr);
    d_dL_ds = reinterpret_cast<float*>(b_ptr);
    d_dL_da = reinterpret_cast<float*>(g_ptr);
    check_cuda(cudaMalloc(&d_A, n_s*n_c*sizeof(float)));
    check_cuda(cudaMalloc(&d_dLc_da, 2*n_s*n_c*sizeof(float)));
    check_cuda(cudaMalloc(&d_dLl_da, 2*n_s*sizeof(float)));

    // copy all from TW:
    d_l_start = make_d_copy(TW->l_start.data(), TW->l_start.size(), stream());
    d_i0 = make_d_copy(TW->i0.data(), TW->i0.size(), stream());
    d_i1 = make_d_copy(TW->i1.data(), TW->i1.size(), stream());
    check_cuda(cudaMemcpyAsync(d_a, TW->a.data(), 4*n_t*sizeof(float), cudaMemcpyHostToDevice, stream()));
    check_cuda(cudaMalloc(&d_tw, TW->n_t*sizeof(twinch<float>)));
    {
        int threads = n_threads;
        int blocks = (n_t + threads - 1) / threads;
        tw_set_i_kernel<<<blocks, threads, 0, stream()>>>(d_tw, d_i0, d_i1, n_t);
        check_launch();
    }
    
    // auxiliary:
    inv_n_c = 1/float(n_c);
    dLc_da_to_dLl_da = std::make_unique<matrix_row_reducer>(d_dLc_da, d_dLl_da, 2*n_s, n_c, stream());
    initialized = true;

    // initialize states of tw from parameter vector a:
    tw_update();
    check_cuda(cudaStreamSynchronize(stream()));
}

shuttle_gpu::~shuttle_gpu(){
    // also frees memory allocated by an init() that failed midway
    free_device(d_l_start);
    free_device(d_i0);
    free_device(d_i1);
    free_device(d_tw);
    free_device(d_A);
    free_device(d_dLc_da);
    free_device(d_dLl_da);
}

template<twincher_mode Mode, int tw_type>
__global__ void forward_kernel(twinch<float> *d_tw, float *d_s, float *d_Q, int *d_l_start, int il, int n_c, int n_s, int n_m){
    int it = d_l_start[il] + blockIdx.x;
    int ic = blockIdx.y * blockDim.x + threadIdx.x;
    if((it < d_l_start[il+1])&&(ic < n_c)){
        twincher_forward<Mode, float, tw_type>(d_tw[it], d_s, d_Q, ic, n_c, n_s, n_m);
    }
}

template<twincher_mode Mode>
void shuttle_gpu::forward(){
    check_initialized();
    for(int il = 0; il < n_l; il++){
        int threads = n_threads;
        dim3 grid((n_s+1)/2, (n_c + threads - 1)/threads);
        if(tw_type == 0)
            forward_kernel<Mode, 0><<<grid, threads, 0, stream()>>>(d_tw, d_s, d_Q, d_l_start, il, n_c, n_s, n_m);
        else if(tw_type == 1)
            forward_kernel<Mode, 1><<<grid, threads, 0, stream()>>>(d_tw, d_s, d_Q, d_l_start, il, n_c, n_s, n_m);
        else throw std::logic_error("shuttle_gpu::forward: unknown tw_type");
        check_launch();
    }
}

template<twincher_mode Mode, int tw_type>
__global__ void backward_kernel(twinch<float> *d_tw, float *d_s, float *d_Q, float *d_B, float *d_A, float *d_dL_ds, float *d_dLc_da, float sga, float sgt, float inv_n_c, int *d_l_start, int il, int n_c, int n_s, int n_m){
    int it = d_l_start[il] + blockIdx.x;
    int ic = blockIdx.y * blockDim.x + threadIdx.x;
    if((it < d_l_start[il+1])&&(ic < n_c)){
        twincher_backward<Mode, float, tw_type>(d_tw[it], d_s, d_Q, d_B, d_A, d_dL_ds, d_dLc_da+4*(it-d_l_start[il])*n_c, sga, sgt, inv_n_c, ic, n_c, n_s, n_m);
    }
}

template<twincher_mode Mode>
void shuttle_gpu::backward(){
    check_initialized();
    // in INFERENCE mode only s is updated (from s_out to s_in)
    if constexpr(Mode != INFERENCE){
        check_cuda(cudaMemsetAsync(d_A, 0, n_s*n_c*sizeof(float), stream()));
        check_cuda(cudaMemsetAsync(d_dL_da, 0, n_t*4*sizeof(float), stream()));
    }

    for(int il = n_l - 1; il >= 0; il--){
        if constexpr(Mode != INFERENCE) check_cuda(cudaMemsetAsync(d_dLc_da, 0, 2*n_s*n_c*sizeof(float), stream()));

        int threads = n_threads;
        dim3 grid((n_s+1)/2, (n_c + threads - 1)/threads);
        if(tw_type == 0)
            backward_kernel<Mode, 0><<<grid, threads, 0, stream()>>>(d_tw, d_s, d_Q, d_B, d_A, d_dL_ds, d_dLc_da, sga, sgt, inv_n_c, d_l_start, il, n_c, n_s, n_m);
        else if(tw_type == 1)
            backward_kernel<Mode, 1><<<grid, threads, 0, stream()>>>(d_tw, d_s, d_Q, d_B, d_A, d_dL_ds, d_dLc_da, sga, sgt, inv_n_c, d_l_start, il, n_c, n_s, n_m);
        else throw std::logic_error("shuttle_gpu::backward: unknown tw_type");
        check_launch();
        
        if constexpr(Mode != INFERENCE){
            dLc_da_to_dLl_da->run(stream());
            check_cuda(cudaMemcpyAsync(d_dL_da + 4*TW->l_start[il], d_dLl_da, 4*(TW->l_start[il+1] - TW->l_start[il])*sizeof(float), cudaMemcpyDeviceToDevice, stream()));
        }
    }
}

void shuttle_gpu::tw_update(){
    check_initialized();
    int threads = n_threads;
    int blocks = (n_t + threads - 1) / threads;
    tw_update_kernel<<<blocks, threads, 0, stream()>>>(d_tw, d_a, n_t, TW->tw_config);
    check_launch();
}

__global__ void set_V_out(float *V, int n_c, int n_s){
    int ip = blockIdx.x;
    int is = blockIdx.y;
    int ic = blockIdx.z * blockDim.x + threadIdx.x;
    if(ic < n_c) V[(ip*n_s + is)*n_c + ic] = (ip == is);
}

template<int tw_type>
__global__ void backward_V_kernel(twinch<float> *d_tw, float *d_s, float *d_V, 
    int *d_l_start, int il, int n_p, int n_s, int n_c)
{
    int it = d_l_start[il] + blockIdx.x;
    int ic = blockIdx.y * blockDim.x + threadIdx.x;
    if((it < d_l_start[il+1])&&(ic < n_c))
        twincher_backward_V<float, tw_type>(d_tw[it], d_s, d_V, ic, n_p, n_s, n_c);
}

void shuttle_gpu::compute_V(){
    check_initialized();
    forward<INFERENCE>();
    {
        int threads = n_threads;
        dim3 grid(n_p, n_s, (n_c + threads - 1)/threads);
        set_V_out<<<grid, threads, 0, stream()>>>(d_V, n_c, n_s);
        check_launch();
    }
    for(int il = n_l - 1; il >= 0; il--){
        int threads = n_threads;
        dim3 grid((n_s+1)/2, (n_c + threads - 1)/threads);
        if(tw_type == 0)
            backward_V_kernel<0><<<grid, threads, 0, stream()>>>(d_tw, d_s, d_V, d_l_start, il, n_p, n_s, n_c);
        else if(tw_type == 1)
            backward_V_kernel<1><<<grid, threads, 0, stream()>>>(d_tw, d_s, d_V, d_l_start, il, n_p, n_s, n_c);
        else throw std::logic_error("shuttle_gpu::compute_V: unknown tw_type");
        check_launch();
    }
}

matrix_row_reducer::matrix_row_reducer(float *d_mat, float *d_vec, int n_rows, int n_cols, cudaStream_t stream):
d_mat(d_mat), d_vec(d_vec), n_rows(n_rows), n_cols(n_cols)
{
    std::vector<int> h_begin(n_rows), h_end(n_rows);
    for(int i = 0; i < n_rows; i++){
        h_begin[i] = i*n_cols;
        h_end[i]   = (i + 1)*n_cols;            
    }
    try {
        check_cuda(cudaMalloc(&d_begin, n_rows * sizeof(int)));
        check_cuda(cudaMalloc(&d_end, n_rows * sizeof(int)));
        check_cuda(cudaMemcpyAsync(d_begin, h_begin.data(), n_rows*sizeof(int), cudaMemcpyHostToDevice, stream));
        check_cuda(cudaMemcpyAsync(d_end, h_end.data(), n_rows*sizeof(int), cudaMemcpyHostToDevice, stream));
        check_cuda(cudaStreamSynchronize(stream)); // h_begin and h_end are destroyed at return

        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        check_cuda(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_mat, d_vec, n_rows, d_begin, d_end));
        check_cuda(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    } catch(...) { // the destructor is not called if the constructor throws
        free_device(d_begin);
        free_device(d_end);
        free_device(d_temp_storage);
        throw;
    }
}

matrix_row_reducer::~matrix_row_reducer(){
    free_device(d_begin);
    free_device(d_end);
    free_device(d_temp_storage);
}

void matrix_row_reducer::run(cudaStream_t stream){
    check_cuda(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_mat, d_vec, n_rows, d_begin, d_end, stream));
}

// copies of the parameters a between device and TW->a, waiting for preceding operations on the stream
void shuttle_gpu::copy_a_to_host(){
    check_cuda(cudaMemcpyAsync(TW->a.data(), d_a, 4*n_t*sizeof(float), cudaMemcpyDeviceToHost, stream()));
    check_cuda(cudaStreamSynchronize(stream()));
}

void shuttle_gpu::copy_a_to_device(){
    check_cuda(cudaMemcpyAsync(d_a, TW->a.data(), 4*n_t*sizeof(float), cudaMemcpyHostToDevice, stream()));
    check_cuda(cudaStreamSynchronize(stream()));
}

void shuttle_gpu::set_rnd_t(int rng_seed){
    check_initialized();
    copy_a_to_host();
    std::mt19937 rng(rng_seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for(int it = 0; it < n_t; it++) TW->a[4*it + 3] = dist(rng);
    copy_a_to_device();
    tw_update();
}

void shuttle_gpu::tw_save(std::string file_name, bool append){
    check_initialized();
    copy_a_to_host();
    TW->save(file_name, append);
}

void shuttle_gpu::set_tw_config(double wl, double ws, double wu){
    check_initialized();
    copy_a_to_host();
    twinch_config<float> tw_config_new(wl, ws, wu);
    for(int it = 0; it < TW->n_t; it++){
        twinch_alter_config(&(TW->a[4*it]), TW->tw_config, tw_config_new);
    }
    copy_a_to_device();
    TW->tw_config = tw_config_new;
    tw_update(); // the derivatives with respect to a depend on the configuration
}

bool gpu_available(){
    int n_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&n_devices);
    if (err != cudaSuccess) return false;
    return (n_devices > 0);
}

template void shuttle_gpu::forward<INFERENCE>();
template void shuttle_gpu::forward<QUERY>();

template void shuttle_gpu::backward<INFERENCE>();
template void shuttle_gpu::backward<QUERY>();