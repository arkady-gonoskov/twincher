// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 Arkady Gonoskov

#pragma once
#include "twincher.h"
#include "shuttle_util.h"
#include <memory>
#ifdef _OPENMP
    #include <omp.h>
#endif

template<typename fp>
struct shuttle_cpu{
    std::unique_ptr<twincher<fp>> TW; // twincher model
    int n_s; // copy from TW
    int n_l; // copy from TW
    int n_t; // copy from TW
    std::vector<int> l_start; // copy from TW 

    int n_c; // number of cases in the batch 
    int n_p; // maximal number of p vector components (used in V)
    int n_m; // number of objective modes (primary loss vectors)
    int n_m_max; // n_m at construction, which sets the size of Q and B
    int tw_type; // type of twinch

    bool useOmp = true; // flag for disabling openMP for debuging
    // number of CPU threads; 0: OpenMP default (OMP_NUM_THREADS if set, otherwise all cores)
    int n_cores = 0;
    inline int n_threads() const {
        #ifdef _OPENMP
            return n_cores > 0 ? n_cores : omp_get_max_threads();
        #else
            return 1;
        #endif
    }

    std::vector<twinch<fp>> tw; // twinches (parameters and wiring) used in computations
    // u:=ds_dm
    fp *s = nullptr; // n_s*n_c running field of s vectors [is*n_c + ic]
    fp *a = nullptr; // 4*n_t, model parameters, aka 'param'
    fp *V = nullptr; // n_p*n_s*n_c // dr_ds
    fp *Q = nullptr; // ds_dm, n_s*n_m*n_c 
    fp *B = nullptr; // dL_du, n_s*n_m*n_c 
    std::vector<fp> A; // n_s*n_c
    fp *dL_ds = nullptr; // n_s*n_c    
    fp *dL_da = nullptr; // 4*n_t, gradient of loss with respect to model parameters, aka 'grad'
    std::vector<fp> dLc_da; // 4*n_s_2*n_c

    bool initialized = false; // set by init()
    inline void check_initialized() const {
        if(!initialized) throw std::logic_error("shuttle_cpu: init() must be called before this operation");
    }

    fp inv_n_c;

    // gate regularization for s:
    fp sga = 1.0; // amplitude
    fp sgt = 1.3; // threshold
    bool compute_gate_loss = false;
    fp gate_loss;

    shuttle_cpu(int n_s, int n_l, int n_c, int n_p, int n_m, int tw_type = 1, int rng_seed = 42):
    n_s(n_s), n_l(n_l), n_c(n_c), n_p(n_p), n_m(n_m), tw_type(tw_type)
    {
        check_batch_sizes(n_s, n_c, n_p, n_m);
        n_m_max = n_m;
        TW = std::make_unique<twincher<fp>>(n_s, n_l, tw_type, rng_seed);
        n_t = TW->n_t;
        l_start = TW->l_start;
    }

    // constructor that uses twincher from file
    shuttle_cpu(std::string file_name, int n_c, int n_p, int n_m, int offset = 0):
    n_c(n_c), n_p(n_p), n_m(n_m)
    {
        TW = std::make_unique<twincher<fp>>();
        TW->load(file_name, offset);
        tw_type = TW->tw_type;
        n_s = TW->n_s;
        n_l = TW->n_l;
        n_t = TW->n_t;
        l_start = TW->l_start;
        check_batch_sizes(n_s, n_c, n_p, n_m);
        n_m_max = n_m;
    }

    // changes the number of objective modes used in computations (Q and B keep their size)
    inline void set_n_m(int n_m_new){
        if(n_m_new < 0 || n_m_new > n_m_max) throw std::invalid_argument(
            "n_m must be in [0, " + std::to_string(n_m_max) + "], got " + std::to_string(n_m_new));
        n_m = n_m_new;
    }

    //initialization that binds data (expects double* or necessary sizes)
    inline void init(
        std::uintptr_t s_ptr, 
        std::uintptr_t a_ptr,
        std::uintptr_t V_ptr,
        std::uintptr_t Q_ptr,
        std::uintptr_t B_ptr,
        std::uintptr_t b_ptr,
        std::uintptr_t g_ptr
    ){
        if(initialized) throw std::logic_error("shuttle_cpu::init: already initialized");
        // bind data:
        a = reinterpret_cast<fp*>(a_ptr);
        s = reinterpret_cast<fp*>(s_ptr);
        V = reinterpret_cast<fp*>(V_ptr);
        Q = reinterpret_cast<fp*>(Q_ptr);
        B = reinterpret_cast<fp*>(B_ptr);
        dL_ds = reinterpret_cast<fp*>(b_ptr);
        dL_da = reinterpret_cast<fp*>(g_ptr);
        
        // allocate internal data:
        A.resize(n_s*n_c);
        dLc_da.resize(2*n_s*n_c);

        // copy all from TW:
        memcpy(a, TW->a.data(), TW->a.size()*sizeof(fp));
        tw.resize(n_t);
        for(int it = 0; it < n_t; it++) {
            tw[it].i0 = TW->i0[it];
            tw[it].i1 = TW->i1[it];
        }
        
        // auxiliary data:
        inv_n_c = 1/fp(n_c);
        initialized = true;

        // initialize states of tw from parameter vector a:
        tw_update();
    }

    // saves to file model itself (sizes, wiring, types, parameters, etc.), 
    // but not the training configuration (n_c, sga, sgt, etc.)
    inline void tw_save(std::string file_name, bool append = false){
        check_initialized();
        memcpy(TW->a.data(), a, TW->a.size()*sizeof(fp));
        TW->save(file_name, append);
    }

    // updates the state of twinches in tw based on the state of vector a
    inline void tw_update(){
        check_initialized();
        for(int it = 0; it < n_t; it++) 
            twinch_update<fp>(&tw[it], &(a[4*it]), TW->tw_config);
    }

    // sets random a values ~U[-1, 1] that corresponds to twinch.t values and updates them
    inline void set_rnd_t(int rng_seed = 42){
        check_initialized();
        std::mt19937 rng(rng_seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for(int it = 0; it < n_t; it++) a[4*it + 3] = dist(rng);
        tw_update();
    }

    // not copyable, since data arrays are bound by raw pointers
    shuttle_cpu(const shuttle_cpu&) = delete;
    shuttle_cpu& operator=(const shuttle_cpu&) = delete;

    template<twincher_mode Mode>
    inline void forward(){
        check_initialized();
        [[maybe_unused]] const int nt = n_threads();
        for(int il = 0; il < n_l; il++){
            if(tw_type == 0){
                #pragma omp parallel for collapse(2) if(useOmp) num_threads(nt) schedule(static, 256)
                for(int it = l_start[il]; it < l_start[il+1]; it++)
                for(int ic = 0; ic < n_c; ic++)
                    twincher_forward<Mode, fp, 0>(tw[it], s, Q, ic, n_c, n_s, n_m);
            } else if(tw_type == 1){
                #pragma omp parallel for collapse(2) if(useOmp) num_threads(nt) schedule(static, 256)
                for(int it = l_start[il]; it < l_start[il+1]; it++)
                for(int ic = 0; ic < n_c; ic++)
                    twincher_forward<Mode, fp, 1>(tw[it], s, Q, ic, n_c, n_s, n_m);
            } else throw std::logic_error("shuttle_cpu::forward: unknown tw_type");
        }
    }

    template<twincher_mode Mode>
    inline void backward(){
        check_initialized();
        // in INFERENCE mode only s is updated (from s_out to s_in)
        if constexpr (Mode != INFERENCE){
            std::fill(A.begin(), A.end(), fp(0)); // by default
            memset(dL_da, 0, n_t*4*sizeof(fp));
            if(compute_gate_loss) gate_loss = 0;
        }

        [[maybe_unused]] const int nt = n_threads();
        for(int il = n_l - 1; il >= 0; il--){
            
            if(Mode != INFERENCE && compute_gate_loss){
                for(int it = l_start[il]; it < l_start[il+1]; it++)
                for(int ic = 0; ic < n_c; ic++){
                    fp s0 = s[tw[it].i0*n_c + ic]; 
                    if(std::abs(s0) > sgt) gate_loss += inv_n_c*sga*sqr(std::abs(s0) - sgt);
                    fp s1 = s[tw[it].i1*n_c + ic]; 
                    if(std::abs(s1) > sgt) gate_loss += inv_n_c*sga*sqr(std::abs(s1) - sgt);
                }
            }
            if(tw_type == 0){
                #pragma omp parallel for collapse(2) if(useOmp) num_threads(nt) schedule(static, 256)
                for(int it = l_start[il]; it < l_start[il+1]; it++)
                for(int ic = 0; ic < n_c; ic++)
                    twincher_backward<Mode, fp, 0>(tw[it], s, Q, B, A.data(), dL_ds, dLc_da.data()+4*(it-l_start[il])*n_c, sga, sgt, inv_n_c, ic, n_c, n_s, n_m);     
            } else if (tw_type == 1){
                #pragma omp parallel for collapse(2) if(useOmp) num_threads(nt) schedule(static, 256)
                for(int it = l_start[il]; it < l_start[il+1]; it++)
                for(int ic = 0; ic < n_c; ic++)
                    twincher_backward<Mode, fp, 1>(tw[it], s, Q, B, A.data(), dL_ds, dLc_da.data()+4*(it-l_start[il])*n_c, sga, sgt, inv_n_c, ic, n_c, n_s, n_m);
            } else throw std::logic_error("shuttle_cpu::backward: unknown tw_type");
            
            // summing dLc_da
            if constexpr (Mode != INFERENCE){
                #pragma omp parallel for collapse(1) if(useOmp) num_threads(nt) schedule(static, 1)
                for(int i = 0; i < 4*(l_start[il+1] - l_start[il]); i++)
                for(int ic = 0; ic < n_c; ic++) dL_da[4*l_start[il] + i] += dLc_da[i*n_c + ic];
            }
        }
    }

    inline void compute_V(){ 
        check_initialized();
        // forward-backward pass to get V
        // input: s_in
        // output: V
        [[maybe_unused]] const int nt = n_threads();

        forward<INFERENCE>();

        // set V at the output layer
        #pragma omp parallel for collapse(3) if(useOmp) num_threads(nt) schedule(dynamic, 16384)
        for(int ip = 0; ip < n_p; ip++)
        for(int is = 0; is < n_s; is++)
        for(int ic = 0; ic < n_c; ic++)
            V[(ip*n_s + is)*n_c + ic] = (ip == is);        
        
        for(int il = n_l - 1; il >= 0; il--){
            if(tw_type == 0){
                #pragma omp parallel for collapse(2) if(useOmp) num_threads(nt) schedule(dynamic, 256)
                for(int it = l_start[il]; it < l_start[il+1]; it++)
                for(int ic = 0; ic < n_c; ic++)
                    twincher_backward_V<fp, 0>(tw[it], s, V, ic, n_p, n_s, n_c);
            } else if (tw_type == 1){
                #pragma omp parallel for collapse(2) if(useOmp) num_threads(nt) schedule(dynamic, 256)
                for(int it = l_start[il]; it < l_start[il+1]; it++)
                for(int ic = 0; ic < n_c; ic++)
                    twincher_backward_V<fp, 1>(tw[it], s, V, ic, n_p, n_s, n_c);
            } else throw std::logic_error("shuttle_cpu::compute_V: unknown tw_type");
        }
    }
    
    inline void set_tw_config(double wl, double ws, double wu){
        check_initialized();
        memcpy(TW->a.data(), a, TW->a.size()*sizeof(fp));
        twinch_config<fp> tw_config_new(wl, ws, wu);
        for(int it = 0; it < TW->n_t; it++){
            twinch_alter_config(&(TW->a[4*it]), TW->tw_config, tw_config_new);
        }
        memcpy(a, TW->a.data(), TW->a.size()*sizeof(fp));
        TW->tw_config = tw_config_new;
        tw_update(); // the derivatives with respect to a depend on the configuration
    }

    inline void tw_print(){
        check_initialized();
        for(int it = 0; it < n_t; it++){ 
            printf("%i: %i-%i (%f, %f, %f, %f)\n", it, tw[it].i0, tw[it].i1, tw[it].c0, tw[it].c1, tw[it].w, tw[it].t);
        }
    }
};
