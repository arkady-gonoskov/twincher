// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 Arkady Gonoskov

#include "common.h"
#include "shuttle_cpu.h"

void set_rnd(double *data, int size, int rng_seed = 42){
    std::mt19937 rng(rng_seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for(int i = 0; i < size; i++) data[i] = dist(rng);
}

void perf_inference(int n_cores = 6){
    int n_s = 1024, n_l = 1024, n_c = 1024, n_p = 4, n_m = 4;
    shuttle_cpu<double> SH(n_s, n_l, n_c, n_p, n_m);
    int n_t = SH.n_t;
    double *s = new double[n_s*n_c];
    double *a = new double[4*n_t];
    double *V = new double[n_p*n_s*n_c];
    double *Q = new double[n_s*n_m*n_c];
    double *B = new double[n_s*n_m*n_c];
    double *b = new double[n_s*n_c];
    double *g = new double[4*n_t];
    SH.init(
        (std::uintptr_t)s, 
        (std::uintptr_t)a, 
        (std::uintptr_t)V, 
        (std::uintptr_t)Q, 
        (std::uintptr_t)B, 
        (std::uintptr_t)b, 
        (std::uintptr_t)g
    );
    SH.n_cores = n_cores;
    SH.set_rnd_t();

    set_rnd(s, n_s*n_c);
    
    int n_iter = 5;
    double time_best_ns = 1e+100;
    double time_av_ns = 0;
    for(int i_iter = 0; i_iter < n_iter; i_iter++){
        auto time_start = std::chrono::steady_clock::now();
        SH.forward<INFERENCE>();
        SH.backward<INFERENCE>();
        auto time_end = std::chrono::steady_clock::now();
        printf("\r %f\r               ", s[0]);
        printf("\r%i/%i", i_iter, n_iter);
        std::cout << std::flush;
        double time_ns = (1e+9)*(1/(2.0*SH.n_t*SH.n_c))*std::chrono::duration<double>(time_end - time_start).count();
    
        time_best_ns = std::min(time_best_ns, time_ns);
        time_av_ns += time_ns/n_iter;
    }
    printf("\n");
    printf("time_best_ns=%f\n", time_best_ns);
    printf("time_av_ns  =%f\n", time_av_ns);


    delete []s;
    delete []a;
    delete []V;
    delete []Q;
    delete []B;
    delete []b;
    delete []g;
}

int main(int argc, char* argv[])
{
    if(argc != 3){
        std::cout << "usage: " << argv[0] << " <mode> <n_cores>\n"
                  << "  mode 0: performance of inference (forward and backward pass)\n"
                  << "  n_cores: number of CPU threads (0: OpenMP default)\n";
        return 1;
    }
    int mode = std::stoi(argv[1]);
    int n_cores = std::stoi(argv[2]);

    std::cout << "mode = " << mode << '\n';
    std::cout << "n_cores = " << n_cores << '\n';
    
    if(mode == 0){
        perf_inference(n_cores);
    } else 
    if(mode == 1){

    } else {
        printf("Error: unknown mode\n");
    }

    //std::cout << "info()" << std::endl;
    //info();
    //printf("main\n");
    //perf_inference();
    
    return 0;
}

