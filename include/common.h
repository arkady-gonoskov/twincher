// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 Arkady Gonoskov

#pragma once
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdint>
#include <climits>
#include <vector>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <algorithm>
#include <stdexcept>

//#define DEBUG

#ifndef __CUDACC__ 
    #define HD
#else 
    #define HD __host__ __device__ 
#endif

template <typename T>
HD inline T sqr(T x) { return x*x; }

template <typename T>
HD inline T cube(T x) { return x*x*x; }

template <typename T>
HD inline T sgn(T x) { return (x > 0.0f) - (x < 0.0f); }

template <typename T>
void print(const std::vector<T>& v, std::string prefix = ""){
    std::cout << prefix;
    for (size_t i = 0; i < v.size(); i++)
        std::cout << v[i] << (i < v.size() - 1 ? ", " : "\n");
}

template <typename T>
void print(T* arr, int n, std::string prefix = ""){
    std::cout << prefix;
    for (int i = 0; i < n; i++)
        std::cout << arr[i] << (i < n - 1 ? ", " : "\n");
}

template<typename T>
struct tensor2_view{
    T *ptr;
    int n1, n0;
    HD tensor2_view(T *ptr, int n1, int n0): ptr(ptr), n1(n1), n0(n0) {};
    HD inline T& operator()(int i1, int i0){
        #ifdef DEBUG
            if((i0 < 0)||(i0 >= n0)||(i1 < 0)||(i1 >= n1)) 
                printf("ERROR: tensor2_view::operator(): calling (%i, %i) that is outside ranges %ix%i.\n", i1, i0, n1, n0);
        #endif
        return ptr[i1*n0 + i0];
    }
};

template<typename T>
struct tensor2_copy{
    T *data;
    int n1, n0;
    tensor2_copy(T *ptr, int n1, int n0): n1(n1), n0(n0) {
        data = new T[n1*n0];
        memcpy(&data[0], &ptr[0], sizeof(T)*n1*n0);
    };
    inline T& operator()(int i1, int i0){
        #ifdef DEBUG
            if((i0 < 0)||(i0 >= n0)||(i1 < 0)||(i1 >= n1)) 
                printf("ERROR: tensor2_copy::operator(): calling (%i, %i) that is outside ranges %ix%i.\n", i1, i0, n1, n0);
        #endif
        return data[i1*n0 + i0];
    }
    ~tensor2_copy(){delete []data;}
    tensor2_copy(const tensor2_copy&) = delete;
    tensor2_copy& operator=(const tensor2_copy&) = delete;
};

template<typename T>
struct tensor2{
    int n1, n0;
    std::vector<T> vec_;
    tensor2(int n1, int n0): n1(n1), n0(n0), vec_(n1*n0){}
    inline T& operator()(int i1, int i0){
        #ifdef DEBUG
            if((i0 < 0)||(i0 >= n0)||(i1 < 0)||(i1 >= n1)) 
                printf("ERROR: tensor2::operator(): calling (%i, %i) that is outside ranges %ix%i.\n", i1, i0, n1, n0);
        #endif
        return vec_[i1*n0 + i0];
    }
    std::size_t size() {return vec_.size();}
    std::vector<T>& vec(){ return vec_;}
    T* data(){ return vec_.data(); }
    tensor2(const tensor2&) = delete;
    tensor2& operator=(const tensor2&) = delete;
};

template<typename T>
struct tensor3_view{
    T *ptr;
    int n2, n1, n0;
    HD tensor3_view(T *ptr, int n2, int n1, int n0): ptr(ptr), n2(n2), n1(n1), n0(n0) {};
    HD inline T& operator()(int i2, int i1, int i0){
        #ifdef DEBUG
            if((i0 < 0)||(i0 >= n0)||(i1 < 0)||(i1 >= n1)||(i2 < 0)||(i2 >= n2)) 
                printf("ERROR: tensor3_view::operator(): calling (%i, %i, %i) that is outside ranges %ix%ix%i.\n", i2, i1, i0, n2, n1, n0);
        #endif
        return ptr[(i2*n1 + i1)*n0 + i0];
    }
};

template<typename T>
struct tensor3_copy{
    T *data;
    int n2, n1, n0;
    tensor3_copy(T *ptr, int n2, int n1, int n0): n2(n2), n1(n1), n0(n0) {
        data = new T[n2*n1*n0];
        memcpy(&data[0], &ptr[0], sizeof(T)*n2*n1*n0);
    };
    inline T& operator()(int i2, int i1, int i0){
        #ifdef DEBUG
            if((i0 < 0)||(i0 >= n0)||(i1 < 0)||(i1 >= n1)||(i2 < 0)||(i2 >= n2)) 
                printf("ERROR: tensor3_copy::operator(): calling (%i, %i, %i) that is outside ranges %ix%ix%i.\n", i2, i1, i0, n2, n1, n0);
        #endif
        return data[(i2*n1 + i1)*n0 + i0];
    }
    ~tensor3_copy(){delete []data;}
    tensor3_copy(const tensor3_copy&) = delete;
    tensor3_copy& operator=(const tensor3_copy&) = delete;
};

template<typename T>
struct tensor3{
    int n2, n1, n0;
    std::vector<T> vec_;
    tensor3(int n2, int n1, int n0): n2(n2), n1(n1), n0(n0), vec_(n2*n1*n0) {}
    inline T& operator()(int i2, int i1, int i0){
        #ifdef DEBUG
            if((i0 < 0)||(i0 >= n0)||(i1 < 0)||(i1 >= n1)||(i2 < 0)||(i2 >= n2)) 
                printf("ERROR: tensor3::operator(): calling (%i, %i, %i) that is outside ranges %ix%ix%i.\n", i2, i1, i0, n2, n1, n0);
        #endif
        return vec_[(i2*n1 + i1)*n0 + i0];
    }
    std::size_t size() {return vec_.size();}
    std::vector<T>& vec() {return vec_;}
    T* data(){ return vec_.data(); }
    tensor3(const tensor3&) = delete;
    tensor3& operator=(const tensor3&) = delete;
};



template<typename fp>
HD inline void inv22(fp A[2][2], fp inv_A[2][2]){ //places the inverse of A in inv_A; safe to use same parameter twice
    fp inv_det = 1/(A[0][0]*A[1][1] - A[0][1]*A[1][0]);
    fp a00 = A[0][0];
    inv_A[0][0] = inv_det*A[1][1];
    inv_A[1][1] = inv_det*a00;
    inv_A[0][1] = -inv_det*A[0][1];
    inv_A[1][0] = -inv_det*A[1][0];
};


template<typename fp>
fp max_diff(const std::vector<fp> &a, const std::vector<fp> &b){
    if(a.size() != b.size()) throw std::invalid_argument("max_diff(a, b): a.size() != b.size()");
    fp result = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        result = std::max(result, std::abs(a[i] - b[i]));
    return result;
}

template<typename fpa, typename fpb>
double max_diff(const fpa *a, const fpb *b, int n){
    double result = 0;
    for (int i = 0; i < n; ++i)
        result = std::max(result, std::abs(a[i] - b[i]));
    return result;
}

// Checks the sizes of batch computations (see shuttle_cpu, shuttle_gpu); throws std::invalid_argument
// if they are invalid, including the case of tensors too large to be indexed with int.
inline void check_batch_sizes(int n_s, int n_c, int n_p, int n_m){
    if(n_c < 1) throw std::invalid_argument("n_c must be >= 1, got " + std::to_string(n_c));
    if(n_p < 0) throw std::invalid_argument("n_p must be >= 0, got " + std::to_string(n_p));
    if(n_m < 0) throw std::invalid_argument("n_m must be >= 0, got " + std::to_string(n_m));
    long long n_max = (long long)n_s*std::max({n_p, n_m, 2})*n_c; // largest tensor size
    if(n_max > INT_MAX)
        throw std::invalid_argument("tensors of n_s*max(n_p, n_m, 2)*n_c = " + std::to_string(n_max) +
            " elements exceed the supported size of " + std::to_string(INT_MAX) + " elements");
}

inline void print_with_word_wrapping(std::string text, std::size_t width) {
    std::string line;
    std::string word;
    std::istringstream input(text);
    while (true) {
        char c = input.peek();
        if (c == EOF) break;

        if (c == '\n') {
            // Consume newline
            input.get();

            // Print current line and start fresh
            if (!line.empty()) {
                std::cout << line << std::endl;
                line.clear();
            } else {
                // Empty line = explicit blank line
                std::cout << std::endl;
            }
            continue;
        }
        // Read next word normally
        input >> word;
        if (!input) break;

        if (line.empty()) {
            line = word;
        } else if (line.size() + 1 + word.size() <= width) {
            line += " " + word;
        } else {
            std::cout << line << std::endl;
            line = word;
        }
    }
    // Print any remaining text in the last line
    if (!line.empty()) {
        std::cout << line << std::endl;
    }
}

// TWINCHER_VERSION is normally defined by the build system (see CMakeLists.txt)
#ifndef TWINCHER_VERSION
    #define TWINCHER_VERSION "dev"
#endif

inline void info(){
    std::string statement =
    "--------------------------------------------------------------------------------\n"
    "Twincher " TWINCHER_VERSION "\n"
    "Copyright (C) 2025-2026 Arkady Gonoskov\n"
    "\n"
    "This program is free software: you can redistribute it and/or modify it under "
    "the terms of the GNU Affero General Public License, version 3, as published by "
    "the Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the "
    "license for details: https://www.gnu.org/licenses/agpl-3.0.html\n"
    "\n"
    "Commercial licenses are available on request.\n"
    "Website: https://www.twincher.ai\n"
    "Contact: contact@twincher.ai\n"
    "--------------------------------------------------------------------------------\n";
    print_with_word_wrapping(statement, 80);
}