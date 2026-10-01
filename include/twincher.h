// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 Arkady Gonoskov

#pragma once
#include "common.h"
#include "twinch.h"

template<typename fp>
struct twincher{
    int n_s; // size of s vector
    int n_l; // number of layers 
    int tw_type; //type of twinch
    int n_t; // total number of twinches
    twinch_config<fp> tw_config;
    std::vector<fp> a; // parameters of twinches
    std::vector<int> i0, i1; // involved indices for a given twinch
    std::vector<int> l_start; // index that starts given layer
    twincher(int n_s = 2, int n_l = 1, int tw_type = 1, int rng_seed = 42): 
    n_s(n_s), n_l(n_l), tw_type(tw_type)
    {
        if(n_s < 2) throw std::invalid_argument("n_s must be >= 2, got " + std::to_string(n_s));
        if(n_l < 1) throw std::invalid_argument("n_l must be >= 1, got " + std::to_string(n_l));
        if(tw_type != 0 && tw_type != 1)
            throw std::invalid_argument("tw_type must be 0 or 1, got " + std::to_string(tw_type));
        if(4LL*(n_s/2)*n_l > INT_MAX)
            throw std::invalid_argument("the number of parameters 4*(n_s/2)*n_l exceeds " + std::to_string(INT_MAX));
        int n_s_2 = n_s/2;
        n_t = n_s_2*n_l;
        a.resize(4*n_t);
        i0.resize(n_t);
        i1.resize(n_t);
        l_start.resize(n_l + 1);
        l_start[n_l] = n_t; 
        for(int il = 0; il < n_l; il++){
            int stage = il%(n_s - 1);
            l_start[il] = il*n_s_2;
            for(int it = 0; it < n_s_2; it++){
                if(it == 0){
                    i0[l_start[il] + it] = stage;
                    i1[l_start[il] + it] = n_s - 1;
                } else {
                    i0[l_start[il] + it] = (n_s_2 - it + stage)%(n_s - 1);
                    i1[l_start[il] + it] = (n_s_2 + it - 1 + stage)%(n_s - 1);
                }
            }
        }
        set_cold(rng_seed);
    }
    inline void set_cold(int rng_seed = 42){
        std::mt19937 rng(rng_seed);
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        fp inv_sqrt3 = 1/sqrt(3.0);
        for(int i = 0; i < n_t; i++){ 
            a[4*i + 0] = inv_sqrt3*(-1 + 2*U01(rng));       // c0
            a[4*i + 1] = inv_sqrt3*(-1 + 2*U01(rng));       // c1
            a[4*i + 2] = -0.333 + (0.5 + 0.333)*U01(rng);   // w
            a[4*i + 3] = 0; //-1 + 2*U01(rng);              // t
        }
    }
    inline void print_state(){
        for(int il = 0; il < n_l; il++){
            std::cout << std::setw(3) << il << "|";
            for(int it = l_start[il]; it < l_start[il + 1]; it++) 
                std::cout << std::setw(3) << std::right << i0[it] << "-" << std::setw(3) << std::left << i1[it] << "|";
            std::cout << std::right << std::endl;
        }       
    }
    inline void save(std::fstream &file){
        if(4*n_t != int(a.size())) throw std::logic_error("twincher::save: 4*n_t != a.size()");

        file.write(reinterpret_cast<char*>(&tw_type), sizeof(int));
        file.write(reinterpret_cast<char*>(&n_s), sizeof(int));
        file.write(reinterpret_cast<char*>(&n_l), sizeof(int));
        file.write(reinterpret_cast<char*>(&n_t), sizeof(int));
        int fp_size = sizeof(fp);
        file.write(reinterpret_cast<char*>(&fp_size), sizeof(int));
        std::vector<fp> twv; tw_config.to_vec(twv);
        file.write(reinterpret_cast<char*>(twv.data()), twv.size()*sizeof(fp));
        file.write(reinterpret_cast<char*>(a.data()), a.size()*sizeof(fp));
        file.write(reinterpret_cast<char*>(i0.data()), i0.size()*sizeof(int));
        file.write(reinterpret_cast<char*>(i1.data()), i1.size()*sizeof(int));
        file.write(reinterpret_cast<char*>(l_start.data()), l_start.size()*sizeof(int));
        if(!file) throw std::runtime_error("twincher::save: error while writing the file");
    }
    inline void load(std::fstream &file){
        auto read = [&file](void *data, std::size_t n_bytes){
            file.read(reinterpret_cast<char*>(data), n_bytes);
            if(!file) throw std::runtime_error(
                "twincher::load: unexpected end of file (the file is truncated or is not a twincher file)");
        };
        auto corrupted = [](const std::string &what){
            return std::runtime_error("twincher::load: " + what + " (the file is corrupted or is not a twincher file)");
        };
        read(&tw_type, sizeof(int));
        read(&n_s, sizeof(int));
        read(&n_l, sizeof(int));
        read(&n_t, sizeof(int));
        int fp_size;
        read(&fp_size, sizeof(int));
        if((tw_type != 0 && tw_type != 1) || n_s < 2 || n_l < 1 || n_t < 0 || n_t > (n_s/2)*n_l || (fp_size != 4 && fp_size != 8))
            throw corrupted("invalid header");
        i0.resize(n_t);
        i1.resize(n_t);
        l_start.resize(n_l + 1);
        a.resize(4*n_t);
        std::vector<fp> twv; tw_config.to_vec(twv);
        if(int(sizeof(fp)) == fp_size){
            read(twv.data(), twv.size()*sizeof(fp));
            read(a.data(), a.size()*sizeof(fp));
        } else if(fp_size == 4){
            std::vector<float> t(twv.size()), tmp(4*n_t);
            read(t.data(), t.size()*sizeof(float));
            read(tmp.data(), tmp.size()*sizeof(float));
            for(std::size_t i = 0; i < t.size(); i++) twv[i] = t[i];
            for(int i = 0; i < 4*n_t; i++) a[i] = tmp[i];
        } else {
            std::vector<double> t(twv.size()), tmp(4*n_t);
            read(t.data(), t.size()*sizeof(double));
            read(tmp.data(), tmp.size()*sizeof(double));
            for(std::size_t i = 0; i < t.size(); i++) twv[i] = t[i];
            for(int i = 0; i < 4*n_t; i++) a[i] = tmp[i];
        }
        tw_config.from_vec(twv);
        tw_config.validate();
        read(i0.data(), i0.size()*sizeof(int));
        read(i1.data(), i1.size()*sizeof(int));
        read(l_start.data(), l_start.size()*sizeof(int));

        // check the wiring: layers are consecutive ranges of twinches, and within a layer
        // each s component is affected by at most one twinch (required for parallel updates)
        if(l_start[0] != 0 || l_start[n_l] != n_t) throw corrupted("invalid layer structure");
        std::vector<int> used_in_layer(n_s, -1);
        for(int il = 0; il < n_l; il++){
            if(l_start[il + 1] < l_start[il] || l_start[il + 1] - l_start[il] > n_s/2)
                throw corrupted("invalid layer structure");
            for(int it = l_start[il]; it < l_start[il + 1]; it++){
                for(int is: {i0[it], i1[it]}){
                    if(is < 0 || is >= n_s || used_in_layer[is] == il) throw corrupted("invalid wiring");
                    used_in_layer[is] = il;
                }
            }
        }
    }
    inline void save(std::string file_name, bool append = false){
        auto mode = std::ios::out | std::ios::binary;
        if(append) mode |= std::ios::app;
        std::fstream file(file_name, mode);
        if(!file) throw std::runtime_error("cannot open file '" + file_name + "' for writing");
        save(file);
    }
    inline void load(std::string file_name, int offset = 0){
        std::fstream file(file_name, std::ios::in | std::ios::binary);
        if(!file) throw std::runtime_error("cannot open file '" + file_name + "' for reading");
        if(offset != 0) file.seekg(offset);
        load(file);
    }
};