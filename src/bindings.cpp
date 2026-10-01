// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 Arkady Gonoskov

#include <pybind11/pybind11.h>
#include "shuttle_cpu.h"
#ifdef TWINCHER_WITH_CUDA
	#include "shuttle_gpu.hpp"
#endif

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
	m.def("info", &info, "info");
#ifdef TWINCHER_WITH_CUDA
	m.attr("cuda_enabled") = true;
	m.def("gpu_available", &gpu_available);
#else
	m.attr("cuda_enabled") = false;
	m.def("gpu_available", []() { return false; });
#endif

	py::class_<shuttle_cpu<double>>(m, "ShuttleCPU")
	//=================== constructors =====================
		.def(py::init<int, int, int, int, int, int, int>(), 
		py::arg("n_s"), py::arg("n_l"), py::arg("n_c"), py::arg("n_p"), py::arg("n_m"), 
		py::arg("tw_type") = 1, py::arg("rng_seed") = 42)
		
		.def(py::init<std::string, int, int, int, int>(), 
		py::arg("file_name"), py::arg("n_c"), py::arg("n_p"), py::arg("n_m"), py::arg("file_offset") = 0)
	
	//================= main parameters ====================
		.def_readonly("n_s", &shuttle_cpu<double>::n_s)
		.def_readonly("n_l", &shuttle_cpu<double>::n_l)
		.def_readonly("n_c", &shuttle_cpu<double>::n_c)
		.def_readonly("n_p", &shuttle_cpu<double>::n_p)
		.def_property("n_m", [](const shuttle_cpu<double> &sh){ return sh.n_m; }, &shuttle_cpu<double>::set_n_m)
		.def_readonly("tw_type", &shuttle_cpu<double>::tw_type)
		.def_readonly("n_t", &shuttle_cpu<double>::n_t)
	
	//============= configuration variables ================
		.def_readwrite("use_omp", &shuttle_cpu<double>::useOmp)
		.def_readwrite("n_cores", &shuttle_cpu<double>::n_cores)
		.def_readwrite("sga", &shuttle_cpu<double>::sga)
		.def_readwrite("sgt", &shuttle_cpu<double>::sgt)

	//==================== functions =======================
		.def("tw_save", &shuttle_cpu<double>::tw_save, py::arg("file_name"), py::arg("append") = false)
		.def("set_rnd_t", &shuttle_cpu<double>::set_rnd_t, py::arg("rng_seed") = 42)

		.def("forward_inference", &shuttle_cpu<double>::forward<INFERENCE>)
		.def("backward_inference", &shuttle_cpu<double>::backward<INFERENCE>)
		.def("tw_update", &shuttle_cpu<double>::tw_update)
		.def("compute_V", &shuttle_cpu<double>::compute_V)
		.def("forward_query", &shuttle_cpu<double>::forward<QUERY>)
		.def("backward_query", &shuttle_cpu<double>::backward<QUERY>)
		.def("set_tw_config", &shuttle_cpu<double>::set_tw_config, py::arg("wl") = 0.01, py::arg("ws") = 0.1, py::arg("wu") = 1.3)

		.def("tw_print", &shuttle_cpu<double>::tw_print)

	//================= initialization =====================
		.def("init", &shuttle_cpu<double>::init, 
			py::arg("s_ptr"), py::arg("a_ptr"), py::arg("V_ptr"), py::arg("Q_ptr"), py::arg("B_ptr"), py::arg("b_ptr"), py::arg("g_ptr"))
	;
	
//==============================GPU==================================
#ifdef TWINCHER_WITH_CUDA
		py::class_<shuttle_gpu>(m, "ShuttleGPU")
	//=================== constructors =====================
		.def(py::init<int, int, int, int, int, int, int>(), 
		py::arg("n_s"), py::arg("n_l"), py::arg("n_c"), py::arg("n_p"), py::arg("n_m"), 
		py::arg("tw_type") = 1, py::arg("rng_seed") = 42)
	
		.def(py::init<std::string, int, int, int, int>(), 
		py::arg("file_name"), py::arg("n_c"), py::arg("n_p"), py::arg("n_m"), py::arg("file_offset") = 0)
	

		//================= main parameters ====================
		.def_readonly("n_s", &shuttle_gpu::n_s)
		.def_readonly("n_l", &shuttle_gpu::n_l)
		.def_readonly("n_c", &shuttle_gpu::n_c)
		.def_readonly("n_p", &shuttle_gpu::n_p)
		.def_property("n_m", [](const shuttle_gpu &sh){ return sh.n_m; }, &shuttle_gpu::set_n_m)
		.def_readonly("tw_type", &shuttle_gpu::tw_type)
		.def_readonly("n_t", &shuttle_gpu::n_t)

		//================= initialization =====================
		.def("init", &shuttle_gpu::init, 
			py::arg("s_ptr"), py::arg("a_ptr"), py::arg("V_ptr"), py::arg("Q_ptr"), py::arg("B_ptr"), py::arg("b_ptr"), py::arg("g_ptr"))

		//============= configuration variables ================
		.def_readwrite("n_threads", &shuttle_gpu::n_threads)
		.def_readwrite("cuda_stream", &shuttle_gpu::cuda_stream_ptr)
		.def_readwrite("sga", &shuttle_gpu::sga)
		.def_readwrite("sgt", &shuttle_gpu::sgt)

		//==================== functions =======================
		.def("tw_save", &shuttle_gpu::tw_save, py::arg("file_name"), py::arg("append") = false)
		.def("set_rnd_t", &shuttle_gpu::set_rnd_t, py::arg("rng_seed") = 42)
		.def("forward_inference", &shuttle_gpu::forward<INFERENCE>)
		.def("backward_inference", &shuttle_gpu::backward<INFERENCE>)
		.def("tw_update", &shuttle_gpu::tw_update)
		.def("compute_V", &shuttle_gpu::compute_V)
		.def("forward_query", &shuttle_gpu::forward<QUERY>)
		.def("backward_query", &shuttle_gpu::backward<QUERY>)
		.def("set_tw_config", &shuttle_gpu::set_tw_config, py::arg("wl") = 0.01, py::arg("ws") = 0.1, py::arg("wu") = 1.3)
	;
#endif
}
