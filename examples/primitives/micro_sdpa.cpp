/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
#include "tensor_utils.hpp"

tensor sdpa(tensor Q, tensor Kt, tensor V, const bool benchmark = false,
        tensor idx = tensor()) {

    memory::dims Q_dims = Q.md_.get_dims(), Kt_dims = Kt.md_.get_dims(),
                 V_dims = V.md_.get_dims();

    const memory::dims dst_shape = {1, 1, Q_dims[2], V_dims[3]};
    tensor dst = zeros(dst_shape);

    auto sdpa_pd = sdpa_micro::primitive_desc(global_engine, Q.md_, Kt.md_,
            V.md_, dst.md_, tensor().md_, idx.md_);

    auto sdpa_prim = sdpa_micro(sdpa_pd);

    std::unordered_map<int, memory> sdpa_args;
    sdpa_args.insert({DNNL_ARG_QUERIES, Q.mem_});
    sdpa_args.insert({DNNL_ARG_KEYS, Kt.mem_});
    sdpa_args.insert({DNNL_ARG_VALUES, V.mem_});
    sdpa_args.insert({DNNL_ARG_DST, dst.mem_});
    if (idx.md_.get_data_type() != dt::undef) {
        sdpa_args.insert({DNNL_ARG_INDICES, idx.mem_});
    }
    sdpa_prim.execute(global_engine_stream, sdpa_args);
    global_engine_stream.wait();

    if (benchmark) {
        tic();
        for (int i = 0; i < 100; i++) {
            sdpa_prim.execute(global_engine_stream, sdpa_args);
        }
        global_engine_stream.wait();
        toc();
    }

    return dst;
}

tensor execute_sdpa_prim(sdpa_micro sdpa_prim, tensor Q, tensor Kt, tensor V,
        tensor A, tensor idx = tensor()) {
    std::unordered_map<int, memory> sdpa_args;
    sdpa_args.insert({DNNL_ARG_QUERIES, Q.mem_});
    sdpa_args.insert({DNNL_ARG_KEYS, Kt.mem_});
    sdpa_args.insert({DNNL_ARG_VALUES, V.mem_});
    sdpa_args.insert({DNNL_ARG_DST, A.mem_});
    if (idx.md_.get_data_type() != dt::undef) {
        sdpa_args.insert({DNNL_ARG_INDICES, idx.mem_});
    }
    sdpa_prim.execute(global_engine_stream, sdpa_args);
    global_engine_stream.wait();
    return A;
}

sdpa_micro init_sdpa_primitive(tensor Q, tensor Kt, tensor V, tensor A,
        tensor idx = tensor()) {
    auto sdpa_pd = sdpa_micro::primitive_desc(
            global_engine, Q.md_, Kt.md_, V.md_, A.md_, tensor().md_, idx.md_);
    auto sdpa_prim = sdpa_micro(sdpa_pd);
    return sdpa_prim;
}

int main(int argc, char **argv) {
    global_engine = dnnl::engine(dnnl::engine::kind::gpu, 0);
    global_engine_stream = dnnl::stream(global_engine);

    const int cached_pages = 3, num_pages = 2;
    const int num_queries = 32, head_size = 32, page_size = 32;

    // clang-format off
    //                  [total_num_queries, num_heads * head_size]
    tensor Q =   eye({           1, 1, num_queries,   head_size});
    // tensor Kt =  eye({cached_pages, 1,   head_size,   page_size});
    // tensor V =   eye({cached_pages, 1,   head_size,   page_size});
    // tensor A = zeros({   num_pages, 1,   head_size, num_queries});

    // tensor page_idxs = tensor({0, 0}, {1, 1, 1, num_pages});
    // page_idxs = cast(page_idxs, dt::s32);
    // // tensor block_indices_begins;
    // // tensor indexes;
    // // clang-format on

    // auto paged_prim = init_sdpa_primitive(Q, Kt, V, A, page_idxs);
    // tensor paged_values = execute_sdpa_prim(paged_prim, Q, Kt, V, A, page_idxs);

    // show(A);
}
