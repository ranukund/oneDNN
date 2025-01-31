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
#include <numeric>
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

sdpa_micro init_sdpa_primitive(
        tensor Q, tensor Kt, tensor V, tensor A, tensor idx = tensor()) {
    auto sdpa_pd = sdpa_micro::primitive_desc(
            global_engine, Q.md_, Kt.md_, V.md_, A.md_, tensor().md_, idx.md_);
    auto sdpa_prim = sdpa_micro(sdpa_pd);
    return sdpa_prim;
}

int sum(const std::vector<int> &input) {
    return std::accumulate(input.begin(), input.end(), 0);
}

int max(const std::vector<int> &input) {
    return *std::max_element(input.begin(), input.end());
}

int main(int argc, char **argv) {
    global_engine = dnnl::engine(dnnl::engine::kind::gpu, 0);
    global_engine_stream = dnnl::stream(global_engine);

    // tensor subsequence_begins(
    //         {0, 35, 47}, {1, 1, 1, batch_size_in_sequences + 1});
    // tensor block_indices({0, 2, 3}, {1, 1, 1, batch_size_in_sequences + 1});
    // tensor indexes({0, 25, 1}, {1, 1, 1, batch_size_in_sequences + 1});

    // const std::vector<int> prompt_lens = {16};
    // const int batch_size_in_sequences = prompt_lens.size();
    // const int batch_size_in_tokens = ::sum(prompt_lens);
    // const int num_blocks = 10;

    // const int num_kv_heads = 1;

    // tensor query = eye({1, 1, batch_size_in_tokens, num_kv_heads * head_size});
    // tensor key_cache   = eye({num_blocks, num_kv_heads, head_size, page_size});
    // tensor value_cache = eye({num_blocks, num_kv_heads, head_size, page_size});
    // tensor output = eye({});

    // batch_size_in_tokens

    /*
      block_indices[num_blocks] (page numbers)
      block_indices_begins[batch_size_in_sequences + 1] ()
     */

    // {batch = 1, m = 32, n = 32, k = 16}

    // /* make sure render is correct */
    // std::vector<float> data(32 * 16);
    // for (int i = 0; i < data.size(); i++) {
    //   int row = i % 32, col = i / 32;
    //   if (row == col) {
    //     data[i] = 1;
    //     data[i + 16] = 1;
    //   }
    // }
    // show(tensor(data, {1, 1, 32, 16}));
    // exit(0);

    // tensor Q = eye({1, 1, 16, 16});
    // tensor Kt = eye({1, 1, 16, 32});
    // tensor V = eye({1, 1, 32, 16});
    // tensor A = zeros({1, 1, 16, 16});

    tensor Q = eye({1, 1, 32, 32});  // (D) head size 32, (Q) num_queries 32
    tensor Kt = eye({1, 1, 32, 64});  // (K) keys 64
    show(Kt);
    exit(0);
    tensor V = eye({1, 1, 64, 32});
    tensor A = zeros({1, 1, 32, 32});

    auto paged_prim = init_sdpa_primitive(Q, Kt, V, A);
    tensor paged_values = execute_sdpa_prim(paged_prim, Q, Kt, V, A);
    show(A);
}
