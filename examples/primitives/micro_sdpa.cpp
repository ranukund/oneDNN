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

int main(int argc, char **argv) {
    global_engine = dnnl::engine(dnnl::engine::kind::gpu, 0);
    global_engine_stream = dnnl::stream(global_engine);

    /*
      future poc implementation to match openvino spec

        tensor subsequence_begins(
                {0, 35, 47}, {1, 1, 1, batch_size_in_sequences + 1});
        tensor block_indices({0, 2, 3}, {1, 1, 1, batch_size_in_sequences + 1});
        tensor indexes({0, 25, 1}, {1, 1, 1, batch_size_in_sequences + 1});

        const std::vector<int> prompt_lens = {16};
        const int batch_size_in_sequences = prompt_lens.size();
        const int batch_size_in_tokens = ::sum(prompt_lens);
        const int num_blocks = 10;

        const int num_kv_heads = 1;

        tensor query       = eye({1, 1, batch_size_in_tokens, num_kv_heads * head_size});
        tensor key_cache   = eye({num_blocks, num_kv_heads, head_size, page_size});
        tensor value_cache = eye({num_blocks, num_kv_heads, head_size, page_size});
     */

    /*
      current fused kernel dimension definitions

        q_desc.dims[q_desc.ndims - 2]; // num queries
        q_desc.dims[q_desc.ndims - 1]; // head size
        k_desc.dims[k_desc.ndims - 1]; // keys
        v_desc.dims[v_desc.ndims - 1]; // values
     */

    const int q = 4, k = 4, d = 8, v = d;

    tensor Q = eye({1, 1, q, d});
    tensor Kt = eye({1, 1, d, k});
    tensor V = eye({1, 1, k, v});
    tensor At = zeros({1, 1, q, d});

    auto paged_prim = init_sdpa_primitive(Q, Kt, V, At);
    tensor paged_values = execute_sdpa_prim(paged_prim, Q, Kt, V, At);

    show(At);

    return 0;
}
