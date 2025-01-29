/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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

#include <assert.h>
#include "oneapi/dnnl/dnnl.h"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

#include "c_types_map.hpp"
#include "memory_desc_wrapper.hpp"
#include "utils.hpp"

#include "sdpa_types.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

status_t dnnl_micro_sdpa_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *qry_desc, const memory_desc_t *key_desc,
        const memory_desc_t *val_desc, const memory_desc_t *dst_desc,
        const memory_desc_t *msk_desc, const memory_desc_t *idx_desc) {

    auto sdpa_desc = sdpa_desc_t();
    sdpa_desc.q_desc = *qry_desc;
    sdpa_desc.k_desc = *key_desc;
    sdpa_desc.v_desc = *val_desc;
    sdpa_desc.dst_desc = *dst_desc;
    sdpa_desc.attn_mask_desc = *msk_desc;
    sdpa_desc.kv_head_number = 1;

    sdpa_desc.idx_desc = *idx_desc;
    sdpa_desc.utilize_idx
            = (sdpa_desc.idx_desc.data_type != dnnl_data_type_undef);

    sdpa_desc.primitive_kind = primitive_kind::sdpa;
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&sdpa_desc, nullptr, &(default_attr()));
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
