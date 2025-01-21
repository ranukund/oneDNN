/*******************************************************************************
* Copyright 2017 - 2025 Intel Corporation
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
#include <functional>
#include <new>

#include "oneapi/dnnl/dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"

#include "cpu/x64/jit_uni_pooling.hpp"
#include "cpu/x64/jit_uni_reorder.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace jit_uni_pooling_utils {

static status_t set_binary_postops_formats(
        post_ops_t &post_ops, const memory_desc_t *dst_md) {
    for (int idx = 0; idx < post_ops.len(); ++idx) {
        if (!post_ops.contain(primitive_kind::binary, idx)) continue;

        auto &src1_md = post_ops.entry_[idx].binary.src1_desc;
        const memory_desc_wrapper src1_mdw(src1_md);
        if (!src1_mdw.format_any()) {
            if (src1_mdw.is_blocking_desc())
                continue;
            else
                return status::unimplemented;
        }

        const memory_desc_wrapper dst_mdw(dst_md);
        assert(!dst_mdw.format_any());

        CHECK(memory_desc_init_by_blocking_desc(
                src1_md, dst_mdw.blocking_desc()));
    }

    return status::success;
}

static bool post_ops_ok(cpu_isa_t isa, jit_pool_conf_t &jpp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d) {
    const auto &post_ops = attr.post_ops_;
    const auto &entries = post_ops.entry_;
    jpp.with_postops = false;
    jpp.with_eltwise = false;
    jpp.with_binary = false;

    if (!jpp.is_backward) {
        for (const auto &entry : entries) {
            if (entry.is_eltwise()) {
                const auto alg = entry.eltwise.alg;
                jpp.with_eltwise = eltwise_injector::is_supported(
                        isa, alg, data_type::f32);
            } else if (entry.is_binary()) {
                const bool is_bf16_ok = IMPLICATION(
                        entry.binary.src1_desc.data_type == data_type::bf16,
                        utils::one_of(isa, avx512_core, avx2_vnni_2));
                const bool is_f16_ok = IMPLICATION(
                        entry.binary.src1_desc.data_type == data_type::f16,
                        utils::one_of(isa, avx512_core_fp16, avx2_vnni_2));
                const bool is_fp8_ok = IMPLICATION(
                        utils::one_of(entry.binary.src1_desc.data_type,
                                data_type::f8_e5m2, data_type::f8_e4m3),
                        utils::one_of(isa, avx512_core_fp16));
                if (!(is_bf16_ok && is_f16_ok && is_fp8_ok)) return false;

                jpp.with_binary = true;
            } else
                return false;
        }

        jpp.with_postops = jpp.with_eltwise || jpp.with_binary;
    }

    return binary_injector::binary_args_broadcast_supported(
            post_ops, dst_d, get_supported_bcast_strategies());
}

static status_t init_conf(cpu_isa_t isa, jit_pool_conf_t &jpp,
        primitive_attr_t &attr, const pooling_pd_t *ppd) {

    using namespace alg_kind;

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(
            ppd->is_fwd() ? ppd->src_md() : ppd->diff_src_md());
    const memory_desc_wrapper dst_d(
            ppd->is_fwd() ? ppd->dst_md() : ppd->diff_dst_md());

    const int ndims = src_d.ndims();

    jpp.nthr = dnnl_get_max_threads();
    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
    jpp.is_backward = pd.prop_kind == prop_kind::backward_data;

    jpp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jpp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];
    jpp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jpp.ow = dst_d.dims()[ndims - 1];
    jpp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];

    const bool is_avx512 = is_superset(isa, avx512_core);
    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];
    jpp.c_without_padding = src_d.dims()[1];
    jpp.c_block = is_avx512 ? 16 : 8;

    jpp.alg = pd.alg_kind;

    jpp.src_dt = jpp.is_backward ? pd.diff_src_desc.data_type
                                 : pd.src_desc.data_type;
    jpp.dst_dt = jpp.is_backward ? pd.diff_dst_desc.data_type
                                 : pd.dst_desc.data_type;

    jpp.tmp_md = memory_desc_t();

    jpp.is_bf16 = (src_d.data_type() == data_type::bf16
            && dst_d.data_type() == data_type::bf16);
    jpp.is_f16 = (src_d.data_type() == data_type::f16
            && dst_d.data_type() == data_type::f16);
    jpp.is_fp8 = utils::one_of(src_d.data_type(), data_type::f8_e5m2,
                         data_type::f8_e4m3)
            && utils::one_of(
                    dst_d.data_type(), data_type::f8_e5m2, data_type::f8_e4m3);

    using namespace format_tag;

    const auto blocked_fmt_tag = is_avx512
            ? utils::pick(ndims - 3, nCw16c, nChw16c, nCdhw16c)
            : utils::pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);

    // src_d.data_type() is equal to dst_d.data_type(). This is checked in init
    auto ncsp_fmt_tag = format_tag::undef;

    const unsigned int L3_cache_size_per_core
            = platform::get_per_core_cache_size(3);
    const size_t block_size
            = ((size_t)jpp.id * jpp.ih * jpp.iw + jpp.od * jpp.oh * jpp.ow)
            * jpp.c_block * types::data_type_size(src_d.data_type());

    const bool forward_ncsp_allowed = !jpp.is_backward
            && jpp.c_without_padding > 3
            && ((jpp.ih > 1 && jpp.iw > 1
                        && block_size <= L3_cache_size_per_core)
                    || utils::one_of(src_d.data_type(), data_type::bf16,
                            data_type::f16, data_type::f8_e5m2,
                            data_type::f8_e4m3));

    const bool backward_ncsp_allowed = jpp.is_backward
            && ((jpp.ih > 1 && jpp.iw > 1 && jpp.c_without_padding > 1
                        && block_size <= L3_cache_size_per_core)
                    || (utils::one_of(src_d.data_type(), data_type::bf16,
                                data_type::f16)
                            && !(jpp.alg == pooling_max
                                    && block_size > L3_cache_size_per_core)));

    ncsp_fmt_tag = ((forward_ncsp_allowed || backward_ncsp_allowed) && is_avx512
                           && ndims <= 5)
            ? utils::pick(ndims - 3, ncw, nchw, ncdhw)
            : format_tag::undef;

    const auto nspc_fmt_tag = (ndims <= 5)
            ? utils::pick(ndims - 3, nwc, nhwc, ndhwc)
            : format_tag::undef;

    const auto fmt_tag = src_d.matches_one_of_tag(
            blocked_fmt_tag, ncsp_fmt_tag, nspc_fmt_tag);

    VDISPATCH_POOLING_IC(
            dst_d.matches_tag(fmt_tag), VERBOSE_UNSUPPORTED_TAG_S, "dst");

    VDISPATCH_POOLING_IC(
            post_ops_ok(isa, jpp, attr, dst_d), VERBOSE_UNSUPPORTED_POSTOP);

    if (fmt_tag == ncsp_fmt_tag) {
        // transform input to blocked f32, call f32 jit, transform result to
        // plain output
        jpp.is_bf16 = false;
        jpp.is_f16 = false;
        jpp.is_fp8 = false;
        jpp.dt_size = types::data_type_size(data_type::f32);
        jpp.tag_kind = jit_memory_tag_kind_t::ncsp;

        // used to initialize binary post-ops
        if (ppd->is_fwd() && jpp.with_binary) {
            CHECK(memory_desc_init_by_tag(jpp.tmp_md, ndims, dst_d.md_->dims,
                    data_type::f32, blocked_fmt_tag));
        }
    } else {
        jpp.dt_size = types::data_type_size(src_d.data_type());
        jpp.tag_kind = (fmt_tag == nspc_fmt_tag)
                ? jit_memory_tag_kind_t::nspc
                : jit_memory_tag_kind_t::blocked;
    }

    if (ppd->is_fwd() && jpp.with_binary) {
        CHECK(set_binary_postops_formats(attr.post_ops_,
                jpp.tag_kind == jit_memory_tag_kind_t::ncsp ? &jpp.tmp_md
                                                            : dst_d.md_));
    }

    jpp.isa = (jpp.is_bf16 && mayiuse(avx512_core_bf16))
            ? avx512_core_bf16
            : ((jpp.is_fp8 && mayiuse(avx512_core_fp16)) ? avx512_core_fp16
                                                         : isa);

    // disabling verbose dispatch messages for unsupported isa for
    // better readability
    if (!mayiuse(isa)) return status::unimplemented;

    VDISPATCH_POOLING_IC(
            (fmt_tag != format_tag::undef), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_POOLING_IC(IMPLICATION(jpp.is_bf16,
                                 utils::one_of(jpp.isa, avx512_core_bf16,
                                         avx512_core, avx2_vnni_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_POOLING_IC(
            IMPLICATION(jpp.is_f16,
                    utils::one_of(jpp.isa, avx512_core_fp16, avx2_vnni_2)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_POOLING_IC(
            IMPLICATION(jpp.is_fp8, utils::one_of(jpp.isa, avx512_core_fp16)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_POOLING_IC(
            utils::one_of(pd.alg_kind, pooling_max, pooling_avg_include_padding,
                    pooling_avg_exclude_padding),
            VERBOSE_BAD_ALGORITHM);

    const bool is_xf16_avx2_vnni_2
            = (jpp.is_bf16 || jpp.is_f16) && isa == avx2_vnni_2;
    // note: avx2_vnni_2 only supports nxc format
    VDISPATCH_POOLING_IC(IMPLICATION(is_xf16_avx2_vnni_2,
                                 jpp.tag_kind == jit_memory_tag_kind_t::nspc),
            "isa, format tag mismatch");

    // note: avx2_vnni_2 only supports FWD direction
    VDISPATCH_POOLING_IC(IMPLICATION(is_xf16_avx2_vnni_2, !jpp.is_backward),
            "isa, propagation kind mismatch");

    jpp.c = jpp.tag_kind == jit_memory_tag_kind_t::blocked
            ? utils::rnd_up(jpp.c_without_padding, jpp.c_block)
            : jpp.c_without_padding;
    if (jpp.tag_kind == jit_memory_tag_kind_t::blocked)
        assert(src_d.padded_dims()[1] == jpp.c);
    jpp.nb_c = utils::div_up(jpp.c, jpp.c_block);
    jpp.c_tail = jpp.c_without_padding % jpp.c_block;
    jpp.is_c_padded = jpp.tag_kind == jit_memory_tag_kind_t::blocked
            && src_d.padded_dims()[1] != jpp.c_without_padding;

    jpp.stride_d = (ndims == 5) ? pd.strides[0] : 1;
    jpp.stride_h = (ndims == 3) ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];
    jpp.kd = (ndims == 5) ? pd.kernel[0] : 1;
    jpp.kh = (ndims == 3) ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];

    jpp.f_pad = (ndims == 5) ? pd.padding[0][0] : 0;
    jpp.t_pad = (ndims == 3) ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    const int back_pad = calculate_end_padding(
            jpp.f_pad, jpp.od, jpp.id, jpp.stride_d, jpp.kd);
    const int bottom_pad = calculate_end_padding(
            jpp.t_pad, jpp.oh, jpp.ih, jpp.stride_h, jpp.kh);
    const int right_pad = calculate_end_padding(
            jpp.l_pad, jpp.ow, jpp.iw, jpp.stride_w, jpp.kw);

    VDISPATCH_POOLING_IC(
            !(jpp.f_pad >= jpp.kd || jpp.t_pad >= jpp.kh || jpp.l_pad >= jpp.kw
                    || back_pad >= jpp.kd || bottom_pad >= jpp.kh
                    || right_pad >= jpp.kw),
            VERBOSE_UNSUPPORTED_PAD_FEATURE, "");

    jpp.ind_dt = ppd->workspace_md() ? ppd->workspace_md()->data_type
                                     : data_type::undef;

    jpp.simple_alg = jpp.is_training
            || IMPLICATION(jpp.is_backward, jpp.kd <= jpp.stride_d);

    jpp.ur = 0;
    if (jpp.alg == pooling_max) {
        jpp.ur = is_avx512 ? 16 : 4;

        if (utils::one_of(isa, avx, avx2, avx2_vnni_2) && jpp.c_tail > 0)
            // Additional register needed for tail mask
            jpp.ur -= 1;

        if (jpp.is_training)
            jpp.ur = is_avx512 ? 9 : 3;
        else if (jpp.is_backward)
            jpp.ur = is_avx512 ? 6 : 3;
    } else {
        if (jpp.is_backward)
            jpp.ur = is_avx512 ? 12 : 6;
        else
            jpp.ur = is_avx512 ? 24 : 12;
    }
    if ((jpp.is_bf16 || jpp.is_f16) && isa != avx2_vnni_2) {
        jpp.ur = (!isa_has_bf16(jpp.isa))
                ? jpp.ur - 4 // Free registers for AVX512 emulation
                : jpp.ur - 1; // Free register for cvt from bf16/f16 to f32
    }

    if (jpp.is_fp8) {
        // TODO: Optimize the ur if native FP8 support is available
        jpp.ur = jpp.ur - 4;
    }
    assert(jpp.ur > 0);

    const bool is_relaxed_acc = utils::one_of(
            attr.acc_mode_, accumulation_mode::relaxed, accumulation_mode::any);
    jpp.needs_f32_accum_for_bf16 = !is_relaxed_acc && jpp.is_bf16
            && jpp.alg == alg_kind::pooling_max && jpp.is_backward
            && (jpp.stride_d < jpp.kd || jpp.stride_h < jpp.kh
                    || jpp.stride_w < jpp.kw);

    // select jpp.ur_bc
    if (jpp.tag_kind == jit_memory_tag_kind_t::nspc) {
        auto min_ur_w = nstl::max(1, utils::div_up(jpp.l_pad, jpp.stride_w));
        int min_ur_w1 = utils::div_up(right_pad, jpp.stride_w);
        if (min_ur_w < min_ur_w1) { min_ur_w = min_ur_w1; }
        jpp.ur_bc = nstl::min(jpp.nb_c, nstl::max(1, jpp.ur / min_ur_w));
        //take into account threading - to have enough work for parallelization
        float best_eff = 0;
        for (int ur_bc = jpp.ur_bc; ur_bc > 0; ur_bc--) {

            const auto nb2_c = utils::div_up(jpp.nb_c, ur_bc);
            auto work = jpp.is_backward
                    ? (ndims == 5 && jpp.simple_alg ? jpp.od : 1)
                    : (ndims == 5 ? jpp.od : jpp.oh);
            work *= jpp.mb * nb2_c;
            auto eff = (float)work / utils::rnd_up(work, jpp.nthr);
            if (eff > best_eff) {

                best_eff = eff;
                jpp.ur_bc = ur_bc;
            }
            if (eff > 0.9f) break; // Heuristic threshold
        }

        //take into account cache re-usage after zeroing on backward
        if (jpp.is_backward && ndims < 5 && !jpp.needs_f32_accum_for_bf16) {
            const int L2 = platform::get_per_core_cache_size(2) / jpp.dt_size;
            int ur_bc = nstl::max(1, L2 / (jpp.kh * jpp.iw * jpp.c_block));
            jpp.ur_bc = nstl::min(jpp.ur_bc, ur_bc);
        }

        jpp.ur_bc_tail = jpp.nb_c % jpp.ur_bc;
    } else {
        jpp.ur_bc = 1;
        jpp.ur_bc_tail = 0;
    }

    jpp.f32_accum_block_size = jpp.ur_bc * jpp.c_block;
    if (jpp.needs_f32_accum_for_bf16) {
        assert(memory_desc_wrapper(jpp.tmp_md).is_zero()
                && (fmt_tag == nspc_fmt_tag || fmt_tag == blocked_fmt_tag));

        dims_t dims {};
        utils::array_copy(dims, src_d.dims(), ndims);

        const auto nb2_c = utils::div_up(jpp.nb_c, jpp.ur_bc);
        dims[0] = nstl::min(dnnl_get_max_threads(), jpp.mb * nb2_c);
        dims[1] = jpp.f32_accum_block_size;

        memory_desc_init_by_tag(
                jpp.tmp_md, ndims, dims, data_type::f32, fmt_tag);
    }

    jpp.post_ops = attr.post_ops_;

    return status::success;
}

static void init_scratchpad(
        jit_pool_conf_t const &jpp, memory_tracking::registrar_t &scratchpad) {

    // scratchpad for c_block slice of input and/or output
    using namespace memory_tracking::names;
    const int nscr = nstl::min(dnnl_get_max_threads(), jpp.mb * jpp.nb_c);
    if (jpp.tag_kind == jit_memory_tag_kind_t::ncsp) {
        scratchpad.book(key_pool_src_plain2blocked_cvt,
                static_cast<size_t>(jpp.c_block) * jpp.id * jpp.ih * jpp.iw
                        * nscr,
                jpp.dt_size);
        scratchpad.book(key_pool_dst_plain2blocked_cvt,
                static_cast<size_t>(jpp.c_block) * jpp.od * jpp.oh * jpp.ow
                        * nscr,
                jpp.dt_size);
        scratchpad.book<uint32_t>(key_pool_ind_plain2blocked_cvt,
                static_cast<size_t>(jpp.c_block) * jpp.od * jpp.oh * jpp.ow
                        * nscr);
    }

    if (jpp.needs_f32_accum_for_bf16) {
        auto tmp_d = memory_desc_wrapper(jpp.tmp_md);
        scratchpad.book<char>(key_pool_src_f32_accum, tmp_d.size());
    }
}

struct trans_wrapper_t {
    trans_wrapper_t(data_type_t inp_dt, dim_t inp_str, data_type_t out_dt,
            dim_t out_str, dim_t ysize, dim_t xsize)
        : inp_dt_size_(types::data_type_size(inp_dt))
        , out_dt_size_(types::data_type_size(out_dt))
        , inp_str_(inp_str)
        , out_str_(out_str)
        , nb_x_(xsize / 8)
        , nb_y_(ysize / 8)
        , x_tail_(xsize % 8)
        , y_tail_(ysize % 8) {
        using namespace cpu::x64::tr;

        auto create_ker = [=](dim_t ys, dim_t y_inp_str, dim_t y_out_str,
                                  dim_t xs, dim_t x_inp_str, dim_t x_out_str) {
            tr::prb_t prb;
            kernel_t::desc_t desc;

            prb.ndims = 2;
            prb.ioff = 0;
            prb.ooff = 0;
            prb.src_scale_type = scale_type_t::NONE;
            prb.dst_scale_type = scale_type_t::NONE;
            prb.beta = 0;
            prb.nodes[0].ss = prb.nodes[1].ss = 1;

            prb.itype = inp_dt;
            prb.otype = out_dt;

            prb.nodes[0].n = ys;
            prb.nodes[0].is = y_inp_str;
            prb.nodes[0].os = y_out_str;

            prb.nodes[1].n = xs;
            prb.nodes[1].is = x_inp_str;
            prb.nodes[1].os = x_out_str;

            prb.full_ndims = prb.ndims;

            kernel_t::desc_init(desc, prb, 2);
            return kernel_t::create(desc);
        };

        if (nb_x_ * nb_y_ > 0)
            ker_.reset(create_ker(8, inp_str_, 1, 8, 1, out_str_));

        if (x_tail_)
            ker_x_tail_.reset(create_ker(8, inp_str_, 1, x_tail_, 1, out_str_));

        if (y_tail_)
            ker_y_tail_.reset(
                    create_ker(y_tail_, inp_str_, 1, xsize, 1, out_str_));
    }

    status_t create_kernel() {
        if (ker_) CHECK(ker_->create_kernel());
        if (ker_x_tail_) CHECK(ker_x_tail_->create_kernel());
        if (ker_y_tail_) CHECK(ker_y_tail_->create_kernel());
        return status::success;
    }

    void exec(const void *inp, void *out) {
        dim_t x_blocked = nb_x_ * 8;
        dim_t y_blocked = nb_y_ * 8;

        auto call_ker = [&](tr::kernel_t &ker, dim_t inp_y, dim_t inp_x,
                                dim_t out_y, dim_t out_x) {
            tr::call_param_t cp;
            cp.src_scales = nullptr;
            cp.dst_scales = nullptr;

            dim_t inp_off = (inp_y * inp_str_ + inp_x) * inp_dt_size_;
            dim_t out_off = (out_y * out_str_ + out_x) * out_dt_size_;
            cp.in = (uint8_t *)inp + inp_off;
            cp.out = (uint8_t *)out + out_off;
            (ker)(&cp);
        };

        for (dim_t by = 0; by < nb_y_; by++) {
            for (dim_t bx = 0; bx < nb_x_; bx++)
                call_ker(*ker_, 8 * by, 8 * bx, 8 * bx, 8 * by);

            if (x_tail_)
                call_ker(*ker_x_tail_, 8 * by, x_blocked, x_blocked, 8 * by);
        }
        if (y_tail_) call_ker(*ker_y_tail_, y_blocked, 0, 0, y_blocked);
    }

    ~trans_wrapper_t() = default;

private:
    std::unique_ptr<tr::kernel_t> ker_;
    std::unique_ptr<tr::kernel_t> ker_x_tail_;
    std::unique_ptr<tr::kernel_t> ker_y_tail_;

    const size_t inp_dt_size_;
    const size_t out_dt_size_;

    const dim_t inp_str_;
    const dim_t out_str_;
    const dim_t nb_x_;
    const dim_t nb_y_;
    const dim_t x_tail_;
    const dim_t y_tail_;
};

struct trans_context_t {
    std::unique_ptr<trans_wrapper_t> src_trans_ = nullptr;
    std::unique_ptr<trans_wrapper_t> src_tail_trans_ = nullptr;
    std::unique_ptr<trans_wrapper_t> ind_trans_ = nullptr;
    std::unique_ptr<trans_wrapper_t> ind_tail_trans_ = nullptr;
    std::unique_ptr<trans_wrapper_t> dst_trans_ = nullptr;
    std::unique_ptr<trans_wrapper_t> dst_tail_trans_ = nullptr;

    // NOLINTNEXTLINE(readability-make-member-function-const)
    status_t create_kernel() {
        if (src_trans_) CHECK(src_trans_->create_kernel());
        if (src_tail_trans_) CHECK(src_tail_trans_->create_kernel());
        if (ind_trans_) CHECK(ind_trans_->create_kernel());
        if (ind_tail_trans_) CHECK(ind_tail_trans_->create_kernel());
        if (dst_trans_) CHECK(dst_trans_->create_kernel());
        if (dst_tail_trans_) CHECK(dst_tail_trans_->create_kernel());
        return status::success;
    }
};

static void trans_exec(trans_wrapper_t *trans, trans_wrapper_t *trans_tail,
        dim_t cs, const void *inp, void *out, dim_t c_block) {

    if (cs == c_block)
        trans->exec(inp, out);
    else
        trans_tail->exec(inp, out);
};

template <typename src_data_t, typename dst_data_t>
struct transpose_ncsp_to_block_fmt_t {
    transpose_ncsp_to_block_fmt_t(trans_wrapper_t *transposer,
            trans_wrapper_t *transposer_tail, const src_data_t *src_nscp_base,
            const memory_desc_wrapper &src_nscp_desc,
            dst_data_t *__restrict dst_blocked_base, dim_t block_size,
            const jit_pool_conf_t &jpp, std::size_t offset_multiplier = 1u)
        : transposer_(transposer)
        , transposer_tail_(transposer_tail)
        , c_without_padding_(jpp.c_without_padding)
        , c_block_(jpp.c_block)
        , src_nscp_base_(src_nscp_base)
        , src_nscp_desc_(src_nscp_desc)
        , dst_blocked_base_(dst_blocked_base)
        , block_size_(block_size)
        , offset_multiplier_(offset_multiplier) {}

    void operator()(std::size_t ithr, int n, int b_c) const {
        const dim_t cs
                = nstl::min(c_without_padding_ - b_c * c_block_, c_block_);
        const src_data_t *src_nscp = src_nscp_base_
                + src_nscp_desc_.blk_off(n, b_c * c_block_, 0)
                        * offset_multiplier_;
        dst_data_t *dst_blocked
                = dst_blocked_base_ + ithr * block_size_ * offset_multiplier_;
        trans_exec(transposer_, transposer_tail_, cs, src_nscp, dst_blocked,
                c_block_);
    }

private:
    trans_wrapper_t *transposer_;
    trans_wrapper_t *transposer_tail_;
    const int c_without_padding_;
    const int c_block_;
    const src_data_t *src_nscp_base_;
    const memory_desc_wrapper &src_nscp_desc_;
    dst_data_t *__restrict dst_blocked_base_;
    const dim_t block_size_;
    std::size_t offset_multiplier_;
};

template <typename src_data_t, typename dst_data_t>
struct transpose_block_fmt_to_ncsp_t {

    transpose_block_fmt_to_ncsp_t(trans_wrapper_t *transposer,
            trans_wrapper_t *transposer_tail,
            const src_data_t *__restrict src_blocked_base, dim_t block_size,
            dst_data_t *dst_ncsp_base, const memory_desc_wrapper &dst_nscp_desc,
            const jit_pool_conf_t &jpp, std::size_t offset_multiplier = 1u)
        : transposer_(transposer)
        , transposer_tail_(transposer_tail)
        , c_without_padding_(jpp.c_without_padding)
        , c_block_(jpp.c_block)
        , src_blocked_base_(src_blocked_base)
        , block_size_(block_size)
        , dst_ncsp_base_(dst_ncsp_base)
        , dst_nscp_desc_(dst_nscp_desc)
        , offset_multiplier_(offset_multiplier) {}

    void operator()(std::size_t ithr, int n, int b_c) const {
        const dim_t cs
                = nstl::min(c_without_padding_ - b_c * c_block_, c_block_);
        const src_data_t *src_blocked
                = src_blocked_base_ + ithr * block_size_ * offset_multiplier_;
        dst_data_t *dst_ncsp = dst_ncsp_base_
                + dst_nscp_desc_.blk_off(n, b_c * c_block_, 0)
                        * offset_multiplier_;
        trans_exec(transposer_, transposer_tail_, cs, src_blocked, dst_ncsp,
                c_block_);
    }

private:
    trans_wrapper_t *transposer_;
    trans_wrapper_t *transposer_tail_;
    const int c_without_padding_;
    const int c_block_;
    const src_data_t *__restrict src_blocked_base_;
    const dim_t block_size_;
    dst_data_t *dst_ncsp_base_;
    const memory_desc_wrapper &dst_nscp_desc_;
    std::size_t offset_multiplier_;
};

template <typename wsp_data_t, impl::data_type_t d_type>
class transpose_facade_base_t {
public:
    transpose_facade_base_t(const jit_pool_conf_t &jpp,
            const memory_desc_wrapper &src_d, const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &indices_d, const char *indices,
            const data_type_t wsp_dt, const exec_ctx_t &ctx)
        : src_sp_(static_cast<dim_t>(jpp.id) * jpp.ih * jpp.iw)
        , dst_sp_(static_cast<dim_t>(jpp.od) * jpp.oh * jpp.ow)
        , src_slice_(src_sp_ * jpp.c_block)
        , dst_slice_(dst_sp_ * jpp.c_block)
        , transpose_src_(jpp.tag_kind == jit_memory_tag_kind_t::ncsp)
        , transpose_dst_(jpp.tag_kind == jit_memory_tag_kind_t::ncsp)
        , src_d_(src_d)
        , dst_d_(dst_d)
        , indices_d_(indices_d)
        , ind_dt_size_(
                  indices ? types::data_type_size(indices_d_.data_type()) : 0)
        , cvt_slice_src_wsp_(nullptr)
        , cvt_slice_dst_wsp_(nullptr)
        , cvt_slice_ind_wsp_(nullptr)
        , execute_transpose_input_(nullptr)
        , execute_transpose_output_(nullptr) {

        auto scratchpad = ctx.get_scratchpad_grantor();

        if (transpose_src_)
            cvt_slice_src_wsp_ = scratchpad.template get<wsp_data_t>(
                    memory_tracking::names::key_pool_src_plain2blocked_cvt);

        if (transpose_dst_) {
            cvt_slice_dst_wsp_ = scratchpad.template get<wsp_data_t>(
                    memory_tracking::names::key_pool_dst_plain2blocked_cvt);
            cvt_slice_ind_wsp_ = scratchpad.template get<char>(
                    memory_tracking::names::key_pool_ind_plain2blocked_cvt);
        }
    }

    inline bool should_transpose_src() const noexcept { return transpose_src_; }
    inline bool should_transpose_dst() const noexcept { return transpose_dst_; }

    const void *get_src_addr(
            std::size_t ithr, int ih, const jit_pool_conf_t &jpp) const {
        const wsp_data_t *const wsp = cvt_slice_src_wsp_ + ithr * src_slice_;
        return static_cast<const void *>(&wsp[ih * jpp.iw * jpp.c_block]);
    }

    const void *get_dst_addr(
            std::size_t ithr, int oh, const jit_pool_conf_t &jpp) const {
        const wsp_data_t *const wsp = cvt_slice_dst_wsp_ + ithr * dst_slice_;
        return static_cast<const void *>(&wsp[oh * jpp.ow * jpp.c_block]);
    }

    const void *get_indices_addr(
            std::size_t ithr, int oh, const jit_pool_conf_t &jpp) const {
        const char *const wsp
                = cvt_slice_ind_wsp_ + ithr * dst_slice_ * ind_dt_size_;
        return static_cast<const void *>(
                &wsp[oh * jpp.ow * jpp.c_block * ind_dt_size_]);
    }

    const void *get_src_addr_3d(std::size_t ithr, int id, int ih,
            const jit_pool_conf_t &jpp) const {
        const wsp_data_t *const wsp = cvt_slice_src_wsp_ + ithr * src_slice_;
        return static_cast<const void *>(&wsp[ih * jpp.iw * jpp.c_block
                + id * jpp.ih * jpp.iw * jpp.c_block]);
    }

    const void *get_dst_addr_3d(std::size_t ithr, int od, int oh,
            const jit_pool_conf_t &jpp) const {
        const wsp_data_t *const wsp = cvt_slice_dst_wsp_ + ithr * dst_slice_;
        return static_cast<const void *>(&wsp[oh * jpp.ow * jpp.c_block
                + od * jpp.oh * jpp.ow * jpp.c_block]);
    }

    const void *get_indices_addr_3d(std::size_t ithr, int od, int oh,
            const jit_pool_conf_t &jpp) const {
        const char *const wsp
                = cvt_slice_ind_wsp_ + ithr * dst_slice_ * ind_dt_size_;
        return static_cast<const void *>(
                &wsp[oh * jpp.ow * jpp.c_block * ind_dt_size_
                        + od * jpp.oh * jpp.ow * jpp.c_block * ind_dt_size_]);
    }

    void execute_transpose_input(std::size_t ithr, int n, int b_c) const {
        execute_transpose_input_(ithr, n, b_c);
    }

    void execute_transpose_output(std::size_t ithr, int n, int b_c) const {
        execute_transpose_output_(ithr, n, b_c);
    }

protected:
    const dim_t src_sp_;
    const dim_t dst_sp_;
    const dim_t src_slice_;
    const dim_t dst_slice_;

    const bool transpose_src_;
    const bool transpose_dst_;

    const memory_desc_wrapper &src_d_;
    const memory_desc_wrapper &dst_d_;
    const memory_desc_wrapper &indices_d_;
    const size_t ind_dt_size_;

    wsp_data_t *__restrict cvt_slice_src_wsp_;
    wsp_data_t *__restrict cvt_slice_dst_wsp_;
    char *__restrict cvt_slice_ind_wsp_;

    std::function<void(std::size_t, int, int)> execute_transpose_input_;
    std::function<void(std::size_t, int, int)> execute_transpose_output_;
};

template <typename data_t, typename wsp_data_t, impl::data_type_t d_type>
class fwd_pooling_transpose_facade_t
    : public transpose_facade_base_t<wsp_data_t, d_type> {
public:
    fwd_pooling_transpose_facade_t(const jit_pool_conf_t &jpp,
            trans_context_t *trans_ctx, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &indices_d, const data_type_t wsp_dt,
            const data_t *src, data_t *dst, char *indices,
            const exec_ctx_t &ctx)
        : transpose_facade_base_t<wsp_data_t, d_type>(
                jpp, src_d, dst_d, indices_d, indices, wsp_dt, ctx) {

        if (this->should_transpose_src()) {
            this->execute_transpose_input_
                    = transpose_ncsp_to_block_fmt_t<data_t, wsp_data_t>(
                            trans_ctx->src_trans_.get(),
                            trans_ctx->src_tail_trans_.get(), src, this->src_d_,
                            this->cvt_slice_src_wsp_, this->src_slice_, jpp);
        }

        if (this->should_transpose_dst()) {
            using namespace std::placeholders;
            this->execute_transpose_output_ = std::bind(
                    [=](const transpose_block_fmt_to_ncsp_t<wsp_data_t, data_t>
                                    &trans_dst,
                            transpose_block_fmt_to_ncsp_t<char, char>
                                    &trans_indices,
                            std::size_t ithr, int n, int b_c) {
                        trans_dst(ithr, n, b_c);
                        if (indices) trans_indices(ithr, n, b_c);
                    },
                    transpose_block_fmt_to_ncsp_t<wsp_data_t, data_t>(
                            trans_ctx->dst_trans_.get(),
                            trans_ctx->dst_tail_trans_.get(),
                            this->cvt_slice_dst_wsp_, this->dst_slice_, dst,
                            this->dst_d_, jpp, 1u),
                    transpose_block_fmt_to_ncsp_t<char, char>(
                            trans_ctx->ind_trans_.get(),
                            trans_ctx->ind_tail_trans_.get(),
                            this->cvt_slice_ind_wsp_, this->dst_slice_, indices,
                            this->indices_d_, jpp, this->ind_dt_size_),
                    _1, _2, _3);
        }
    }
};

template <typename data_t, typename wsp_data_t, impl::data_type_t d_type>
class bwd_pooling_transpose_facade_t
    : public transpose_facade_base_t<wsp_data_t, d_type> {
public:
    bwd_pooling_transpose_facade_t(const jit_pool_conf_t &jpp,
            trans_context_t *trans_ctx, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d,
            const memory_desc_wrapper &indices_d, const data_type_t wsp_dt,
            data_t *src, const data_t *dst, const char *indices,
            const exec_ctx_t &ctx)
        : transpose_facade_base_t<wsp_data_t, d_type>(
                jpp, src_d, dst_d, indices_d, indices, wsp_dt, ctx)
        , c_tail_(jpp.c_without_padding % jpp.c_block) {

        if (this->should_transpose_src())
            this->execute_transpose_output_
                    = transpose_block_fmt_to_ncsp_t<wsp_data_t, data_t>(
                            trans_ctx->src_trans_.get(),
                            trans_ctx->src_tail_trans_.get(),
                            this->cvt_slice_src_wsp_, this->src_slice_, src,
                            this->src_d_, jpp, 1u);

        if (this->should_transpose_dst()) {
            using namespace std::placeholders;

            this->execute_transpose_input_ = std::bind(
                    [=](const transpose_ncsp_to_block_fmt_t<data_t, wsp_data_t>
                                    &trans_dst,
                            transpose_ncsp_to_block_fmt_t<char, char>
                                    &trans_indices,
                            std::size_t ithr, int n, int b_c) {
                        trans_dst(ithr, n, b_c);
                        if (indices) trans_indices(ithr, n, b_c);
                    },
                    transpose_ncsp_to_block_fmt_t<data_t, wsp_data_t>(
                            trans_ctx->dst_trans_.get(),
                            trans_ctx->dst_tail_trans_.get(), dst, this->dst_d_,
                            this->cvt_slice_dst_wsp_, this->dst_slice_, jpp),
                    transpose_ncsp_to_block_fmt_t<char, char>(
                            trans_ctx->ind_trans_.get(),
                            trans_ctx->ind_tail_trans_.get(), indices,
                            this->indices_d_, this->cvt_slice_ind_wsp_,
                            this->dst_slice_, jpp, this->ind_dt_size_),
                    _1, _2, _3);
        }
    }

    inline bool should_fill_input_c_tail_with_zeros() const noexcept {
        return this->should_transpose_dst() && c_tail_ != 0;
    }

    void fill_input_c_tail_with_zeros(
            std::size_t ithr, const jit_pool_conf_t &jpp) const {

        wsp_data_t *__restrict wsp_ptr
                = this->cvt_slice_dst_wsp_ + ithr * this->dst_slice_;
        for_(dim_t s = 0; s < this->dst_sp_; s++)
        for (dim_t c = c_tail_; c < jpp.c_block; c++)
            wsp_ptr[s * jpp.c_block + c] = 0.f;

        char *__restrict ind_ptr = this->cvt_slice_ind_wsp_
                + ithr * this->dst_slice_ * this->ind_dt_size_;
        for_(dim_t s = 0; s < this->dst_sp_; s++)
        for_(dim_t c = c_tail_; c < jpp.c_block; c++)
        for (size_t i = 0; i < this->ind_dt_size_; i++)
            ind_ptr[(s * jpp.c_block + c) * this->ind_dt_size_ + i] = 0;
    }

private:
    const dim_t c_tail_;
};

struct bwd_f32_accum_for_bf16_t {
    using value_type = typename prec_traits<data_type::f32>::type;

    bwd_f32_accum_for_bf16_t(const jit_pool_conf_t &jpp, const exec_ctx_t &ctx);

    value_type *get_addr_2d(int ithr, dim_t ih) const {
        return blk_data(ithr, 0, ih, 0);
    }

    value_type *get_addr_3d(int ithr, dim_t id, dim_t ih) const {
        return blk_data(ithr, 0, id, ih, 0);
    }

    void zero_data(int ithr);

    void cvt_to_bf16_slice_2d(int ithr, bfloat16_t *dst,
            memory_desc_wrapper const &dst_d, dim_t n, dim_t b_c,
            dim_t ur_bc) const;

    void cvt_to_bf16_slice_3d(int ithr, bfloat16_t *dst,
            memory_desc_wrapper const &dst_d, dim_t n, dim_t b_c,
            dim_t ur_bc) const;

private:
    template <typename... Args>
    value_type *blk_data(Args... args) const {
        assert(wsp_);
        return wsp_ + accum_d_.blk_off(std::forward<Args>(args)...);
    }

    const jit_pool_conf_t &jpp_;
    value_type *wsp_ {nullptr};
    memory_desc_wrapper accum_d_ {nullptr};
};

bwd_f32_accum_for_bf16_t::bwd_f32_accum_for_bf16_t(
        const jit_pool_conf_t &jpp, const exec_ctx_t &ctx)
    : jpp_ {jpp} {
    if (jpp_.needs_f32_accum_for_bf16) {
        accum_d_ = memory_desc_wrapper(jpp_.tmp_md);
        auto &scratchpad = ctx.get_scratchpad_grantor();
        wsp_ = scratchpad.template get<value_type>(
                memory_tracking::names::key_pool_src_f32_accum);
        assert(wsp_);
    }
}

void bwd_f32_accum_for_bf16_t::zero_data(int ithr) {
    auto *data = blk_data(ithr);
    memset(data, 0,
            jpp_.tmp_md.format_desc.blocking.strides[0] * sizeof(value_type));
}

void bwd_f32_accum_for_bf16_t::cvt_to_bf16_slice_2d(int ithr, bfloat16_t *dst,
        memory_desc_wrapper const &dst_d, dim_t n, dim_t b_c,
        dim_t ur_bc) const {

    assert(wsp_ && (jpp_.ndims == 3 || jpp_.ndims == 4)
            && (jpp_.tag_kind == jit_memory_tag_kind_t::nspc
                    || jpp_.tag_kind == jit_memory_tag_kind_t::blocked));

    if (jpp_.tag_kind == jit_memory_tag_kind_t::nspc) {
        if (jpp_.tmp_md.dims[1] == jpp_.c && b_c == 0
                && jpp_.c == ur_bc * jpp_.c_block) {
            // all channels
            const size_t nelems = jpp_.ih * jpp_.iw * jpp_.c;
            const auto *cur_src = blk_data(ithr);
            auto *cur_dst = dst + dst_d.blk_off(n);
            cvt_float_to_bfloat16(cur_dst, cur_src, nelems);
        } else {
            const auto c_b = jpp_.c_block * b_c;
            const auto c_e = nstl::min(
                    static_cast<dim_t>(jpp_.c), jpp_.c_block * (b_c + ur_bc));

            if (c_b >= c_e) return;

            const size_t nelems = c_e - c_b;
            if (jpp_.ndims == 4) {
                for (dim_t h = 0; h < jpp_.ih; ++h) {
                    for (dim_t w = 0; w < jpp_.iw; ++w) {
                        const auto *cur_src = blk_data(ithr, 0, h, w);
                        auto *cur_dst = dst + dst_d.blk_off(n, c_b, h, w);
                        cvt_float_to_bfloat16(cur_dst, cur_src, nelems);
                    }
                }
            } else {
                for (dim_t w = 0; w < jpp_.iw; ++w) {
                    const auto *cur_src = blk_data(ithr, 0, w);
                    auto *cur_dst = dst + dst_d.blk_off(n, c_b, w);
                    cvt_float_to_bfloat16(cur_dst, cur_src, nelems);
                }
            }
        }
    } else if (jpp_.tag_kind == jit_memory_tag_kind_t::blocked) {
        assert(ur_bc == 1);

        const size_t nelems = jpp_.ih * jpp_.iw * jpp_.c_block;
        const auto *src_b = blk_data(ithr);
        auto *dst_b = dst + dst_d.blk_off(n, b_c);
        cvt_float_to_bfloat16(dst_b, src_b, nelems);
    }
}

void bwd_f32_accum_for_bf16_t::cvt_to_bf16_slice_3d(int ithr, bfloat16_t *dst,
        memory_desc_wrapper const &dst_d, dim_t n, dim_t b_c,
        dim_t ur_bc) const {

    assert(wsp_ && jpp_.ndims == 5
            && (jpp_.tag_kind == jit_memory_tag_kind_t::nspc
                    || jpp_.tag_kind == jit_memory_tag_kind_t::blocked));

    if (jpp_.tag_kind == jit_memory_tag_kind_t::blocked) {
        assert(ur_bc == 1);
        const size_t nelems = jpp_.id * jpp_.ih * jpp_.iw * jpp_.c_block;
        const auto *src_b = blk_data(ithr);
        auto *dst_b = dst + dst_d.blk_off(n, b_c);
        cvt_float_to_bfloat16(dst_b, src_b, nelems);
    } else if (jpp_.tag_kind == jit_memory_tag_kind_t::nspc) {
        if (jpp_.tmp_md.dims[1] == jpp_.c && b_c == 0
                && jpp_.c == ur_bc * jpp_.c_block) {
            // all channels
            const size_t nelems = jpp_.id * jpp_.ih * jpp_.iw * jpp_.c;
            cvt_float_to_bfloat16(
                    dst + dst_d.blk_off(n), blk_data(ithr), nelems);
        } else {
            const auto c_b = jpp_.c_block * b_c;
            const auto c_e = nstl::min(
                    static_cast<dim_t>(jpp_.c), jpp_.c_block * (b_c + ur_bc));

            if (c_b >= c_e) return;

            const size_t nelems = c_e - c_b;
            for (dim_t id = 0; id < jpp_.id; ++id) {
                for (dim_t h = 0; h < jpp_.ih; ++h) {
                    for (dim_t w = 0; w < jpp_.iw; ++w) {
                        const auto *cur_src = blk_data(ithr, 0, id, h, w);
                        auto *cur_dst = dst + dst_d.blk_off(n, c_b, id, h, w);
                        cvt_float_to_bfloat16(cur_dst, cur_src, nelems);
                    }
                }
            }
        }
    }
}

} // namespace jit_uni_pooling_utils

template <cpu_isa_t isa, impl::data_type_t d_type>
status_t jit_uni_pooling_fwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace utils;

    VDISPATCH_POOLING(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_POOLING(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_POOLING(
            everyone_is(d_type, src_md()->data_type, dst_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_POOLING(attr()->has_default_values(
                              primitive_attr_t::skip_mask_t::post_ops, d_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
            "does not support dilations");
    VDISPATCH_POOLING(
            set_default_params() == status::success, VERBOSE_UNSUPPORTED_TAG);

    const bool is_training = desc_.prop_kind == prop_kind::forward_training;
    if (desc()->alg_kind == alg_kind::pooling_max && is_training)
        init_default_ws();

    CHECK(jit_uni_pooling_utils::init_conf(isa, jpp_, attr_, this));

    auto scratchpad = scratchpad_registry().registrar();
    jit_uni_pooling_utils::init_scratchpad(jpp_, scratchpad);

    return status::success;
}

template <cpu_isa_t isa, impl::data_type_t d_type>
jit_uni_pooling_fwd_t<isa, d_type>::jit_uni_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd), kernel_(nullptr), trans_ctx_(nullptr) {}

template <cpu_isa_t isa, impl::data_type_t d_type>
status_t jit_uni_pooling_fwd_t<isa, d_type>::init(engine_t *engine) {

    CHECK(safe_ptr_assign(kernel_,
            new jit_uni_pool_kernel<isa>(
                    pd()->jpp_, pd()->invariant_dst_md())));

    if (pd()->jpp_.tag_kind == jit_memory_tag_kind_t::ncsp)
        CHECK(init_ncsp_trans_ctx());
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_pooling_fwd_t<isa, d_type>::init_ncsp_trans_ctx() {
    using namespace dnnl::impl;
    using namespace jit_uni_pooling_utils;

    const auto &jpp = pd()->jpp_;
    trans_ctx_ = utils::make_unique<trans_context_t>();
    const dim_t src_sp = static_cast<dim_t>(jpp.id) * jpp.ih * jpp.iw;
    const dim_t dst_sp = static_cast<dim_t>(jpp.od) * jpp.oh * jpp.ow;
    const auto res = std::div(jpp.c_without_padding, jpp.c_block);
    const dim_t &nb_c = res.quot;
    const dim_t &c_tail = res.rem;
    const memory_desc_wrapper indices_d = pd()->workspace_md();
    const bool have_indices = indices_d.data_type() != data_type::undef;
    static constexpr auto wsp_dt = wsp_dt_;

    if (nb_c) {
        trans_ctx_->src_trans_ = utils::make_unique<trans_wrapper_t>(
                d_type, src_sp, wsp_dt, jpp.c_block, jpp.c_block, src_sp);
        trans_ctx_->dst_trans_ = utils::make_unique<trans_wrapper_t>(
                wsp_dt, jpp.c_block, d_type, dst_sp, dst_sp, jpp.c_block);
        if (have_indices)
            trans_ctx_->ind_trans_ = utils::make_unique<trans_wrapper_t>(
                    indices_d.data_type(), jpp.c_block, indices_d.data_type(),
                    dst_sp, dst_sp, jpp.c_block);
    }

    if (c_tail) {
        trans_ctx_->src_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                d_type, src_sp, wsp_dt, jpp.c_block, c_tail, src_sp);
        trans_ctx_->dst_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                wsp_dt, jpp.c_block, d_type, dst_sp, dst_sp, c_tail);
        if (have_indices)
            trans_ctx_->ind_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                    indices_d.data_type(), jpp.c_block, indices_d.data_type(),
                    dst_sp, dst_sp, c_tail);
    }

    return trans_ctx_->create_kernel();
}

template <cpu_isa_t isa, impl::data_type_t d_type>
jit_uni_pooling_fwd_t<isa, d_type>::~jit_uni_pooling_fwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_fwd_t<isa, d_type>::execute_forward(const data_t *src,
        data_t *dst, char *indices, const exec_ctx_t &ctx) const {

    const memory_desc_wrapper src_d = pd()->src_md();
    const memory_desc_wrapper dst_d = pd()->dst_md();
    const memory_desc_wrapper indices_d = pd()->workspace_md();
    const auto ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;
    const auto &jpp = pd()->jpp_;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jpp.post_ops, ctx);

    using wsp_data_t = typename prec_traits<wsp_dt_>::type;
    using namespace jit_uni_pooling_utils;

    const auto transpose_facade
            = fwd_pooling_transpose_facade_t<data_t, wsp_data_t, d_type>(jpp,
                    trans_ctx_.get(), src_d, dst_d, indices_d, wsp_dt_, src,
                    dst, indices, ctx);

    const auto trans_src = transpose_facade.should_transpose_src();
    const auto trans_dst = transpose_facade.should_transpose_dst();

    const auto ker = [&](std::size_t ithr, int n, int b_c, int oh, int ur_bc) {
        assert(ur_bc == jpp.ur_bc || ur_bc == jpp.ur_bc_tail);
        auto arg = jit_pool_call_s();

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad - ij);
        const int i_b_overflow
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        assert(IMPLICATION(pd()->ndims() == 3, utils::everyone_is(0, ih, oh)));
        const int c_off
                = ((jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c_block
                                                                 : 1)
                * b_c;

        if (trans_src)
            arg.src = transpose_facade.get_src_addr(ithr, ih, jpp);
        else
            arg.src = static_cast<const void *>(
                    &src[src_d.blk_off(n, c_off, ih)]);

        arg.dst_orig = dst;
        if (trans_dst) {
            arg.dst = transpose_facade.get_dst_addr(ithr, oh, jpp);
            if (!types::is_zero_md(&jpp.tmp_md)) {
                const memory_desc_wrapper tmp_d
                        = memory_desc_wrapper(jpp.tmp_md);
                // offset needs to be f32
                const int dt_scale
                        = sizeof(float) / types::data_type_size(d_type);
                const auto blk_off = tmp_d.blk_off(n, c_off, oh) * dt_scale;
                arg.dst_po_helper = static_cast<const void *>(&dst[blk_off]);
            }
        } else {
            arg.dst = static_cast<const void *>(
                    &dst[dst_d.blk_off(n, c_off, oh)]);
        }

        if (indices) {
            if (trans_dst)
                arg.indices = transpose_facade.get_indices_addr(ithr, oh, jpp);
            else {
                const size_t ind_off = indices_d.blk_off(n, c_off, oh);
                arg.indices = static_cast<const void *>(
                        &indices[ind_off * ind_dt_size]);
            }
        }
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift = i_t_overflow * jpp.kw;
        arg.ker_area_h = static_cast<float>(jpp.kh
                - nstl::max(0, oh * jpp.stride_h - jpp.t_pad + jpp.kh - jpp.ih)
                - nstl::max(0, jpp.t_pad - oh * jpp.stride_h));
        arg.ur_bc = ur_bc;
        arg.b_c = b_c;
        arg.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
        (*kernel_)(&arg);
    };

    const int nthr = jpp.nthr;

    if (jpp.tag_kind == jit_memory_tag_kind_t::nspc) {
        const auto nb2_c = utils::div_up(jpp.nb_c, jpp.ur_bc);
        parallel_nd(jpp.mb, jpp.oh, nb2_c, [&](dim_t n, dim_t oh, dim_t b2_c) {
            const auto b_c = b2_c * jpp.ur_bc;
            const auto ur_bc = nstl::min(dim_t(jpp.ur_bc), jpp.nb_c - b_c);
            ker(0, n, b_c, oh, ur_bc);
        });
    } else {
        if (trans_src || trans_dst) {
            // ncsp format
            parallel_nd_ext(nthr, jpp.mb, jpp.nb_c,
                    [&](dim_t ithr, dim_t nthr, dim_t n, dim_t b_c) {
                        if (trans_src)
                            transpose_facade.execute_transpose_input(
                                    ithr, n, b_c);
                        for (dim_t oh = 0; oh < jpp.oh; ++oh)
                            ker(ithr, n, b_c, oh, 1);
                        if (trans_dst)
                            transpose_facade.execute_transpose_output(
                                    ithr, n, b_c);
                    });
        } else {
            // nChw16c, nChw8c format
            parallel(nthr, [&](dim_t ithr, dim_t nthr) {
                dim_t work_amount
                        = static_cast<dim_t>(jpp.mb) * jpp.nb_c * jpp.oh;
                if (ithr >= work_amount) return;

                dim_t start {0}, end {0};
                dim_t n {0}, b_c {0}, oh {0};

                balance211(work_amount, nthr, ithr, start, end);
                utils::nd_iterator_init(
                        start, n, jpp.mb, b_c, jpp.nb_c, oh, jpp.oh);

                for (dim_t iwork = start; iwork < end; ++iwork) {
                    ker(ithr, n, b_c, oh, 1);
                    utils::nd_iterator_step(
                            n, jpp.mb, b_c, jpp.nb_c, oh, jpp.oh);
                }
            });
        }
    }
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_fwd_t<isa, d_type>::execute_forward_3d(const data_t *src,
        data_t *dst, char *indices, const exec_ctx_t &ctx) const {

    const auto &jpp = pd()->jpp_;
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());
    const size_t ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jpp.post_ops, ctx);

    using wsp_data_t = typename prec_traits<wsp_dt_>::type;
    using namespace jit_uni_pooling_utils;
    static constexpr int first_ithr = 0;

    const auto transpose_facade
            = fwd_pooling_transpose_facade_t<data_t, wsp_data_t, d_type>(jpp,
                    trans_ctx_.get(), src_d, dst_d, indices_d, wsp_dt_, src,
                    dst, indices, ctx);

    const auto trans_src = transpose_facade.should_transpose_src();
    const auto trans_dst = transpose_facade.should_transpose_dst();

    auto ker = [&](int n, int b_c, int od, int oh, int id, int d_t_overflow,
                       int d_b_overflow, int ur_bc, int ithr) {
        assert(ur_bc == jpp.ur_bc || ur_bc == jpp.ur_bc_tail);
        auto arg = jit_pool_call_s();

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad - ij);
        const int i_b_overflow
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        const int c_off
                = ((jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c_block
                                                                 : 1)
                * b_c;

        if (trans_src)
            arg.src = transpose_facade.get_src_addr_3d(ithr, id, ih, jpp);
        else
            arg.src = &src[src_d.blk_off(n, c_off, id, ih)];

        arg.dst_orig = dst;
        if (trans_dst) {
            arg.dst = transpose_facade.get_dst_addr_3d(ithr, od, oh, jpp);
            if (!types::is_zero_md(&jpp.tmp_md)) {
                const memory_desc_wrapper tmp_d
                        = memory_desc_wrapper(jpp.tmp_md);
                // offset needs to be f32
                const int dt_scale
                        = sizeof(float) / types::data_type_size(d_type);
                const auto blk_off = tmp_d.blk_off(n, c_off, od, oh) * dt_scale;
                arg.dst_po_helper = static_cast<const void *>(&dst[blk_off]);
            }
        } else {
            arg.dst = &dst[dst_d.blk_off(n, c_off, od, oh)];
        }

        if (indices) {
            if (trans_dst) {
                arg.indices = transpose_facade.get_indices_addr_3d(
                        ithr, od, oh, jpp);
            } else {
                const size_t ind_off = indices_d.blk_off(n, c_off, od, oh);
                arg.indices = &indices[ind_off * ind_dt_size];
            }
        }

        arg.kd_padding = jpp.kd - d_t_overflow - d_b_overflow;
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift
                = i_t_overflow * jpp.kw + d_t_overflow * jpp.kw * jpp.kh;
        arg.kd_padding_shift = (i_t_overflow + i_b_overflow) * jpp.kw;
        arg.ker_area_h = (float)(jpp.kh
                                 - nstl::max(0,
                                         oh * jpp.stride_h - jpp.t_pad + jpp.kh
                                                 - jpp.ih)
                                 - nstl::max(0, jpp.t_pad - oh * jpp.stride_h))
                * (jpp.kd
                        - nstl::max(0,
                                od * jpp.stride_d - jpp.f_pad + jpp.kd - jpp.id)
                        - nstl::max(0, jpp.f_pad - od * jpp.stride_d));

        arg.ur_bc = ur_bc;
        arg.b_c = b_c;
        arg.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
        (*kernel_)(&arg);
    };

    const int nthr = jpp.nthr;

    if (jpp.tag_kind == jit_memory_tag_kind_t::nspc) {
        const auto nb2_c = utils::div_up(jpp.nb_c, jpp.ur_bc);
        parallel_nd(jpp.mb, jpp.od, nb2_c, [&](dim_t n, dim_t od, dim_t b2_c) {
            const dim_t b_c = b2_c * jpp.ur_bc;
            const dim_t ur_bc = nstl::min(dim_t(jpp.ur_bc), jpp.nb_c - b_c);

            const dim_t ik = od * jpp.stride_d;
            const dim_t d_t_overflow = nstl::max(dim_t(0), jpp.f_pad - ik);
            const dim_t d_b_overflow
                    = nstl::max(dim_t(jpp.id), ik + jpp.kd - jpp.f_pad)
                    - jpp.id;
            const dim_t id = nstl::max(ik - jpp.f_pad, dim_t(0));
            for (dim_t oh = 0; oh < jpp.oh; ++oh) {
                ker(n, b_c, od, oh, id, d_t_overflow, d_b_overflow, ur_bc,
                        first_ithr);
            }
        });
    } else {
        if (trans_src || trans_dst) {
            parallel_nd_ext(nthr, jpp.mb, jpp.nb_c,
                    [&](dim_t ithr, dim_t nthr, dim_t n, dim_t b_c) {
                        if (trans_src)
                            transpose_facade.execute_transpose_input(
                                    ithr, n, b_c);

                        for (int od = 0; od < jpp.od; ++od) {
                            const int ik = od * jpp.stride_d;
                            const int d_t_overflow
                                    = nstl::max(0, jpp.f_pad - ik);
                            const int d_b_overflow
                                    = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad)
                                    - jpp.id;
                            const int id = nstl::max(ik - jpp.f_pad, 0);
                            for (int oh = 0; oh < jpp.oh; ++oh) {
                                ker(n, b_c, od, oh, id, d_t_overflow,
                                        d_b_overflow, 1, ithr);
                            }
                        }

                        if (trans_dst)
                            transpose_facade.execute_transpose_output(
                                    ithr, n, b_c);
                    });
        } else {
            parallel_nd(jpp.mb, jpp.nb_c, jpp.od,
                    [&](dim_t n, dim_t b_c, dim_t od) {
                        const int ik = od * jpp.stride_d;
                        const int d_t_overflow = nstl::max(0, jpp.f_pad - ik);
                        const int d_b_overflow
                                = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad)
                                - jpp.id;
                        const int id = nstl::max(ik - jpp.f_pad, 0);
                        for (int oh = 0; oh < jpp.oh; ++oh) {
                            ker(n, b_c, od, oh, id, d_t_overflow, d_b_overflow,
                                    1, first_ithr);
                        }
                    });
        }
    }
}

template <cpu_isa_t isa, impl::data_type_t d_type>
status_t jit_uni_pooling_bwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace utils;

    VDISPATCH_POOLING(
            set_default_params() == status::success, VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_POOLING(!is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_POOLING(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_POOLING(everyone_is(d_type, diff_src_md()->data_type,
                              diff_dst_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_POOLING(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
            "does not support dilations");

    if (desc()->alg_kind == alg_kind::pooling_max) {
        const auto ws_dt = hint_fwd_pd_->workspace_md()->data_type;
        init_default_ws(ws_dt);
        VDISPATCH_POOLING(compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);
    }

    CHECK(jit_uni_pooling_utils::init_conf(isa, jpp_, attr_, this));

    auto scratchpad = scratchpad_registry().registrar();
    jit_uni_pooling_utils::init_scratchpad(jpp_, scratchpad);

    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_pooling_bwd_t<isa, d_type>::jit_uni_pooling_bwd_t(const pd_t *apd)
    : primitive_t(apd), kernel_(nullptr), trans_ctx_(nullptr) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_pooling_bwd_t<isa, d_type>::~jit_uni_pooling_bwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_pooling_bwd_t<isa, d_type>::init_ncsp_trans_ctx() {
    using namespace dnnl::impl;
    using namespace jit_uni_pooling_utils;

    const auto &jpp = pd()->jpp_;
    trans_ctx_ = utils::make_unique<trans_context_t>();
    const dim_t diff_src_sp = static_cast<dim_t>(jpp.id) * jpp.ih * jpp.iw;
    const dim_t diff_dst_sp = static_cast<dim_t>(jpp.od) * jpp.oh * jpp.ow;
    const auto res = std::div(jpp.c_without_padding, jpp.c_block);
    const dim_t &nb_c = res.quot;
    const dim_t &c_tail = res.rem;
    const memory_desc_wrapper indices_d = pd()->workspace_md();
    const bool have_indices = indices_d.data_type() != data_type::undef;
    static constexpr auto wsp_dt = wsp_dt_;

    if (nb_c) {
        trans_ctx_->dst_trans_ = utils::make_unique<trans_wrapper_t>(d_type,
                diff_dst_sp, wsp_dt, jpp.c_block, jpp.c_block, diff_dst_sp);
        trans_ctx_->src_trans_ = utils::make_unique<trans_wrapper_t>(wsp_dt,
                jpp.c_block, d_type, diff_src_sp, diff_src_sp, jpp.c_block);
        if (have_indices)
            trans_ctx_->ind_trans_ = utils::make_unique<trans_wrapper_t>(
                    indices_d.data_type(), diff_dst_sp, indices_d.data_type(),
                    jpp.c_block, jpp.c_block, diff_dst_sp);
    }
    if (c_tail) {
        trans_ctx_->dst_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                d_type, diff_dst_sp, wsp_dt, jpp.c_block, c_tail, diff_dst_sp);
        trans_ctx_->src_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                wsp_dt, jpp.c_block, d_type, diff_src_sp, diff_src_sp, c_tail);
        if (have_indices)
            trans_ctx_->ind_tail_trans_ = utils::make_unique<trans_wrapper_t>(
                    indices_d.data_type(), diff_dst_sp, indices_d.data_type(),
                    jpp.c_block, c_tail, diff_dst_sp);
    }

    return trans_ctx_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_pooling_bwd_t<isa, d_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_,
            new jit_uni_pool_kernel<isa>(
                    pd()->jpp_, pd()->invariant_dst_md())));
    if (pd()->jpp_.tag_kind == jit_memory_tag_kind_t::ncsp)
        CHECK(init_ncsp_trans_ctx());
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_bwd_t<isa, d_type>::execute_backward(
        const data_t *diff_dst, const char *indices, data_t *diff_src,
        const exec_ctx_t &ctx) const {

    using namespace jit_uni_pooling_utils;
    using wsp_data_t = typename prec_traits<wsp_dt_>::type;

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());
    const size_t ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;
    const auto &jpp = pd()->jpp_;
    const auto transpose_facade
            = jit_uni_pooling_utils::bwd_pooling_transpose_facade_t<data_t,
                    wsp_data_t, d_type>(jpp, trans_ctx_.get(), diff_src_d,
                    diff_dst_d, indices_d, wsp_dt_, diff_src, diff_dst, indices,
                    ctx);

    bwd_f32_accum_for_bf16_t f32_accum(jpp, ctx);

    auto get_first_ih = [&](int oh) {
        return nstl::min(nstl::max(oh * jpp.stride_h - jpp.t_pad, 0), jpp.ih);
    };

    auto get_last_ih = [&](int oh) {
        return nstl::min(
                nstl::max(oh * jpp.stride_h - jpp.t_pad + jpp.kh, 0), jpp.ih);
    };
    const auto ker = [&](int ithr, int n, int b_c, int oh, int ur_bc) {
        auto arg = jit_pool_call_s();

        const int ih = get_first_ih(oh);
        assert(IMPLICATION(pd()->ndims() == 3, utils::everyone_is(0, ih, oh)));
        assert(pd()->ndims() != 3 || utils::everyone_is(0, ih, oh));

        const auto c_off = jpp.is_plain() ? b_c * jpp.c_block : b_c;
        if (transpose_facade.should_transpose_src())
            arg.src = transpose_facade.get_src_addr(ithr, ih, jpp);
        else if (jpp.needs_f32_accum_for_bf16)
            arg.src = f32_accum.get_addr_2d(ithr, ih);
        else
            arg.src = &diff_src[diff_src_d.blk_off(n, c_off, ih)];

        if (transpose_facade.should_transpose_dst())
            arg.dst = transpose_facade.get_dst_addr(ithr, oh, jpp);
        else
            arg.dst = &diff_dst[diff_dst_d.blk_off(n, c_off, oh)];

        if (indices) {
            if (transpose_facade.should_transpose_dst())
                arg.indices = transpose_facade.get_indices_addr(ithr, oh, jpp);

            else {
                const size_t ind_off = indices_d.blk_off(n, c_off, oh);
                arg.indices = &indices[ind_off * ind_dt_size];
            }
        }

        const int zero_ih_start = (oh == 0) ? 0 : get_last_ih(oh - 1);
        const int zero_ih_end = (oh == jpp.oh - 1) ? jpp.ih : get_last_ih(oh);

        arg.zero_id = 1;
        arg.zero_ih = zero_ih_end - zero_ih_start;
        if (transpose_facade.should_transpose_src())
            arg.zero_ptr
                    = transpose_facade.get_src_addr(ithr, zero_ih_start, jpp);
        else if (jpp.needs_f32_accum_for_bf16)
            arg.zero_ptr = f32_accum.get_addr_2d(ithr, zero_ih_start);
        else
            arg.zero_ptr
                    = &diff_src[diff_src_d.blk_off(n, c_off, zero_ih_start, 0)];

        const int i_t_overflow = nstl::max(0, jpp.t_pad - oh * jpp.stride_h);
        const int i_b_overflow
                = nstl::max(jpp.ih, oh * jpp.stride_h + jpp.kh - jpp.t_pad)
                - jpp.ih;
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift = i_t_overflow * jpp.kw;
        arg.ker_area_h = static_cast<float>(jpp.kh
                - nstl::max(0, oh * jpp.stride_h - jpp.t_pad + jpp.kh - jpp.ih)
                - nstl::max(0, jpp.t_pad - oh * jpp.stride_h));

        arg.ur_bc = ur_bc;
        arg.b_c = b_c;
        (*kernel_)(&arg);
    };

    auto process_block = [&](int ithr, int n, int b_c, int ur_bc) {
        if (transpose_facade.should_transpose_dst())
            transpose_facade.execute_transpose_input(ithr, n, b_c);

        for (int oh = 0; oh < jpp.oh; ++oh)
            ker(ithr, n, b_c, oh, ur_bc);

        if (transpose_facade.should_transpose_src())
            transpose_facade.execute_transpose_output(ithr, n, b_c);

        if (jpp.needs_f32_accum_for_bf16)
            f32_accum.cvt_to_bf16_slice_2d(
                    ithr, (bfloat16_t *)diff_src, diff_src_d, n, b_c, ur_bc);
    };

    const int nthr = jpp.nthr;

    parallel(nthr, [&](int ithr, int nthr) {
        const auto nb2_c = utils::div_up(jpp.nb_c, jpp.ur_bc);
        const std::size_t work_amount
                = static_cast<std::size_t>(jpp.mb) * nb2_c;
        if (static_cast<std::size_t>(ithr) >= work_amount) return;

        if (transpose_facade.should_fill_input_c_tail_with_zeros())
            transpose_facade.fill_input_c_tail_with_zeros(ithr, jpp);

        std::size_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int n {0}, b2_c {0};
        utils::nd_iterator_init(start, n, jpp.mb, b2_c, nb2_c);
        for (size_t iwork = start; iwork < end; ++iwork) {
            const auto b_c = b2_c * jpp.ur_bc;
            const auto ur_bc = nstl::min(jpp.ur_bc, jpp.nb_c - b_c);

            process_block(ithr, n, b_c, ur_bc);
            utils::nd_iterator_step(n, jpp.mb, b2_c, nb2_c);
        }
    });
}

template <cpu_isa_t isa, data_type_t d_type>
void jit_uni_pooling_bwd_t<isa, d_type>::execute_backward_3d(
        const data_t *diff_dst, const char *indices, data_t *diff_src,
        const exec_ctx_t &ctx) const {
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper indices_d(pd()->workspace_md());
    const size_t ind_dt_size
            = indices ? types::data_type_size(indices_d.data_type()) : 0;

    const auto &jpp = pd()->jpp_;

    using wsp_data_t = typename prec_traits<wsp_dt_>::type;
    using namespace jit_uni_pooling_utils;
    static constexpr int first_ithr = 0;

    const auto transpose_facade
            = bwd_pooling_transpose_facade_t<data_t, wsp_data_t, d_type>(jpp,
                    trans_ctx_.get(), diff_src_d, diff_dst_d, indices_d,
                    wsp_dt_, diff_src, diff_dst, indices, ctx);

    const auto trans_src = transpose_facade.should_transpose_src();
    const auto trans_dst = transpose_facade.should_transpose_dst();

    bwd_f32_accum_for_bf16_t f32_accum(jpp, ctx);

    const size_t input_dt_size = jpp.needs_f32_accum_for_bf16
            ? sizeof(bwd_f32_accum_for_bf16_t::value_type)
            : jpp.dt_size;

    auto get_last_ih = [&](int oh) {
        return nstl::min(
                nstl::max(oh * jpp.stride_h - jpp.t_pad + jpp.kh, 0), jpp.ih);
    };

    auto get_last_id = [&](int od) {
        return nstl::min(
                nstl::max(od * jpp.stride_d - jpp.f_pad + jpp.kd, 0), jpp.id);
    };

    auto ker = [&](int n, int b_c, int od, int oh, int id, int d_t_overflow,
                       int d_b_overflow, bool zero_inp, int kd, int ur_bc,
                       int ithr) {
        auto arg = jit_pool_call_s();

        const int ij = oh * jpp.stride_h;
        const int i_t_overflow = nstl::max(0, jpp.t_pad - ij);
        const int i_b_overflow
                = nstl::max(jpp.ih, ij + jpp.kh - jpp.t_pad) - jpp.ih;
        const int ih = nstl::max(ij - jpp.t_pad, 0);
        const int c_off
                = ((jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c_block
                                                                 : 1)
                * b_c;

        if (trans_src)
            arg.src = transpose_facade.get_src_addr_3d(ithr, id + kd, ih, jpp);
        else if (jpp.needs_f32_accum_for_bf16)
            arg.src = f32_accum.get_addr_3d(ithr, id + kd, ih);
        else
            arg.src = (const void *)&diff_src[diff_src_d.blk_off(
                    n, c_off, id + kd, ih)];

        if (trans_dst)
            arg.dst = transpose_facade.get_dst_addr_3d(ithr, od, oh, jpp);
        else
            arg.dst = (const void
                            *)&diff_dst[diff_dst_d.blk_off(n, c_off, od, oh)];

        if (indices) {
            if (trans_dst) {
                arg.indices = transpose_facade.get_indices_addr_3d(
                        ithr, od, oh, jpp);
            } else {
                const size_t ind_off = indices_d.blk_off(n, c_off, od, oh);
                arg.indices = (const void *)&indices[ind_off * ind_dt_size];
            }
        }

        if (zero_inp) {
            const int zero_id_start = (od == 0) ? 0 : get_last_id(od - 1);
            const int zero_id_end
                    = (od == jpp.od - 1) ? jpp.id : get_last_id(od);

            arg.zero_id = zero_id_end - zero_id_start;

            const int zero_ih_start = (oh == 0) ? 0 : get_last_ih(oh - 1);
            const int zero_ih_end
                    = (oh == jpp.oh - 1) ? jpp.ih : get_last_ih(oh);
            arg.zero_ih = zero_ih_end - zero_ih_start;

            if (trans_src)
                arg.zero_ptr = transpose_facade.get_src_addr_3d(
                        ithr, zero_id_start, zero_ih_start, jpp);
            else if (jpp.needs_f32_accum_for_bf16)
                arg.zero_ptr = f32_accum.get_addr_3d(
                        ithr, zero_id_start, zero_ih_start);
            else
                arg.zero_ptr = &diff_src[diff_src_d.blk_off(
                        n, c_off, zero_id_start, zero_ih_start, 0)];
        } else {
            arg.zero_id = 0;
            arg.zero_ih = 0;
        }

        arg.kd_padding = jpp.kd - d_t_overflow - d_b_overflow;
        arg.kh_padding = jpp.kh - i_t_overflow - i_b_overflow;
        arg.kh_padding_shift = i_t_overflow * jpp.kw
                + d_t_overflow * jpp.kw * jpp.kh + kd * jpp.kw * jpp.kh;
        arg.kd_padding_shift = (i_t_overflow + i_b_overflow) * jpp.kw;
        arg.ker_area_h = (float)(jpp.kh
                                 - nstl::max(0,
                                         oh * jpp.stride_h - jpp.t_pad + jpp.kh
                                                 - jpp.ih)
                                 - nstl::max(0, jpp.t_pad - oh * jpp.stride_h))
                * (jpp.kd
                        - nstl::max(0,
                                od * jpp.stride_d - jpp.f_pad + jpp.kd - jpp.id)
                        - nstl::max(0, jpp.f_pad - od * jpp.stride_d));

        arg.ur_bc = ur_bc;
        arg.b_c = b_c;
        (*kernel_)(&arg);
    };

    auto process_simple = [&](int n, int b_c, int od, int ur_bc, int ithr) {
        const int ik = od * jpp.stride_d;
        const int d_t_overflow = nstl::max(0, jpp.f_pad - ik);
        const int d_b_overflow
                = nstl::max(jpp.id, ik + jpp.kd - jpp.f_pad) - jpp.id;
        const int id = nstl::max(ik - jpp.f_pad, 0);

        for (int oh = 0; oh < jpp.oh; ++oh) {
            ker(n, b_c, od, oh, id, d_t_overflow, d_b_overflow, true, 0, ur_bc,
                    ithr);
        }
    };

    const int nthr = jpp.nthr;

    if (jpp.simple_alg) {
        const dim_t nb2_c = utils::div_up(jpp.nb_c, jpp.ur_bc);

        if (jpp.tag_kind == jit_memory_tag_kind_t::nspc) {
            if (!jpp.needs_f32_accum_for_bf16) {
                parallel_nd(jpp.mb, jpp.od, nb2_c,
                        [&](dim_t n, dim_t od, dim_t b2_c) {
                            const dim_t b_c = b2_c * jpp.ur_bc;
                            const dim_t ur_bc = nstl::min(
                                    dim_t(jpp.ur_bc), jpp.nb_c - b_c);
                            process_simple(n, b_c, od, ur_bc, first_ithr);
                        });
            } else {
                parallel_nd_ext(nthr, jpp.mb, nb2_c,
                        [&](dim_t ithr, dim_t nthr, dim_t n, dim_t b2_c) {
                            const dim_t b_c = b2_c * jpp.ur_bc;
                            const dim_t ur_bc = nstl::min(
                                    dim_t(jpp.ur_bc), jpp.nb_c - b_c);
                            for (int od = 0; od < jpp.od; ++od) {
                                process_simple(n, b_c, od, ur_bc, ithr);
                            }
                            f32_accum.cvt_to_bf16_slice_3d(ithr,
                                    (bfloat16_t *)diff_src, diff_src_d, n, b_c,
                                    ur_bc);
                        });
            }
        } else {
            assert(jpp.ur_bc == 1);
            if (trans_src || trans_dst || jpp.needs_f32_accum_for_bf16) {
                parallel_nd_ext(nthr, jpp.mb, jpp.nb_c,
                        [&](dim_t ithr, dim_t nthr, dim_t n, dim_t b_c) {
                            if (trans_src)
                                transpose_facade.execute_transpose_input(
                                        ithr, n, b_c);
                            for (int od = 0; od < jpp.od; ++od) {
                                process_simple(n, b_c, od, 1, ithr);
                            }
                            if (trans_dst)
                                transpose_facade.execute_transpose_output(
                                        ithr, n, b_c);
                            if (jpp.needs_f32_accum_for_bf16)
                                f32_accum.cvt_to_bf16_slice_3d(ithr,
                                        (bfloat16_t *)diff_src, diff_src_d, n,
                                        b_c, 1);
                        });
            } else {
                parallel_nd(jpp.mb, jpp.nb_c, jpp.od,
                        [&](dim_t n, dim_t b_c, dim_t od) {
                            process_simple(n, b_c, od, 1, first_ithr);
                        });
            }
        }
    } else {
        const data_t zero_val = 0;
        if (!jpp.needs_f32_accum_for_bf16) {
            if (jpp.tag_kind == jit_memory_tag_kind_t::nspc) {
                const size_t chunk_size = (size_t)jpp.ih * jpp.iw * jpp.c;
                parallel_nd(jpp.mb, jpp.id, [&](dim_t n, dim_t id) {
                    const size_t offset
                            = ((size_t)n * jpp.id + id) * chunk_size;
                    PRAGMA_OMP_SIMD()
                    for (size_t idx = 0; idx < chunk_size; ++idx)
                        diff_src[offset + idx] = zero_val;
                });
            } else {
                if (!trans_src) {
                    const size_t chunk_size
                            = (size_t)jpp.id * jpp.ih * jpp.iw * jpp.c_block;
                    parallel_nd_ext(nthr, jpp.mb, jpp.nb_c,
                            [&](dim_t ithr, dim_t nthr, dim_t n, dim_t b_c) {
                                const size_t offset
                                        = ((size_t)n * jpp.nb_c + b_c)
                                        * chunk_size;
                                PRAGMA_OMP_SIMD()
                                for (size_t idx = 0; idx < chunk_size; ++idx)
                                    diff_src[offset + idx] = zero_val;
                            });
                }
            }
        }

        const auto nb2_c = utils::div_up(jpp.nb_c, jpp.ur_bc);
        if (trans_src || trans_dst || jpp.needs_f32_accum_for_bf16) {
            parallel_nd_ext(nthr, jpp.mb, nb2_c,
                    [&](dim_t ithr, dim_t nthr, dim_t n, dim_t b2_c) {
                        const dim_t b_c = b2_c * jpp.ur_bc;

                        if (trans_dst) {
                            transpose_facade.execute_transpose_input(
                                    ithr, n, b_c);

                            size_t block_size = jpp.c_block * jpp.id * jpp.ih
                                    * jpp.iw * input_dt_size;

                            const void *src = transpose_facade.get_src_addr_3d(
                                    ithr, 0, 0, jpp);
                            std::memset((void *)src, zero_val, block_size);
                        }

                        if (jpp.needs_f32_accum_for_bf16)
                            f32_accum.zero_data(ithr);

                        const dim_t ur_bc
                                = nstl::min(dim_t(jpp.ur_bc), jpp.nb_c - b_c);
                        for (dim_t kd = 0; kd < jpp.kd; ++kd) {
                            for (int od = 0; od < jpp.od; ++od) {
                                const dim_t ik
                                        = static_cast<dim_t>(od) * jpp.stride_d;
                                const dim_t d_t_overflow
                                        = nstl::max(dim_t(0), jpp.f_pad - ik);
                                const dim_t d_b_overflow
                                        = nstl::max(dim_t(jpp.id),
                                                  ik + jpp.kd - jpp.f_pad)
                                        - jpp.id;
                                if (kd >= jpp.kd - d_t_overflow - d_b_overflow)
                                    continue;
                                const dim_t id
                                        = nstl::max(ik - jpp.f_pad, dim_t(0));
                                for (dim_t oh = 0; oh < jpp.oh; ++oh) {
                                    ker(n, b_c, od, oh, id, d_t_overflow,
                                            d_b_overflow, false, kd, ur_bc,
                                            ithr);
                                }
                            }
                        }

                        if (trans_src)
                            transpose_facade.execute_transpose_output(
                                    ithr, n, b_c);

                        if (jpp.needs_f32_accum_for_bf16)
                            f32_accum.cvt_to_bf16_slice_3d(ithr,
                                    (bfloat16_t *)diff_src, diff_src_d, n, b_c,
                                    ur_bc);
                    });
        } else {
            for (dim_t kd = 0; kd < jpp.kd; ++kd) {
                parallel_nd(jpp.mb, nb2_c, [&](dim_t n, dim_t b2_c) {
                    const dim_t b_c = b2_c * jpp.ur_bc;
                    const dim_t ur_bc
                            = nstl::min(dim_t(jpp.ur_bc), jpp.nb_c - b_c);
                    for (int od = 0; od < jpp.od; ++od) {
                        const dim_t ik = static_cast<dim_t>(od) * jpp.stride_d;
                        const dim_t d_t_overflow
                                = nstl::max(dim_t(0), jpp.f_pad - ik);
                        const dim_t d_b_overflow
                                = nstl::max(dim_t(jpp.id),
                                          ik + jpp.kd - jpp.f_pad)
                                - jpp.id;
                        if (kd >= jpp.kd - d_t_overflow - d_b_overflow)
                            continue;
                        const dim_t id = nstl::max(ik - jpp.f_pad, dim_t(0));
                        for (dim_t oh = 0; oh < jpp.oh; ++oh) {
                            ker(n, b_c, od, oh, id, d_t_overflow, d_b_overflow,
                                    false, kd, ur_bc, first_ithr);
                        }
                    }
                });
            }
        }
    }
}

template struct jit_uni_pooling_fwd_t<sse41, data_type::f32>;
template struct jit_uni_pooling_bwd_t<sse41, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx, data_type::f32>;
template struct jit_uni_pooling_bwd_t<avx, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx2, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx2_vnni_2, data_type::bf16>;
template struct jit_uni_pooling_fwd_t<avx2_vnni_2, data_type::f16>;
template struct jit_uni_pooling_bwd_t<avx2, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx512_core, data_type::f32>;
template struct jit_uni_pooling_bwd_t<avx512_core, data_type::f32>;
template struct jit_uni_pooling_fwd_t<avx512_core, data_type::bf16>;
template struct jit_uni_pooling_bwd_t<avx512_core, data_type::bf16>;
template struct jit_uni_pooling_fwd_t<avx512_core_fp16, data_type::f16>;
template struct jit_uni_pooling_bwd_t<avx512_core_fp16, data_type::f16>;

template struct jit_uni_pooling_fwd_t<avx512_core_fp16, data_type::f8_e5m2>;
template struct jit_uni_pooling_fwd_t<avx512_core_fp16, data_type::f8_e4m3>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
