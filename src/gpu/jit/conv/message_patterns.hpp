/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef GPU_JIT_CONV_MESSAGE_PATTERNS_HPP
#define GPU_JIT_CONV_MESSAGE_PATTERNS_HPP

#include "common/type_helpers.hpp"
#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/ir/message_patterns.hpp"
#include "gpu/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Helper struct for to handle indexing with convolution problem dimensions
struct conv_dim_t {
    conv_dim_t() = default;
    conv_dim_t(const conv_dim_t &) = default;
    void operator=(int dim) { *this = conv_dim_t(dim); };
    int as_int() const { return dim_; };

    static const int mb_value = 0;
    static const int g_value = 1;
    static const int oc_value = 2;
    static const int ic_value = 3;
    static const int od_value = 4;
    static const int id_value = 5;
    static const int kd_value = 6;
    static const int oh_value = 7;
    static const int ih_value = 8;
    static const int kh_value = 9;
    static const int ow_value = 10;
    static const int iw_value = 11;
    static const int kw_value = 12;
    static const int NDIMS = 13;

    static conv_dim_t mb() { return mb_value; }
    static conv_dim_t g() { return g_value; }
    static conv_dim_t oc() { return oc_value; }
    static conv_dim_t ic() { return ic_value; }
    static conv_dim_t od() { return od_value; }
    static conv_dim_t id() { return id_value; }
    static conv_dim_t kd() { return kd_value; }
    static conv_dim_t oh() { return oh_value; }
    static conv_dim_t ih() { return ih_value; }
    static conv_dim_t kh() { return kh_value; }
    static conv_dim_t ow() { return ow_value; }
    static conv_dim_t iw() { return iw_value; }
    static conv_dim_t kw() { return kw_value; }
    static const std::array<conv_dim_t, NDIMS> &dims() {
        static const std::array<conv_dim_t, NDIMS> dims = {mb(), g(), oc(),
                ic(), od(), id(), kd(), oh(), ih(), kh(), ow(), iw(), kw()};
        return dims;
    }

    std::string str() const {
        switch (dim_) {
            case mb_value: return "mb";
            case g_value: return "g";
            case oc_value: return "oc";
            case ic_value: return "ic";
            case od_value: return "od";
            case id_value: return "id";
            case kd_value: return "kd";
            case oh_value: return "oh";
            case ih_value: return "ih";
            case kh_value: return "kh";
            case ow_value: return "ow";
            case iw_value: return "iw";
            case kw_value: return "kw";
            default: return "unknown";
        }
    }

    bool operator<(conv_dim_t other) const { return dim_ < other.dim_; }

private:
    conv_dim_t(int dim) : dim_(dim) {};

    int dim_;
};

struct conv_stride_layout_t : public stride_layout_t<conv_dim_t> {
    using base_layout_t = stride_layout_t<conv_dim_t>;

    enum class input_tensor_t {
        src,
        wei,
        dst,
    };
    conv_stride_layout_t(const conv_problem_t &prb, input_tensor_t type)
        : base_layout_t(0) {

        const memory_desc_t &md = [&]() {
            if (prb.is_fwd) {
                if (type == input_tensor_t::src)
                    return prb.a_md();
                else if (type == input_tensor_t::wei)
                    return prb.b_md();
                else
                    ir_error_not_expected();
            } else if (prb.is_bwd_d) {
                if (type == input_tensor_t::dst)
                    return prb.a_md();
                else if (type == input_tensor_t::wei)
                    return prb.b_md();
                else
                    ir_error_not_expected();
            } else if (prb.is_bwd_w) {
                if (type == input_tensor_t::src)
                    return prb.a_md();
                else if (type == input_tensor_t::dst)
                    return prb.b_md();
                else
                    ir_error_not_expected();
            } else {
                ir_error_not_expected();
            }
            return prb.a_md();
        }();
        const memory_desc_wrapper mdw {md};

        type_size = mdw.data_type_size();
        buffer_size = type_size;
        for (int i = 0; i < mdw.ndims(); i++)
            buffer_size *= mdw.padded_dims()[i];

        const auto &blk = mdw.blocking_desc();
        auto s = strides.begin();

        auto write_strides =
                [&](std::array<base_layout_t::stride_dim_t, MAX_NDIMS>::iterator
                                s,
                        conv_dim_t conv_dim, dim_t desc_dim, dim_t size,
                        dim_t access_stride = 1, bool can_overflow = false) {
                    // Size 1 dimensions are effectively non-existent
                    if (size == 1) return s;

                    bool is_complex = access_stride == 0;

                    // Complex expressions can produce any number as f_dim(dim)
                    if (is_complex) access_stride = 1;

                    auto outer = size;
                    auto stride = 1;
                    for (int j = 0; j < blk.inner_nblks; j++) {
                        const dim_t blk_size = blk.inner_blks[j];
                        if (blk.inner_idxs[j] == desc_dim) {
                            outer = utils::div_up(outer, blk_size);
                            auto next = stride;
                            if (access_stride > 1) {
                                if (blk_size % access_stride == 0) {
                                    next *= access_stride;
                                    access_stride = 1;
                                } else {
                                    access_stride = 1;
                                    is_complex = true;
                                }
                            }
                            ir_assert(s != strides.end());
                            *s++ = stride_dim_t(conv_dim, blk_size, next,
                                    can_overflow, is_complex);
                            ndims++;
                        }
                        stride *= blk_size;
                    }
                    ir_assert(s != strides.end());
                    *s++ = stride_dim_t(conv_dim, outer,
                            access_stride * blk.strides[desc_dim], can_overflow,
                            is_complex);
                    ndims++;
                    return s;
                };

        switch (type) {
            case input_tensor_t::src:
            case input_tensor_t::dst: {
                bool is_src = type == input_tensor_t::src;
                int i = 0;
                s = write_strides(s, conv_dim_t::mb(), i++, prb.mb);
                if (is_src)
                    s = write_strides(s, conv_dim_t::ic(), i++, prb.ic);
                else
                    s = write_strides(s, conv_dim_t::oc(), i++, prb.oc);

                if (mdw.ndims() >= 5) {
                    bool is_padded = is_src
                            && (prb.pd
                                    || prb.id < prb.od * prb.sd
                                                    + (prb.kd - 1)
                                                            * (prb.dd + 1));
                    auto x_dim = !prb.is_bwd_d ? conv_dim_t::od()
                                               : conv_dim_t::id();
                    auto x = !prb.is_bwd_d ? prb.od : prb.id;
                    auto xas = !prb.is_bwd_d ? prb.sd : prb.sd == 1;
                    auto kx = prb.kd;
                    auto kxas = !prb.is_bwd_w ? prb.dd + 1 : prb.dd == 0;
                    s = write_strides(s, x_dim, i, x, xas, is_padded);
                    s = write_strides(
                            s, conv_dim_t::kd(), i++, kx, kxas, is_padded);
                }
                if (mdw.ndims() >= 4) {
                    bool is_padded = is_src
                            && (prb.ph
                                    || prb.ih < prb.oh * prb.sh
                                                    + (prb.kh - 1)
                                                            * (prb.dh + 1));
                    auto x_dim = !prb.is_bwd_d ? conv_dim_t::oh()
                                               : conv_dim_t::ih();
                    auto x = !prb.is_bwd_d ? prb.oh : prb.ih;
                    auto xas = !prb.is_bwd_d ? prb.sh : prb.sh == 1;
                    auto kx = prb.kh;
                    auto kxas = !prb.is_bwd_w ? prb.dh + 1 : prb.dh == 0;
                    s = write_strides(s, x_dim, i, x, xas, is_padded);
                    s = write_strides(
                            s, conv_dim_t::kh(), i++, kx, kxas, is_padded);
                }
                bool is_padded = is_src
                        && (prb.pw
                                || prb.iw < prb.ow * prb.sw
                                                + (prb.kw - 1) * (prb.dw + 1));
                auto x_dim
                        = !prb.is_bwd_d ? conv_dim_t::ow() : conv_dim_t::iw();
                auto x = !prb.is_bwd_d ? prb.ow : prb.iw;
                auto xas = !prb.is_bwd_d ? prb.sw : prb.sw == 1;
                auto kx = prb.kw;
                auto kxas = !prb.is_bwd_w ? prb.dw + 1 : prb.dw == 0;
                s = write_strides(s, x_dim, i, x, xas, is_padded);
                s = write_strides(
                        s, conv_dim_t::kw(), i++, kx, kxas, is_padded);
                break;
            }
            case input_tensor_t::wei: {
                int i = 0;
                if (prb.with_groups)
                    s = write_strides(s, conv_dim_t::g(), i++, prb.g);
                s = write_strides(s, conv_dim_t::oc(), i++, prb.oc);
                s = write_strides(s, conv_dim_t::ic(), i++, prb.ic);
                if (mdw.ndims() >= 5 + prb.with_groups) {
                    s = write_strides(s, conv_dim_t::kd(), i++, prb.kd);
                }
                if (mdw.ndims() >= 4 + prb.with_groups) {
                    s = write_strides(s, conv_dim_t::kh(), i++, prb.kh);
                }
                s = write_strides(s, conv_dim_t::kw(), i++, prb.kw);
                break;
            }
            default: assert("unimplemented");
        }

        // Normalize into a sorted order by stride, dimension, size, and
        // dimension id
        std::sort(strides.begin(), strides_end());
    }
};

inline std::ostream &operator<<(
        std::ostream &out, conv_stride_layout_t::input_tensor_t t) {
    switch (t) {
        case conv_stride_layout_t::input_tensor_t::src: out << "src"; break;
        case conv_stride_layout_t::input_tensor_t::wei: out << "wei"; break;
        case conv_stride_layout_t::input_tensor_t::dst: out << "dst"; break;
    }
    return out;
}
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif
