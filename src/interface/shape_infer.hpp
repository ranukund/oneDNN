/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef INTERFACE_SHAPE_INFER_HPP
#define INTERFACE_SHAPE_INFER_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "interface/logical_tensor.hpp"
#include "interface/op.hpp"

namespace dnnl {
namespace graph {
namespace impl {

/// convert shape to ncx or oix
static dims canonicalize(const dims &shape, const std::string &format);

static inline dims ncx2nxc(const dims &shape);

/// make a dims according to the format. Only for data format ncx or nxc.
static inline dims make_data_dims(
        const std::string &format, const dim_t n, const dim_t c, const dims &x);

/// make a dims according to the format. Only for filter format xio or oix.
static inline dims make_filter_dims(
        const std::string &format, const dim_t i, const dim_t o, const dims &x);

/// validate the inferred shape with the expected one.
static inline bool validate(const dims &inferred, const dims &expected);

/// get the dense strides of a given shape
/// eg. (3, 4, 5) -> (20, 5, 1)
static inline dims get_dense_strides(const dims &shape);

/// shapes of the logical tensors in the vector are known
static inline bool every_shape_is_known(
        const std::vector<logical_tensor_t *> &lts);

static inline bool verify_shapes_in_range(
        const std::vector<logical_tensor_t *> &lts, const size_t begin,
        const size_t end, const std::function<bool(const dims)> &validator);

void set_shape_and_strides(logical_tensor_t &lt, const dims &shape);

static inline void set_shapes_in_range(
        const std::vector<logical_tensor_t *> &lts, const size_t begin,
        const size_t end, const dims &shape);

/// infer the padding sizes according auto_pad type
static status_t infer_auto_pad(const dim_t in_dim, const dim_t stride,
        const dim_t kernel, const dim_t dilation, const std::string &auto_pad,
        dim_t &pad_begin, dim_t &pad_end, bool is_deconv = false);

/// numpy broadcasting
/// TODO(xxx): 0-D broadcasting?
static status_t broadcast(const dims &lhs, const dims &rhs, dims &broadcasted);

/// This function assumes the size of all vectors are correct. Eg. size of
/// strides/dilations/pads should be the same as spatial size of src_dims and
/// fil_dims. Size of output_dims should be the same as size of src_dims.
static inline void infer_conv_ncx_oix(const dims &src_dims,
        const dims &fil_dims, const dims &strides, const dims &dilations,
        const dims &pads_begin, const dims &pads_end, dims &output_dims);

/// Calculate convolution output shape according to the input shapes. If
/// auto_pad, the real size of pads_begin and pads_end will also be calculated.
/// The inferred output shape will be written to the logical tensor in outputs.
/// The inferred pads_begin and pads_end will be attached to the operator
/// directly. Hence the function will change the state of the input operator.
static status_t infer_conv_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_conv_bprop_data_output_shape(op_t *n,

        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_conv_bprop_filters_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_convtranspose_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

// check if output shape is already known
// if shape is unknown, infer output shape (change output lt)
// otherwise infer pad (change op attrs)
status_t infer_pool_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_matmul_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_identity_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

static status_t identity_output_shape_on_pos(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs,
        std::vector<uint32_t> &positions);

status_t infer_bias_backprop_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_bias_add_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_norm_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_norm_bprop_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_elemwise_arithmetic_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_bn_fwd_train_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_bn_bwd_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_concat_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_unsupported_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

/// Shape inference function for PowBackpropExponent
status_t infer_exponent_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

status_t infer_reduce_sum_output_shape(op_t *n,
        std::vector<logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs);

} // namespace impl
} // namespace graph
} // namespace dnnl

#endif
