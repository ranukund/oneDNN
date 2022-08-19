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

#include "lowering.hpp"
#include <algorithm>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include "fusible_op.hpp"
#include "graph.hpp"
#include "pass/pass.hpp"
#include "runtime_op.hpp"
#include "visitor.hpp"
#include <compiler/ir/builder.hpp>
#include <compiler/ir/graph/dynamic_dispatch_key.hpp>
#include <compiler/ir/graph/dynamic_utils.hpp>
#include <compiler/ir/graph/fused_op.hpp>
#include <compiler/ir/ir_utils.hpp>
#include <compiler/ir/pass/ir_copy_internal.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/cpu/local_tensor_lower.hpp>
#include <compiler/ir/transform/dead_write_eliminate.hpp>
#include <compiler/ir/transform/dyn_tsr_transform.hpp>
#include <compiler/ir/transform/index2var.hpp>
#include <compiler/ir/transform/tensor2var.hpp>
#include <microkernel/builtin.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <ops/fusible/reduce.hpp>
#include <ops/matmul_core.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <unordered_map>
#include <util/scoped_timer.hpp>

namespace sc {

SC_MODULE(graph.lowering)

struct result_dump_config_t {
    bool enabled_ = false;
    std::vector<std::string> filter_;
    std::string path_ = "./dump";
    bool binary_format_ = false;
    size_t bytes_per_dump_ = 0;

    bool should_function_dump(const std::string &name) {
        if (filter_.empty()) { return true; }
        for (auto &f : filter_) {
            if (utils::string_startswith(name, f)) { return true; }
        }
        return false;
    }

    result_dump_config_t(const std::string &cfg) {
        if (cfg.empty()) { return; }
        auto configs = utils::string_split(cfg, ",");
        for (auto &c : configs) {
            auto kv = utils::string_split(c, "=");
            if (kv.size() != 2) {
                SC_MODULE_WARN << "Bad graph result dump config: " << c;
                continue;
            }
            if (kv[0] == "filter") {
                enabled_ = true;
                filter_ = utils::string_split(kv[1], ":");
            } else if (kv[0] == "path") {
                enabled_ = true;
                path_ = kv[1];
            } else if (kv[0] == "format") {
                enabled_ = true;
                binary_format_ = std::stoi(kv[1]);
            } else if (kv[0] == "bytes") {
                enabled_ = true;
                bytes_per_dump_ = std::stoull(kv[1]);
            } else {
                SC_MODULE_WARN << "Bad dump config key name " << kv[0];
                continue;
            }
        }
        if (enabled_) {
            SC_MODULE_WARN << "The generated code will dump tensor results to "
                           << path_
                           << ", filter=" << utils::print_vector(filter_)
                           << ", binaryformat=" << binary_format_
                           << ", byteslimit=" << bytes_per_dump_;
        }
    }
};

static expr make_global_string(
        const ir_module_ptr &mod, const std::string &v, int &counter) {
    std::string name = "__gstring";
    name += std::to_string(counter++);
    auto contents = std::make_shared<static_data_t>(v.c_str(), v.size() + 1);
    auto ret = builder::make_tensor(name, {v.size() + 1}, datatypes::s8,
            address_space::automatic, contents);
    auto def = builder::make_var_tensor_def_unattached(
            ret, linkage::private_global);
    mod->add_global_var(def.checked_as<define>());
    return ret;
}

static void make_dump_tensor_call(const std::vector<expr> &outs,
        const sc_op_ptr &node, const ir_module_ptr &ret_mod,
        const std::string &callee_name, int &global_str_counter,
        result_dump_config_t &dump_config, const expr &dump_out_path,
        stmts_node_t *target_body) {
    for (size_t i = 0; i < outs.size(); i++) {
        auto &out = outs[i];
        auto &graph_tsr = node->get_outputs()[i];
        if (!out.isa<tensor>()) continue;
        auto tsr = out.checked_as<tensor>();
        std::stringstream tensor_name;
        tensor_name << callee_name << '.' << tsr->name_ << '.'
                    << graph_tsr->details_.get_format();
        auto namestr = make_global_string(
                ret_mod, tensor_name.str(), global_str_counter);
        std::stringstream shape_name;
        size_t total_shape1 = utils::get_sizeof_type(tsr->elem_dtype_);
        for (auto &dimv : tsr->dims_) {
            auto dim = get_const_as_int(dimv.checked_as<constant_c>());
            total_shape1 *= dim;
            shape_name << dim << ',';
        }
        auto shapestr = make_global_string(
                ret_mod, shape_name.str(), global_str_counter);
        auto the_call = builtin::call_dump_tensor(out, namestr, shapestr,
                total_shape1, dump_config.bytes_per_dump_, dump_out_path,
                dump_config.binary_format_,
                static_cast<uint64_t>(tsr->elem_dtype_));
        target_body->seq_.emplace_back(
                builder::make_evaluate_unattached(the_call));
    }
}

static void make_value_check_call(const std::vector<expr> &outs,
        const ir_module_ptr &ret_mod, const std::string &callee_name,
        int &global_str_counter, stmts_node_t *target_body) {
    for (auto &out : outs) {
        auto tsr = out.checked_as<tensor>();
        if (tsr->elem_dtype_.type_code_ != sc_data_etype::F32) { continue; }
        auto namestr = make_global_string(
                ret_mod, callee_name + "." + tsr->name_, global_str_counter);
        size_t total_shape1 = utils::get_sizeof_type(tsr->elem_dtype_);
        for (auto &dimv : tsr->dims_) {
            total_shape1 *= get_const_as_int(dimv.checked_as<constant_c>());
        }
        auto the_call = builtin::call_value_check(out, namestr, total_shape1);
        target_body->seq_.emplace_back(
                builder::make_evaluate_unattached(the_call));
    }
}

static graph_tensor_ptr get_linked_output_tsr(const graph_tensor_ptr &ltensor) {
    if (!ltensor->uses_.empty()) {
        for (size_t i = 0; i < ltensor->uses_.size(); i++) {
            if (ltensor->uses_[i].second->isa<tensor_view_op_t>()) {
                auto reshape = ltensor->uses_[i].second;
                auto next_ltensor = reshape->get_outputs()[0];
                for (auto &cld : next_ltensor->uses_) {
                    if (cld.second->isa<output_op>()) {
                        return cld.second->get_inputs()[cld.first];
                    } else if (cld.second->isa<tensor_view_op_t>()) {
                        auto cur_linked_out
                                = get_linked_output_tsr(next_ltensor);
                        if (cur_linked_out) { return cur_linked_out; }
                    }
                }
            }
        }
    }
    return nullptr;
}

static bool has_output_uses(const graph_tensor_ptr &ltensor) {
    if (!ltensor->uses_.empty()) {
        for (size_t i = 0; i < ltensor->uses_.size(); i++) {
            if (ltensor->uses_[i].second->isa<output_op>()) { return true; }
        }
    }
    return false;
}

struct lowering_visitor_state_t {
    std::unordered_map<graph_tensor_ptr, size_t> tensor_pending_refcount_;
    op_visitor_t::updater_func topo_sorter_;
    std::vector<size_t> op_exec_tick_;
    std::vector<bool> op_visited_;
    //  need to visit the input outs in reversed order to align to old lowering
    //  input argument order (like pop_back_selector). Our visitor must visit
    //  the input ops first
    std::list<sc_op_ptr>::iterator input_op_itr;
    size_t cur_tick_ = 0;
    size_t max_tensor_size_;
    bool is_dynamic_;

    lowering_visitor_state_t(sc_graph_t &g)
        : topo_sorter_ {op_visitor_t::create_DAG_updater(g.ops_.size())}
        , op_exec_tick_(g.ops_.size())
        , op_visited_(g.ops_.size()) {
        max_tensor_size_ = 0;
        is_dynamic_ = g.is_dynamic();
        if (!is_dynamic_) {
            for (auto &op : g.ops_) {
                for (auto &tsr : op->get_outputs()) {
                    max_tensor_size_ = std::max(max_tensor_size_,
                            tsr->details_.get_blocking_byte_size());
                }
            }
        }
    }

    size_t &get_tensor_pending_refcount(const graph_tensor_ptr &p) {
        auto itr = tensor_pending_refcount_.find(p);
        if (itr == tensor_pending_refcount_.end()) {
            auto ret = tensor_pending_refcount_.insert(
                    std::make_pair(p, p->uses_.size()));
            return ret.first->second;
        }
        return itr->second;
    }

    op_visitor_t::updater_func get_updater() {
        auto ths = this;
        return [ths](op_visitor_t *vis, const sc_op_ptr &op) {
            for (auto &in : op->get_inputs()) {
                ths->get_tensor_pending_refcount(in)--;
            }
            auto tick = ths->cur_tick_++;
            if (op->isa<output_op>() || op->isa<constant_op_t>()) {
                ths->op_exec_tick_[op->logical_op_id_] = 0;
            } else {
                ths->op_exec_tick_[op->logical_op_id_] = tick;
            }
            ths->op_visited_[op->logical_op_id_] = true;
            ths->topo_sorter_(vis, op);
        };
    }

    // find the distance of an op to the visited ops
    int get_op_distance_to_visited_set(sc_op *op, std::vector<int> &d) {
        auto id = op->logical_op_id_;
        if (op_visited_[id]) { return 0; }
        if (d[id] != 0) { return d[id]; }
        if (op->isa<output_op>()) {
            d[id] = 0;
            return 0;
        }
        int ret = -1;
        for (auto &v : op->get_inputs()) {
            int cur_d
                    = get_op_distance_to_visited_set(v->producer_owner_, d) + 1;
            ret = std::max(ret, cur_d);
        }
        d[id] = ret;
        return ret;
    }

    static constexpr float distance_factor = 2.0f;
    // for each input tensor, check if the refcount=1. If so, it means that
    // after the Op is visited, the input tensor is no longer needed compute the
    // score of each visitable candidate op. the score is "SUM_{each input
    // tensor}(normalized_sizeof(tensor)/ref_count_modifier*heat_modifier) -
    // SUM_{each output tensor}(normalized_sizeof(tensor)+ distance_modifier)"
    float evaluate_op_score(sc_op *op, std::vector<int> &distance_to_visited) {
        float cur_score = 0;

        for (auto &in : op->get_inputs()) {
            // if the input tensor is input_op, there is no temp buffer to be
            // free'd
            if (!in->producer_owner_->isa<input_op>()) {
                // compute the heat modifier of the tensor. The hotter
                // the tensor is (computed lately), the larger the
                // modifier.
                auto owner = in->producer_owner_;
                auto tick_diff
                        = cur_tick_ - op_exec_tick_[owner->logical_op_id_];
                assert(cur_tick_ > op_exec_tick_[owner->logical_op_id_]);
                float heat_modifier;
                switch (tick_diff) {
                    case 0:
                    case 1: heat_modifier = 2.5f; break;
                    case 2: heat_modifier = 1.5f; break;
                    default: heat_modifier = 1.0f;
                }
                // if it is last use, ref_count_modifier=1. If not,
                // ref_count_modifier=number of uses
                size_t ref_count_modifier;
                if (this->get_tensor_pending_refcount(in) == 1) {
                    ref_count_modifier = 1;
                } else {
                    ref_count_modifier = in->uses_.size();
                }
                float cur_tsr = is_dynamic_
                        ? heat_modifier
                        : float(in->details_.get_blocking_byte_size())
                                / ref_count_modifier / max_tensor_size_
                                * heat_modifier;
                cur_score += cur_tsr;
            }
        }
        for (auto &out : op->get_outputs()) {
            // if this output is connected to output op, it is not a temp
            // buffer, and we don't need to count its size
            if (out->uses_.size() == 1UL
                    && out->uses_[0].second->isa<output_op>()) {
                continue;
            }
            int distance = 1;
            for (auto &use : out->uses_) {
                distance = std::max(distance,
                        get_op_distance_to_visited_set(
                                use.second.get(), distance_to_visited));
            }
            float cur_tsr = (distance - 1) * distance_factor
                    + (is_dynamic_ ? 1.f
                                   : float(out->details_
                                                     .get_blocking_byte_size())
                                            / max_tensor_size_);
            cur_score -= cur_tsr;
        }
        return cur_score;
    }

    using queue_iterator_t = std::list<sc_op_ptr>::iterator;
    op_visitor_t::selector_func get_selector() {
        auto ths = this;
        return [ths](op_visitor_t *vis) -> sc_op_ptr {
            if (ths->cur_tick_ == 0) {
                ths->input_op_itr = vis->to_visit_.end();
                --ths->input_op_itr;
            }
            if (ths->input_op_itr != vis->to_visit_.end()) {
                // if there is input ops, return and advance the input_op_itr
                auto ret = *ths->input_op_itr;
                auto to_remove = ths->input_op_itr;
                if (ths->input_op_itr == vis->to_visit_.begin()) {
                    ths->input_op_itr = vis->to_visit_.end();
                } else {
                    --ths->input_op_itr;
                }
                vis->to_visit_.erase(to_remove);

                SC_MODULE_INFO << "Scheduling const/input: iter "
                               << ths->cur_tick_ << ", Op " << ret->op_name_
                               << "_" << ret->logical_op_id_;
                return ret;
            }
            // fast path: if there is only one op, just pop it
            if (vis->to_visit_.size() == 1) {
                auto ret = vis->to_visit_.back();
                vis->to_visit_.pop_back();
                return ret;
            }
            float best_score = std::numeric_limits<float>::lowest();
            std::list<sc_op_ptr>::reverse_iterator to_remove;

            std::vector<int> distance(ths->op_visited_.size());
            // visit the queue in reversed order to align to old lowering input
            // argument order (like pop_back_selector)
            for (auto itr = vis->to_visit_.rbegin();
                    itr != vis->to_visit_.rend(); ++itr) {
                auto &op = *itr;
                assert(!op->isa<input_op>() && !op->isa<constant_op_t>());
                float cur_score = ths->evaluate_op_score(op.get(), distance);
                SC_MODULE_INFO << "Scheduling score: iter " << ths->cur_tick_
                               << ", Op " << op->op_name_ << "_"
                               << op->logical_op_id_ << " = " << cur_score;
                if (cur_score > best_score) {
                    best_score = cur_score;
                    to_remove = itr;
                }
            }
            auto ret = *to_remove;
            SC_MODULE_INFO << "Scheduling selects: iter " << ths->cur_tick_
                           << ", Op " << ret->op_name_ << "_"
                           << ret->logical_op_id_;
            vis->to_visit_.erase(std::next(to_remove).base());
            return ret;
        };
    }
};

namespace graph {
std::string get_tensor_name(graph_tensor *t, sc_op *linked_output) {
    std::string tensor_name;
    if (t->producer_owner_->get_outputs().size() == 1UL) {
        tensor_name = t->producer_owner_->attrs_.get_or_else(
                "temp.name", tensor_name);
    }
    if (tensor_name.empty() && linked_output
            && linked_output->get_inputs().size() == 1UL) {
        tensor_name
                = linked_output->attrs_.get_or_else("temp.name", tensor_name);
    }
    return tensor_name;
}
} // namespace graph

static bool need_query_next_first(const sc_op_ptr &node) {
    return node->get_outputs()[0]->details_.get_format_candidates().size() > 1;
}

expr call_op_dynamic_query_function(
        const sc_op_ptr &op, const std::vector<expr> &args) {
    if (op->isa<ops::matmul_core_op_t>()) {
        assert(args.size() == 13);
        return builtin::call_matmul_core_query_format(args[0], args[1], args[2],
                args[3], args[4], args[5], args[6], args[7], args[8], args[9],
                args[10], args[11], args[12]);
    } else if (op->isa<unary_elementwise_op_t>()) {
        assert(args.size() == 7);
        return builtin::call_unary_fusible_op_query_format(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
    } else if (op->isa<binary_elementwise_op_t>()) {
        assert(args.size() == 9);
        return builtin::call_binary_fusible_op_query_format(args[0], args[1],
                args[2], args[3], args[4], args[5], args[6], args[7], args[8]);
    } else if (op->isa<reorder_op_t>()) {
        assert(args.size() == 7);
        return builtin::call_reorder_op_query_format(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
    } else if (op->isa<reduce_op_t>()) {
        assert(args.size() == 7);
        return builtin::call_reduce_op_query_format(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
    } else if (op->isa<tensor_view_op_t>()) {
        assert(args.size() == 7);
        return builtin::call_tensor_view_op_query_format(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
    } else {
        COMPILE_ASSERT(
                false, "unsupported op query function: " << op->op_name_);
    }
    return expr();
}

class tv_tsr_replacer_t : public ir_copier_impl_t {
public:
    using ir_copier_impl_t::dispatch;
    using ir_copier_impl_t::view;
    tv_tsr_replacer_t(std::unordered_map<expr_c, expr> &replace_map,
            bool create_var_tensor = false)
        : ir_copier_impl_t(replace_map, create_var_tensor) {}
    void view(define_c v) override {
        if (replace_map_.find(v->var_) != replace_map_.end()) {
            returned_stmt_ = builder::make_stmts_unattached({});
        } else {
            ir_copier_impl_t::view(v);
        }
    }
};

static void add_def_comments(const stmt &def_node, graph_tensor *t) {
    std::stringstream ss;
    t->details_.to_string(ss);
    def_node->attr()["comments"] = std::vector<std::string> {ss.str()};
}

enum op_kinds : int {
    kother = 0,
    kinput,
    koutput,
    kconstant,
    kreorder,
    kreshape,
};

struct general_lower_params_t {
    ir_module_ptr ret_mod;
    std::unordered_map<graph_tensor_ptr, tsr_info_t> &ltsr_rtsr;
    sc_graph_t &graph;
    stmts func_body;
    stmts init_body;
    int &tensor_counter;
    int &global_tensor_counter;
    bool is_graph_dynamic;
};

expr get_or_create_tensor(general_lower_params_t &gp, const graph_tensor_ptr &t,
        bool is_arg, int const_type,
        info_etype_t type = info_etype_t::real_tensor) {
    bool tsr_is_dynamic = t->details_.is_dynamic();
    sc_op *linked_output = nullptr;
    if (!is_arg) {
        for (auto &use : t->uses_) {
            // finds if any of the use of the tensor is marked output
            if (use.second->isa<output_op>()) {
                is_arg = true;
                linked_output = use.second.get();
                break;
            }
        }
    }
    if (is_arg || const_type != const_kind::not_const) {
        // input/output and const tsr don't need placeholder
        if (gp.is_graph_dynamic) {
            COMPILE_ASSERT(t->details_.get_format_candidates().size() <= 1,
                    "Input/output/constant tsr should have only empty or "
                    "one format candidate");
        }
        if (type == info_etype_t::placeholder) {
            type = info_etype_t::real_tensor;
        }
    }
    auto itr = gp.ltsr_rtsr.find(t);
    if (itr == gp.ltsr_rtsr.end()) {
        gp.ltsr_rtsr[t] = tsr_info_t();
        itr = gp.ltsr_rtsr.find(t);
        itr->second.count_ = gp.tensor_counter++;
    } else {
        if (type == info_etype_t::real_tensor
                && itr->second.tensor_.defined()) {
            if (gp.is_graph_dynamic) {
                itr->second.tensor_->attr().set(attr_keys::always_trans, true);
            }
            return itr->second.tensor_;
        }
        if (type == info_etype_t::placeholder
                && itr->second.placeholder_.defined()) {
            return itr->second.placeholder_;
        }
        if (type == info_etype_t::format && itr->second.format_.defined()) {
            return itr->second.format_;
        }
        if (type == info_etype_t::out_size && itr->second.size_.defined()) {
            return itr->second.size_;
        }
    }

    std::vector<expr> dims, strides;
    sc_data_type_t tsr_dtype;
    expr tsr;

    std::string tensor_name = graph::get_tensor_name(t.get(), linked_output);
    if (tensor_name.empty()) {
        tensor_name
                = std::string("buffer_") + std::to_string(itr->second.count_);
    }
    if (type == info_etype_t::real_tensor) {
        bool multi_candidates = gp.is_graph_dynamic
                && t->details_.get_format_candidates().size() > 1;
        expr dyn_tsr_size;
        if (multi_candidates) {
            assert(itr->second.size_.defined());
            dyn_tsr_size = builder::make_indexing(itr->second.size_, {0});
            dyn_tsr_size->attr().set(attr_keys::no_index2var, true);
        }

        dims = multi_candidates ? std::vector<expr> {dyn_tsr_size}
                                : t->details_.get_blocking_dims_expr(gp.graph);
        strides = multi_candidates ? std::vector<expr> {UINT64_C(1)}
                                   : t->details_.get_strides_expr(gp.graph);
        tsr_dtype = t->details_.dtype_;
        tsr = builder::make_stensor(tensor_name, dims, strides, tsr_dtype);
        tsr->attr()[attr_keys::plain_dims]
                = gp.graph.dims_to_expr(t->details_.get_plain_dims());
        if (itr->second.placeholder_.defined()) {
            // for dynamic tensor transform
            tsr->attr()["temp.dyn_placeholder"] = itr->second.placeholder_;
        }
        itr->second.tensor_ = tsr;
        if (is_arg || const_type != const_kind::not_const) {
            itr->second.placeholder_ = tsr;
        }
        if (gp.is_graph_dynamic) {
            tsr->attr().set(attr_keys::always_trans, true);
        }
    } else if (type == info_etype_t::placeholder) {
        if (itr->second.tensor_.defined()) {
            // first check if the real tensor exist
            tsr = itr->second.tensor_;
            itr->second.placeholder_ = tsr;
            tsr->attr().set(attr_keys::always_trans, true);
        } else {
            tensor_name += "_placeholder";
            dims = std::vector<expr> {sizeof(runtime::dynamic_tensor_t)};
            tsr_dtype = datatypes::u8;
            tsr = builder::make_tensor(tensor_name, dims, tsr_dtype);
            itr->second.placeholder_ = tsr;
        }
    } else if (type == info_etype_t::format) {
        tensor_name += "_format";
        if (t->details_.get_format_candidates().size() <= 1) {
            std::vector<uint64_t> init_format
                    = {uint64_t(t->details_.get_format().to_runtime())};
            tsr = builder::make_tensor(tensor_name, {UINT64_C(1)},
                    datatypes::index, address_space::automatic,
                    std::make_shared<static_data_t>(init_format));
        } else {
            tsr = builder::make_tensor(
                    tensor_name, {UINT64_C(1)}, datatypes::index);
        }
        itr->second.format_ = tsr;
    } else {
        assert(type == info_etype_t::out_size);
        tensor_name += "_size";
        tsr = builder::make_tensor(
                tensor_name, {UINT64_C(1)}, datatypes::index);
        itr->second.size_ = tsr;
    }
    if (type == info_etype_t ::real_tensor) {
        stmt def_node;
        if (!is_arg) {
            if (const_type != const_kind::not_const) {
                if (const_type == const_kind::global_const) {
                    auto folded_name = "folded_const_"
                            + std::to_string(gp.global_tensor_counter++);
                    tsr = copy_attr(*tsr,
                            gp.ret_mod->make_global_stensor(
                                    tsr.checked_as<tensor>()->elem_dtype_,
                                    folded_name,
                                    tsr.checked_as<tensor>()->dims_,
                                    tsr.checked_as<tensor>()->strides_,
                                    linkage::private_global, &def_node));
                    // global tensor does not need cache.
                    tsr->attr_->set("temp.dyn_placeholder", expr());
                    if (auto const_node
                            = t->producer_owner_->dyn_cast<constant_op_t>()) {
                        auto const_value = const_node->get_constant_values();
                        tsr.checked_as<tensor>()->init_value_ = const_value;
                    }
                    if (gp.is_graph_dynamic) {
                        tsr->attr().set(attr_keys::always_trans, true);
                    }
                    itr->second.tensor_ = tsr;
                    itr->second.placeholder_ = tsr;
                } else {
                    def_node = builder::make_var_tensor_def_unattached(tsr);
                    gp.init_body->seq_.emplace_back(def_node);
                }
            } else {
                def_node = builder::make_var_tensor_def_unattached(tsr);
                gp.func_body->seq_.emplace_back(def_node);
            }
        }
        if (def_node.defined()) { add_def_comments(def_node, t.get()); }
    } else if (type == info_etype_t::placeholder) {
        // placeholder
        // if use tensor as plhd, do nothing.
        if (!itr->second.tensor_.defined()) {
            gp.func_body->seq_.emplace_back(
                    builder::make_var_tensor_def_unattached(tsr));
            std::string name;
            if (tsr.isa<tensor>()) {
                name = tsr.checked_as<tensor>()->name_;
            } else {
                assert(tsr.isa<tensorptr>());
                name = tsr.checked_as<tensorptr>()
                                ->base_->ptr_.checked_as<tensor>()
                                ->name_
                        + "_tptr";
            }
            auto shape_tsr = builder::make_tensor(
                    std::string("dyn_shape_") + tsr.checked_as<tensor>()->name_,
                    {t->details_.get_plain_dims().size()}, datatypes::index);
            shape_tsr->attr().set(attr_keys::no_dead_write, true);
            shape_tsr->attr().set(attr_keys::no_tensor2var, true);
            tsr->attr().set("temp.dyn_shape_of_placeholder", shape_tsr);
            gp.func_body->seq_.emplace_back(
                    builder::make_var_tensor_def_unattached(shape_tsr));
            gp.func_body->seq_.emplace_back(builder::make_evaluate_unattached(
                    builder::make_write_struct(tsr, shape_tsr,
                            dyn_tsr_struct_t::name,
                            dyn_tsr_struct_t::fields::dim_ptr)));
            gp.func_body->seq_.emplace_back(builder::make_evaluate_unattached(
                    builder::make_write_struct(tsr,
                            builder::make_constant(
                                    {t->details_.get_plain_dims().size()},
                                    datatypes::s32),
                            dyn_tsr_struct_t::name,
                            dyn_tsr_struct_t::fields::ndims)));
            uint64_t etype = t->details_.dtype_.is_etype_pointer()
                    ? t->details_.dtype_.get_pointer_element().as_etype_int()
                    : t->details_.dtype_.as_etype_int();
            gp.func_body->seq_.emplace_back(builder::make_evaluate_unattached(
                    builder::make_write_struct(tsr,
                            builder::make_constant({etype}, datatypes::u32),
                            dyn_tsr_struct_t::name,
                            dyn_tsr_struct_t::fields::dtype)));
            auto plain_shapes
                    = gp.graph.dims_to_expr(t->details_.get_plain_dims());
            uint64_t dyn_mask_int = 0;
            for (size_t i = 0; i < plain_shapes.size(); i++) {
                gp.func_body->seq_.emplace_back(builder::make_assign_unattached(
                        builder::make_indexing(shape_tsr, {i}),
                        plain_shapes[i]));
                dyn_mask_int |= ((!plain_shapes[i].isa<constant>()) << i);
            }
            gp.func_body->seq_.emplace_back(builder::make_evaluate_unattached(
                    builder::make_write_struct(tsr,
                            builder::make_constant(
                                    {dyn_mask_int}, datatypes::u8),
                            dyn_tsr_struct_t::name,
                            dyn_tsr_struct_t::fields::dyn_mask)));
        }
    } else if (type == info_etype_t::format) {
        // placeholder can be replaced by tensor while format can't
        if (tsr.checked_as<tensor>()->init_value_) {
            gp.ret_mod->add_global_var(builder::make_var_tensor_def_unattached(
                    tsr, linkage::private_global)
                                               .checked_as<define>());
        } else {
            gp.func_body->seq_.emplace_back(
                    builder::make_var_tensor_def_unattached(tsr));
            gp.func_body->seq_.emplace_back(builder::make_assign_unattached(
                    builder::make_indexing(tsr, {0}), UINT64_C(0)));
        }
    } else if (type == info_etype_t::out_size) {
        if (const_type == const_kind::not_const) {
            gp.func_body->seq_.emplace_back(
                    builder::make_var_tensor_def_unattached(tsr));
            gp.func_body->seq_.back()->attr().set(
                    attr_keys::tsr_dont_buf_sched, true);
        }
    }
    return tsr;
};

expr create_op_query_func(const context_ptr &ctx, general_lower_params_t &gp,
        std::vector<expr> &op_dispatch_kernel, const sc_op_ptr &node,
        op_kinds kind) {
    std::vector<expr> plhd_ins, fmt_ins;
    std::vector<expr> plhd_outs, fmt_outs, size_outs;
    bool need_dispatch = node->get_dispatch_key_set()->set_.size() > 1;
    // current input
    for (auto &ltensor : node->get_inputs()) {
        auto const_type = ltensor->producer_owner_->attrs_.get_or_else(
                "constant", const_kind::not_const);
        plhd_ins.emplace_back(get_or_create_tensor(
                gp, ltensor, false, const_type, info_etype_t::placeholder));
        fmt_ins.emplace_back(get_or_create_tensor(
                gp, ltensor, false, const_type, info_etype_t::format));
    }
    // input before reorder
    if (node->isa<tunable_op_t>()
            || (node->isa<fused_op_t>()
                    && !node->stc_cast<fused_op_t>()->main_op_.empty())) {
        auto &inputs = node->get_inputs();
        auto query_sz = inputs.size();
        if (node->isa<fused_op_t>()) {
            query_sz = node->stc_cast<fused_op_t>()
                               ->main_op_.ops_[1]
                               ->get_inputs()
                               .size();
        }
        for (size_t i = 0; i < query_sz; i++) {
            auto ltensor = node->get_inputs()[i];
            auto node_before = ltensor->producer_owner_;
            auto const_type_before = node_before->attrs_.get_or_else(
                    "constant", const_kind::not_const);
            // find the buffer before reorder.
            if (node_before->isa<reorder_op_t>()
                    && (node_before->attrs_.as_map().empty()
                            || node_before->attrs_.get_or_else(
                                       "constant", const_kind::not_const)
                                    == const_kind::not_const)) {
                ltensor = node_before->get_inputs()[0];
            }
            plhd_ins.emplace_back(get_or_create_tensor(gp, ltensor, false,
                    const_type_before, info_etype_t::placeholder));
            fmt_ins.emplace_back(get_or_create_tensor(gp, ltensor, false,
                    const_type_before, info_etype_t::format));
        }
    }
    auto const_type
            = node->attrs_.get_or_else("constant", const_kind::not_const);
    for (auto &ltensor : node->get_outputs()) {
        expr plhd, fmt, size;
        if (kind == kinput) {
            // use real tensor instead of placeholder.
            plhd = get_or_create_tensor(
                    gp, ltensor, true, const_type, info_etype_t::real_tensor);
            fmt = get_or_create_tensor(
                    gp, ltensor, true, const_type, info_etype_t::format);
        } else if (kind == kconstant) {
            plhd = get_or_create_tensor(gp, ltensor, false,
                    const_kind::global_const, info_etype_t::real_tensor);
            fmt = get_or_create_tensor(gp, ltensor, false,
                    const_kind::global_const, info_etype_t::format);
        } else {
            plhd = get_or_create_tensor(
                    gp, ltensor, false, const_type, info_etype_t::placeholder);
            // expect for output tsr
            if (!plhd.defined()) {
                plhd = get_or_create_tensor(gp, ltensor, false, const_type,
                        info_etype_t::real_tensor);
            }
            fmt = get_or_create_tensor(
                    gp, ltensor, false, const_type, info_etype_t::format);
            size = get_or_create_tensor(
                    gp, ltensor, false, const_type, info_etype_t::out_size);
        }
        plhd_outs.emplace_back(plhd);
        fmt_outs.emplace_back(fmt);
        size_outs.emplace_back(size);
    }
    // Pruning, because the format propagation is broken after reorder,
    // so it doesn't need query to deliver formats. Notes that only
    // reorder could, other ops should propagate their format even does
    // not need dispatch.
    if ((node->isa<reorder_op_t>() && !need_dispatch)
            || const_type != const_kind::not_const) {
        return expr();
    }
    expr dyn_ker_ptr;
    // update dynamic query format
    if (!op_dispatch_kernel[node->logical_op_id_].defined()) {
        if (!utils::is_one_of(kind, kinput, koutput, kconstant)) {
            auto &table_map = gp.ret_mod->get_op_table_map();
            auto func_name = node->op_name_ + "__"
                    + std::to_string(node->logical_op_id_) + "_ptr";
            auto table_name = func_name + "_table";
            auto table_it = table_map.find(table_name);
            auto table_var = builder::make_var(datatypes::pointer, table_name);
            auto table_ptr = table_it != table_map.end()
                    ? table_it->second
                    : std::make_shared<op_dispatch_tables_t>();
            dyn_ker_ptr = builder::make_tensor(
                    func_name, {UINT64_C(1)}, datatypes::pointer);
            std::vector<expr> query_func_args;
            query_func_args.emplace_back(table_var);
            query_func_args.insert(
                    query_func_args.end(), plhd_outs.begin(), plhd_outs.end());
            query_func_args.insert(
                    query_func_args.end(), plhd_ins.begin(), plhd_ins.end());
            query_func_args.insert(
                    query_func_args.end(), fmt_outs.begin(), fmt_outs.end());
            query_func_args.insert(
                    query_func_args.end(), fmt_ins.begin(), fmt_ins.end());
            query_func_args.insert(
                    query_func_args.end(), size_outs.begin(), size_outs.end());
            query_func_args.push_back(dyn_ker_ptr);
            expr query_call; // call node
            if (node->isa<fused_op_t>()) {
                auto fused_node = node->stc_cast<fused_op_t>();
                auto query_mod = fused_node->get_dynamic_query_func(ctx);
                query_func_args[0] = fused_node->main_table_var_;
                gp.ret_mod->merge(*query_mod);
                assert(table_ptr);
                query_call = builder::make_call(
                        query_mod->get_entry_func(), query_func_args);

            } else {
                auto table_ptr = std::make_shared<op_dispatch_tables_t>();
                gp.ret_mod->add_op_table(std::make_pair(table_name, table_ptr));
                initialize_format_table_with_op(node, table_ptr);
                query_call
                        = call_op_dynamic_query_function(node, query_func_args);
            }
            stmts_node_t *target_body = gp.func_body.get();
            if (table_it == table_map.end()) {
                auto table_def = builder::make_var_tensor_def_unattached(
                        table_var, linkage::private_global);
                gp.ret_mod->add_global_var(table_def.checked_as<define>());
            }
            target_body->seq_.emplace_back(
                    builder::make_var_tensor_def_unattached(dyn_ker_ptr));
            target_body->seq_.emplace_back(
                    builder::make_evaluate_unattached(query_call));
            op_dispatch_kernel[node->logical_op_id_]
                    = builder::make_indexing(dyn_ker_ptr, 0);
            op_dispatch_kernel[node->logical_op_id_]->attr().set(
                    attr_keys::no_index2var, true);
            op_dispatch_kernel[node->logical_op_id_]->attr().set(
                    attr_keys::always_trans, true);
        }
    }
    return dyn_ker_ptr;
}

static void set_op_dispatch_key(
        const sc_op_ptr &node, const op_dispatch_key_t &key) {
    if (node->isa<fused_op_t>()) {
        node->stc_cast<fused_op_t>()->update_internal_graph_format(key);
    } else {
        if (auto tunable_node = node->dyn_cast<tunable_op_t>()) {
            tunable_node->set_config_by_key(key);
        }
        auto &inputs = node->get_inputs();
        auto &outputs = node->get_outputs();
        int idx = 0;
        for (auto &in : inputs) {
            in->details_.set_format(key.in_out_formats_[idx++]);
        }
        for (auto &out : outputs) {
            out->details_.set_format(key.in_out_formats_[idx++]);
        }
    }
    node->info_.cur_impl_ = key.impl_;
}

std::pair<expr, expr> get_reshape_tptr(general_lower_params_t &gp,
        const graph_tensor_ptr &old_tsr, const graph_tensor_ptr &new_tsr,
        int const_type, op_kinds kind) {
    auto base_tsr
            = get_or_create_tensor(gp, old_tsr, kind == kinput, const_type);
    size_t ndims;
    if (base_tsr.isa<tensorptr>()) {
        ndims = base_tsr.static_as<tensorptr>()->shape_.size();
    } else {
        assert(base_tsr.isa<tensor>());
        ndims = base_tsr.static_as<tensor>()->dims_.size();
    }
    std::vector<expr_c> base_idx(ndims, expr(0));
    std::vector<expr> new_shape_tmp
            = new_tsr->details_.get_blocking_dims_expr(gp.graph);
    std::vector<expr_c> new_shape(new_shape_tmp.begin(), new_shape_tmp.end());
    auto new_tptr = builder::tensor_ptr(base_tsr, base_idx, new_shape);
    new_tptr->attr().set(attr_keys::plain_dims,
            gp.graph.dims_to_expr(new_tsr->details_.get_plain_dims()));
    return std::make_pair(base_tsr, new_tptr);
}

void create_op_tensors(general_lower_params_t &gp, std::vector<expr> &ins,
        std::vector<expr> &outs, const sc_op_ptr &node, op_kinds kind) {
    int const_type
            = node->attrs_.get_or_else("constant", const_kind::not_const);
    for (auto &ltensor : node->get_inputs()) {
        // As the traversal is not in order, so the constant type of
        // tensor should be decided by the node before.
        ins.emplace_back(get_or_create_tensor(gp, ltensor, false, const_type));
    }
    for (auto &ltensor : node->get_outputs()) {
        if (kind == kconstant) {
            get_or_create_tensor(gp, ltensor, false, const_kind::global_const);
        } else if (kind == kreshape) {
            COMPILE_ASSERT(node->get_inputs().size() == 1,
                    "Reshape should have 1 input");
            // If the output of tensor view is output of graph
            if (gp.ltsr_rtsr.find(ltensor) != gp.ltsr_rtsr.end()
                    && has_output_uses(ltensor)) {
                break;
            }
            auto out_tsr_pair = get_reshape_tptr(
                    gp, node->get_inputs()[0], ltensor, const_type, kind);
            auto it = gp.ltsr_rtsr.find(ltensor);
            if (it != gp.ltsr_rtsr.end() && it->second.tensor_.defined()) {
                COMPILE_ASSERT(gp.is_graph_dynamic,
                        "If output tsr of tensor view is defined, it "
                        "should in dynamic mode.");
                // the tsr replace map for tensor view op. Because in
                // dynamic mode, the output of tensor view may be
                // traversed first.
                std::unordered_map<expr_c, expr> tv_replace_map;
                tv_replace_map.insert(std::make_pair(
                        it->second.tensor_, out_tsr_pair.second));
                tv_tsr_replacer_t cpy(tv_replace_map, false);
                gp.func_body = cpy.dispatch(gp.func_body)
                                       .remove_const()
                                       .checked_as<stmts>();
                gp.init_body = cpy.dispatch(gp.init_body)
                                       .remove_const()
                                       .checked_as<stmts>();
            }
            gp.ltsr_rtsr[ltensor].tensor_ = out_tsr_pair.second;
        } else {
            graph_tensor_ptr out_tsr;
            // for pattern like node->reshape->output
            if (auto out_tsr = get_linked_output_tsr(ltensor)) {
                gp.ltsr_rtsr[ltensor].tensor_ = get_reshape_tptr(
                        gp, out_tsr, ltensor, const_type, kind)
                                                        .second;
                outs.emplace_back(gp.ltsr_rtsr[ltensor].tensor_);
            } else {
                outs.emplace_back(get_or_create_tensor(
                        gp, ltensor, kind == kinput, const_type));
            }
        }
    }
}

static std::string get_dispatch_callee_name(const expr &kernel) {
    assert(kernel.isa<indexing>());
    return kernel.checked_as<indexing>()->ptr_.checked_as<tensor>()->name_;
}

ir_module_ptr lower_graph(context_ptr ctx, sc_graph_t &graph,
        const std::vector<sc_op_ptr> &args) {
    auto timer = SC_SCOPED_TIMER_INFO("graph.driver.time.lowering", "");
    if (!ctx->flags_.dump_graph_.empty()) {
        SC_INFO << "visualize graph to a dot file and a json file";
        visualize(ctx->flags_.dump_graph_, graph);
    }
    result_dump_config_t dump_config {ctx->flags_.graph_dump_results_};
    lowering_visitor_state_t visiter_state(graph);
    op_visitor_t vis {
            visiter_state.get_selector(), visiter_state.get_updater()};
    visiter_state.input_op_itr = vis.to_visit_.end();
    std::vector<expr> params;
    stmts func_body = make_stmt<stmts_node_t>(std::vector<stmt>());
    stmts init_body = make_stmt<stmts_node_t>(std::vector<stmt>());
    // todo: use graph-id to generate name
    auto func = builder::make_func(
            graph.attrs_.get_or_else<std::string>("temp.name", "main_entry"),
            params, func_body, datatypes::void_t);
    // todo: logical tensor should also have an unique id
    // tsr_info_t include dynamic placeholder(dynamic tensor with empty
    // datapointer) and runtime format.
    std::unordered_map<graph_tensor_ptr, tsr_info_t> ltsr_rtsr;
    // function pointer
    std::vector<expr> op_dispatch_kernel(graph.ops_.size());
    int tensor_counter = 0;
    int global_tensor_counter = 0;
    auto ret_mod = ir_module_t::from_entry_func(ctx, func);

    expr dump_out_path;
    int global_str_counter = 0;
    if (dump_config.enabled_) {
        dump_out_path = make_global_string(
                ret_mod, dump_config.path_, global_str_counter);
    }

    if (graph.attrs_.get_or_else("folded_input", false)) {
        ret_mod->attr_.set("folded_input", true);
    }
    bool is_graph_dynamic = graph.is_dynamic();
    general_lower_params_t gp {ret_mod, ltsr_rtsr, graph, func_body, init_body,
            tensor_counter, global_tensor_counter, is_graph_dynamic};
    vis.visit_graph(graph, [&](const sc_op_ptr &node) {
        std::vector<expr> ins, outs;
        // special kinds of Ops that we need to take care of
        op_kinds kind = kother;
        if (node->isa<input_op>()) {
            kind = kinput;
        } else if (node->isa<output_op>()) {
            kind = koutput;
        } else if (node->isa<constant_op_t>()) {
            kind = kconstant;
            if (node->attrs_.get_or_else("constant", const_kind::not_const)
                    == const_kind::not_const) {
                node->attrs_.set("constant", const_kind::global_const);
            }
        } else if (node->isa<reorder_op_t>()) {
            // todo: assume reorder is fused break in dynamic now.
            kind = kreorder;
        } else if (node->isa<tensor_view_op_t>()) {
            kind = kreshape;
        }
        // fixme(jingze): Use jit instead of runtime op.
        auto update_runtime_op_args = [&](const sc_op_ptr &node,
                                              std::vector<expr> &exprargs) {
            std::vector<graph_tensor_ptr> ins, outs;
            auto extra_infos
                    = node->stc_cast<runtime_op_t>()->get_extra_lower_infos(
                            graph, ret_mod);
            ins = extra_infos.in_ltsrs_;
            outs = extra_infos.out_ltsrs_;
            exprargs.clear();
            int const_type = node->attrs_.get_or_else(
                    "constant", const_kind::not_const);
            for (auto &out : outs) {
                exprargs.emplace_back(get_or_create_tensor(
                        gp, out, false, const_type, info_etype_t::real_tensor));
            }
            for (auto &in : ins) {
                exprargs.emplace_back(get_or_create_tensor(
                        gp, in, false, const_type, info_etype_t::real_tensor));
            }
            for (auto &out : outs) {
                exprargs.emplace_back(get_or_create_tensor(
                        gp, out, false, const_type, info_etype_t::format));
            }
            for (auto &in : ins) {
                exprargs.emplace_back(get_or_create_tensor(
                        gp, in, false, const_type, info_etype_t::format));
            }
            exprargs.insert(exprargs.end(), extra_infos.attrs_.begin(),
                    extra_infos.attrs_.end());
        };

        // if the node is reorder, query its uses op first.
        if (is_graph_dynamic && node->get_dispatch_key_set()->set_.size() > 1) {
            if (kind == kreorder
                    && node->attrs_.get_or_else(
                               "constant", const_kind::not_const)
                            == const_kind::not_const
                    && need_query_next_first(node)) {
                auto query_node
                        = node->get_outputs()[0]->uses_[0].second.lock();
                create_op_query_func(
                        ctx, gp, op_dispatch_kernel, query_node, kind);
            }
            create_op_query_func(ctx, gp, op_dispatch_kernel, node, kind);
        }
        // tensor decl should put after query functions.
        create_op_tensors(gp, ins, outs, node, kind);
        if (is_graph_dynamic && kind == kreorder
                && node->attrs_.get_or_else("constant", const_kind::not_const)
                        == const_kind::not_const) {
            outs[0]->attr().set("temp.may_inplace", true);
        }
        int const_type
                = node->attrs_.get_or_else("constant", const_kind::not_const);
        switch (kind) {
            case kinput: {
                for (auto &v : outs) {
                    params.emplace_back(v);
                }
                break;
            }
            case koutput: {
                for (auto &v : ins) {
                    params.emplace_back(v);
                }
                break;
            }
            case kconstant:
            case kreshape: {
                break;
                // nothing to do.
            }
            default: {
                std::vector<expr> exprargs;
                exprargs.insert(exprargs.end(), outs.begin(), outs.end());
                exprargs.insert(exprargs.end(), ins.begin(), ins.end());
                if (node->isa<runtime_op_t>()) {
                    exprargs.clear();
                    update_runtime_op_args(node, exprargs);
                }
                expr kernel_call;
                std::string callee_name;
                bool need_dispatch
                        = node->get_dispatch_key_set()->set_.size() > 1;
                if (need_dispatch) {
                    assert(is_graph_dynamic);
                    assert(op_dispatch_kernel[node->logical_op_id_].defined());
                    callee_name = get_dispatch_callee_name(
                            op_dispatch_kernel[node->logical_op_id_]);
                    std::string table_name = callee_name + "_table";
                    auto &key_set = node->get_dispatch_key_set()->set_;
                    int dyn_idx = 0;
                    for (auto &key : key_set) {
                        set_op_dispatch_key(node, key);
                        // todo: add padding support
                        auto mod = node->get_func(ctx);
                        auto func = mod->get_entry_func();
                        func->attr().set(attr_keys::always_trans, true);
                        func->name_ += "_" + std::to_string(dyn_idx);
                        func->decl_->name_ = func->name_;
                        ret_mod->merge(*mod);
                        if (!dyn_idx) {
                            // mark the first function as prototype.
                            op_dispatch_kernel[node->logical_op_id_]
                                    ->attr()
                                    .set("prototype", mod->get_entry_func());
                        }
                        auto cur_table
                                = ret_mod->get_op_table_map()[table_name];
                        assert(cur_table);
                        add_dispatch_symbol_to_kernel_table(
                                cur_table, key, func->name_);
                        dyn_idx++;
                    }
                    kernel_call = make_expr<call_node>(
                            op_dispatch_kernel[node->logical_op_id_], exprargs);
                } else {
                    // no dispatch
                    auto mod = node->get_func(ctx);
                    ret_mod->merge(*mod);
                    auto callee = mod->get_entry_func();
                    if (!callee) {
                        // runtime op
                        assert(mod->attr_.has_key("temp.runtime_func"));
                        callee = mod->attr_.get<func_t>("temp.runtime_func");
                    } else if (is_graph_dynamic) {
                        callee->attr().set(attr_keys::always_trans, true);
                        callee->decl_->attr().set(
                                attr_keys::always_trans, true);
                    }
                    callee_name = callee->name_;
                    kernel_call = builder::make_call(callee, exprargs);
                }
                stmts_node_t *target_body
                        = (const_type != const_kind::not_const)
                        ? init_body.get()
                        : func_body.get();
                target_body->seq_.emplace_back(
                        builder::make_evaluate_unattached(kernel_call));
                if (ctx->flags_.value_check_) {
                    make_value_check_call(outs, ret_mod, callee_name,
                            global_str_counter, target_body);
                }
                if (dump_config.enabled_
                        && dump_config.should_function_dump(callee_name)) {
                    make_dump_tensor_call(outs, node, ret_mod, callee_name,
                            global_str_counter, dump_config, dump_out_path,
                            target_body);
                }
            }
        }
    });
    if (!args.empty()) {
        std::vector<expr> new_param;
        for (auto &v : args) {
            if (auto inop = v->dyn_cast<input_op>()) {
                for (auto &in : inop->get_outputs()) {
                    auto itr = ltsr_rtsr.find(in);
                    COMPILE_ASSERT(itr != ltsr_rtsr.end(),
                            "Cannot find the input op in the generated "
                            "function");
                    new_param.emplace_back(itr->second.tensor_);
                }
            } else if (auto outop = v->dyn_cast<output_op>()) {
                for (auto &out : outop->get_inputs()) {
                    auto itr = ltsr_rtsr.find(out);
                    COMPILE_ASSERT(itr != ltsr_rtsr.end(),
                            "Cannot find the output op in the generated "
                            "function");
                    new_param.emplace_back(itr->second.tensor_);
                }
            } else {
                COMPILE_ASSERT(false,
                        "The Op given in the args is not input or output");
            }
        }
        COMPILE_ASSERT(new_param.size() == params.size(),
                "The args count does not match the count of in/out "
                "tensors");
        params = std::move(new_param);
    }
    if (!init_body->seq_.empty()) {
        expr is_init_var = ret_mod->make_global_var(datatypes::boolean,
                "is_init", linkage::private_global,
                graph.attrs_.get_or_else("folded_input", false));
        init_body->seq_.emplace_back(
                builder::make_assign_unattached(is_init_var, true));
        sc::func_t init_func = builder::make_func(
                "__init_const_globals", params, init_body, datatypes::void_t);
        init_func->attr()[function_attrs::private_] = true;
        ret_mod->add_func({init_func});
        stmt const_init = builder::make_if_else_unattached(
                builder::make_logic_not(is_init_var),
                builder::make_stmts_unattached(
                        {builder::make_evaluate_unattached(
                                builder::make_call(init_func, params))}),
                stmts());
        func_body->seq_.insert(func_body->seq_.begin(), const_init);
    }
    func->params_ = std::move(params);
    func->decl_->params_ = func->params_;
    func->body_ = std::move(func_body);
    if (utils::compiler_configs_t::get().print_pass_result_) {
        SC_MODULE_INFO << ret_mod;
    }
    ret_mod->attr_[ir_module_t::attr_key_t::GFLOP]
            = graph.attrs_.get_or_else(sc_graph_t::attr_key_t::gflop, 0.0f);
    return ret_mod;
}
} // namespace sc
