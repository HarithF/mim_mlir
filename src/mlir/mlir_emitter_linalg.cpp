
#include <format>
#include <functional>
#include <map>
#include <set>

#include <mim/lam.h>
#include <mim/tuple.h>

#include <mim/plug/mem/mem.h>
#include <mim/plug/tensor/autogen.h>
#include <mim/plug/tensor/tensor.h>

#include "mlir/mlir_emitter.h"

namespace mim::mlir_be {

std::optional<MLIRValue> MLIREmitter::try_emit_tensor_op(const App* app, MLIRBlock& into) {
    auto* def = app;

    if (Axm::isa<plug::tensor::map_reduce>(app) || Axm::isa<plug::tensor::map_reduce_ds>(app)) {
        emit_linalg_generic(app, into);
        return values_[def];
    }

    if (Axm::isa<plug::tensor::map_reduce_aff>(app)) {
        assert(false && "map_reduce_aff not yet supported");
        return MLIRValue{};
    }

    if (auto bc = Axm::isa<plug::tensor::broadcast>(app)) {
        // mirrors lower_broadcast extraction exactly
        auto [s_in, s_out, input] = bc->arg()->projs<3>();
        auto callee               = bc->callee()->as<App>();
        auto [T, r]               = callee->args<2>();

        auto r_lit = Lit::isa(r);
        assert(r_lit && "broadcast rank must be literal");
        auto r_nat = *r_lit;

        // derive broadcast dimensions: where s_in dim == 1
        std::vector<int64_t> bcast_dims;
        for (size_t i = 0; i < r_nat; ++i) {
            auto dim_in = s_in->proj(r_nat, i);
            if (auto lit = Lit::isa(dim_in); lit && *lit == 1) bcast_dims.push_back(static_cast<int64_t>(i));
        }

        auto in_val   = get_or_emit(input, into);
        auto out_type = types_.convert(def->type());

        if (!std::holds_alternative<MLIRTensorType>(in_val.type)) in_val = wrap_as_tensor(input, in_val, into);

        // tensor.empty for output buffer
        std::string buf_name = fresh_name(def) + ".buf";
        MLIRValue out_buf{buf_name, out_type};
        into.ops.emplace_back(std::make_unique<TensorEmptyOp>(out_buf));

        MLIRValue result{fresh_name(def), out_type};
        into.ops.emplace_back(std::make_unique<LinalgBroadcastOp>(result, in_val, out_buf, std::move(bcast_dims)));
        return result;
    }

    return std::nullopt;
}

void MLIREmitter::emit_linalg_generic(const App* app, MLIRBlock& into) {
    // Unpack currying chain:
    //   app0->arg = inputs pack
    //   app1->arg = subscripts
    //   app2->arg = (comb, zero)
    //   app3->arg = (Tis, Ris, Sis)   [skipped]
    //   app4->arg = S (result shape)
    //   app5->arg = (T, rank)
    auto* app0 = app;
    auto* app1 = app0->callee()->as<App>();
    auto* app2 = app1->callee()->as<App>();
    auto* app3 = app2->callee()->as<App>();
    auto* app4 = app3->callee()->as<App>(); // skip app3
    auto* app5 = app4->callee()->as<App>();
    auto* app6 = app5->callee()->as<App>();

    auto* inputs_pack    = app0->arg();
    auto* subs           = app1->arg();
    auto [comb, zero]    = app2->arg()->projs<2>();
    auto [Tis, Ris, Sis] = app3->arg()->projs<3>();

    auto* S        = app4->arg();
    auto [T, rank] = app5->arg()->projs<2>();
    auto* nis      = app6->arg();

    auto n_inputs_opt = Lit::isa(nis);
    assert(n_inputs_opt && "nis must be a literal");
    size_t n_inputs = *n_inputs_opt;

    auto proj_input
        = [&](size_t i) -> const Def* { return n_inputs == 1 ? inputs_pack : inputs_pack->proj(n_inputs, i); };
    auto proj_sub = [&](size_t i) -> const Def* { return n_inputs == 1 ? subs : subs->proj(n_inputs, i); };

    auto* body_lam = comb->isa_mut<Lam>();
    assert(body_lam && "comb must be a mutable Lam");

    auto res_rank_opt = Lit::isa(rank);
    assert(res_rank_opt);
    size_t res_rank = *res_rank_opt;

    std::vector<std::optional<int64_t>> res_shape;
    for (size_t i = 0; i < res_rank; ++i) {
        auto dim = S->proj(res_rank, i);
        if (auto lit = Lit::isa(dim))
            res_shape.push_back(static_cast<int64_t>(*lit));
        else
            res_shape.push_back(std::nullopt);
    }
    auto res_elem_type = types_.convert(T);
    MLIRTensorType res_tensor;
    res_tensor.shape = res_shape;
    res_tensor.elem  = std::make_shared<MLIRTypeNode>(res_elem_type);
    MLIRType res_type{std::move(res_tensor)};

    std::vector<MLIRValue> ins;
    for (size_t i = 0; i < n_inputs; ++i)
        ins.push_back(get_or_emit(proj_input(i), into));

    // Output buffer
    std::string buf_name = fresh_name(app) + ".buf";
    MLIRValue out_buf{buf_name, res_type};
    into.ops.emplace_back(std::make_unique<TensorEmptyOp>(out_buf));

    std::vector<MLIRValue> outs{out_buf};

    auto get_sub_rank = [&](size_t i) -> size_t {
        auto ris_i = Ris->proj(n_inputs, i);
        auto lit   = Lit::isa(ris_i);
        assert(lit && "Ris must be literal");
        return static_cast<size_t>(*lit);
    };

    auto get_sub_dim = [&](const Def* sub_i, size_t rank, size_t j) -> const Def* {
        return rank == 1 ? sub_i : sub_i->proj(rank, j);
    };

    // Affine maps
    size_t total_dims = res_rank;
    for (size_t i = 0; i < n_inputs; ++i) {
        // auto sub_i = proj_sub(i);
        //  auto sub_rank_opt = Lit::isa(sub_i->type()->isa<Arr>()->arity());
        //  assert(sub_rank_opt);
        /*for (size_t j = 0; j < *sub_rank_opt; ++j) {
            auto idx = sub_i->proj(*sub_rank_opt, j);
            if (auto lit = Lit::isa(idx)) total_dims = std::max(total_dims, (size_t)(*lit + 1));
        }*/
        for (size_t i = 0; i < n_inputs; ++i) {
            auto sub_i      = proj_sub(i);
            size_t sub_rank = get_sub_rank(i);
            for (size_t j = 0; j < sub_rank; ++j) {
                auto idx = get_sub_dim(sub_i, sub_rank, j);
                if (auto lit = Lit::isa(idx)) total_dims = std::max(total_dims, (size_t)(*lit + 1));
            }
        }
    }

    auto make_map = [&](std::vector<size_t> dims) -> std::string {
        std::string a, b;
        for (size_t i = 0; i < total_dims; ++i)
            a += (i ? ", " : "") + std::format("d{}", i);
        for (size_t i = 0; i < dims.size(); ++i)
            b += (i ? ", " : "") + std::format("d{}", dims[i]);
        return std::format("affine_map<({}) -> ({})>", a, b);
    };

    std::vector<std::string> indexing_maps;

    for (size_t i = 0; i < n_inputs; ++i) {
        auto sub_i      = proj_sub(i);
        size_t sub_rank = get_sub_rank(i);
        std::vector<size_t> dims;
        for (size_t j = 0; j < sub_rank; ++j)
            dims.push_back((size_t)(*Lit::isa(get_sub_dim(sub_i, sub_rank, j))));
        indexing_maps.push_back(make_map(dims));
    }
    /*for (size_t i = 0; i < n_inputs; ++i) {
        auto sub_i        = proj_sub(i);
        auto sub_rank_opt = Lit::isa(sub_i->type()->isa<Arr>()->arity());
        std::vector<size_t> dims;
        for (size_t j = 0; j < *sub_rank_opt; ++j)
            dims.push_back((size_t)(*Lit::isa(sub_i->proj(*sub_rank_opt, j))));
        indexing_maps.push_back(make_map(dims));
    }*/
    std::vector<size_t> out_dims;
    for (size_t i = 0; i < res_rank; ++i)
        out_dims.push_back(i);
    indexing_maps.push_back(make_map(out_dims));

    // Iterator types
    std::set<size_t> out_dim_set(out_dims.begin(), out_dims.end());
    std::vector<std::string> iterator_types;
    for (size_t i = 0; i < total_dims; ++i)
        iterator_types.push_back(out_dim_set.count(i) ? "parallel" : "reduction");

    std::vector<MLIRValue> body_args;

    auto* body_var     = body_lam->var();
    auto body_var_type = body_var->type();

    std::vector<size_t> arg_path;
    const Def* arg_type = body_var_type;
    if (auto sigma = body_var_type->isa<Sigma>()) {
        for (size_t i = 0; i < sigma->num_ops(); ++i) {
            if (!sigma->op(i)->isa<Pi>() && !Axm::isa<plug::mem::M>(sigma->op(i))) {
                arg_path.push_back(i);
                arg_type = sigma->op(i);
                break;
            }
        }
    }

    std::map<std::vector<size_t>, MLIRValue> path_to_val;
    MLIRValue acc_val;
    bool have_acc = false;

    auto put = [&](std::vector<size_t> p, MLIRValue v) { path_to_val[std::move(p)] = std::move(v); };

    auto plan_ins = [&](const Def* ins_t, std::vector<size_t> ins_path) {
        if (auto s = ins_t->isa<Sigma>()) {
            for (size_t i = 0; i < s->num_ops(); ++i) {
                auto p = ins_path;
                p.push_back(i);
                MLIRValue v{fresh_name("%in_"), types_.convert(s->op(i))};
                body_args.push_back(v);
                put(std::move(p), v);
            }
        } else if (auto a = ins_t->isa<Arr>()) {
            if (auto n = Lit::isa(a->arity())) {
                for (size_t i = 0; i < *n; ++i) {
                    auto p = ins_path;
                    p.push_back(i);
                    MLIRValue v{fresh_name("%in_"), types_.convert(a->body())};
                    body_args.push_back(v);
                    put(std::move(p), v);
                }
            }
        } else if (!ins_t->isa<Pi>()) {
            MLIRValue v{fresh_name("%in_"), types_.convert(ins_t)};
            body_args.push_back(v);
            put(std::move(ins_path), v);
        }
    };

    if (auto s = arg_type->isa<Sigma>()) {
        // [acc, ins]
        auto acc_p = arg_path;
        acc_p.push_back(0);
        auto ins_p = arg_path;
        ins_p.push_back(1);
        plan_ins(s->op(1), std::move(ins_p));
        acc_val  = MLIRValue{fresh_name("%acc_"), res_elem_type};
        have_acc = true;
        put(std::move(acc_p), acc_val);
    } else if (auto a = arg_type->isa<Arr>()) {
        // «2; T» from a [T, «1; T»] singleton collapse: [acc, single_input]
        if (auto n = Lit::isa(a->arity()); n && *n == 2) {
            auto acc_p = arg_path;
            acc_p.push_back(0);
            auto ins_p = arg_path;
            ins_p.push_back(1);
            MLIRValue v{fresh_name("%in_"), types_.convert(a->body())};
            body_args.push_back(v);
            put(std::move(ins_p), v);
            acc_val  = MLIRValue{fresh_name("%acc_"), res_elem_type};
            have_acc = true;
            put(std::move(acc_p), acc_val);
        } else {
            acc_val  = MLIRValue{fresh_name("%acc_"), res_elem_type};
            have_acc = true;
            put(arg_path, acc_val);
        }
    } else {
        acc_val  = MLIRValue{fresh_name("%acc_"), res_elem_type};
        have_acc = true;
        put(arg_path, acc_val);
    }

    if (have_acc) body_args.push_back(acc_val);

    auto type_arity = [](const Def* t) -> size_t {
        if (auto s = t->isa<Sigma>()) return s->num_ops();
        if (auto a = t->isa<Arr>())
            if (auto n = Lit::isa(a->arity())) return *n;
        return 0;
    };
    auto nav_path = [&](const Def* root, const std::vector<size_t>& path) -> const Def* {
        const Def* cur = root;
        for (size_t i : path) {
            size_t arity = type_arity(cur->type());
            if (arity == 0) return nullptr;
            cur = cur->proj(arity, i);
        }
        return cur;
    };

    for (auto& [path, val] : path_to_val)
        if (auto d = nav_path(body_var, path)) values_[d] = val;

    DefSet body_visited;
    std::function<void(const Def*)> walk_body = [&](const Def* d) {
        if (!body_visited.insert(d).second) return;
        if (d->isa<Var>()) return;
        if (d->isa_mut<Lam>()) return;

        if (d->isa<Extract>()) {
            std::vector<size_t> path;
            const Def* cur = d;
            bool ok        = true;
            while (auto exi = cur->isa<Extract>()) {
                auto lit = Lit::isa(exi->index());
                if (!lit) {
                    ok = false;
                    break;
                }
                path.insert(path.begin(), static_cast<size_t>(*lit));
                cur = exi->tuple();
            }
            if (ok && cur == body_var) {
                if (auto it = path_to_val.find(path); it != path_to_val.end()) values_[d] = it->second;
            }
        }
        for (auto op : d->ops())
            walk_body(op);
    };
    walk_body(body_lam->body());

    DefSet pre_body_keys;
    for (auto& [d, _] : values_)
        pre_body_keys.insert(d);

    auto* op = new LinalgGenericOp(ins, outs, indexing_maps, iterator_types, body_args);
    emit_linalg_body(body_lam, op->body().entry());

    std::vector<const Def*> body_added;
    for (auto& [d, _] : values_)
        if (!pre_body_keys.contains(d)) body_added.push_back(d);
    for (auto* d : body_added)
        values_.erase(d);

    values_[app] = op->result();
    into.ops.emplace_back(op);
}

void MLIREmitter::emit_linalg_body(Lam* body_lam, MLIRBlock& body_bb) {
    assert(body_lam->is_set());
    auto* app = body_lam->body()->isa<App>();
    assert(app);
    auto* callee = app->callee();
    auto* arg    = app->arg();

    if (is_return_callee(callee, body_lam->ret_var())) {
        std::vector<MLIRValue> yield_vals;
        auto* arg = app->arg();
        if (!Axm::isa<plug::mem::M>(arg->type())) {
            auto v = get_or_emit(arg, body_bb);
            if (!v.empty()) yield_vals.push_back(v);
        }
        body_bb.ops.emplace_back(std::make_unique<LinalgYieldOp>(std::move(yield_vals)));
        return;
    }

    // Local continuation call
    if (auto* local_lam = callee->isa_mut<Lam>()) {
        if (local_lam->is_set() && !local_lam->is_external()) {
            // Seed the local lam's parameter(s)
            auto dom = local_lam->type()->dom();
            if (auto sigma = dom->isa<Sigma>()) {
                for (size_t i = 0; i < sigma->num_ops(); ++i) {
                    if (Axm::isa<plug::mem::M>(sigma->op(i))) continue;
                    auto var_op = local_lam->var()->proj(sigma->num_ops(), i);
                    auto v      = get_or_emit(arg->proj(sigma->num_ops(), i), body_bb);
                    if (!v.empty()) values_[var_op] = v;
                }
            } else if (!dom->isa<Pi>() && !Axm::isa<plug::mem::M>(dom)) {
                // single scalar argument
                auto v = get_or_emit(arg, body_bb);
                if (!v.empty()) values_[local_lam->var()] = v;
            }
            // Recurse into the local lam's body in the same block
            emit_linalg_body(local_lam, body_bb);
            return;
        }
    }

    std::cerr << "unhandled callee in emit_linalg_body: " << callee->sym().str() << "\n";
    assert(false && "unhandled callee in emit_linalg_body");
}

} // namespace mim::mlir_be
