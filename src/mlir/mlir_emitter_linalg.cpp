#include <functional>

#include <mim/lam.h>
#include <mim/tuple.h>

#include <mim/plug/mem/mem.h>
#include <mim/plug/tensor/autogen.h>
#include <mim/plug/tensor/tensor.h>

#include "mlir/mlir_emitter.h"
#include "mlir/ops/tensor_util.h"

namespace mim::mlir_be {

std::optional<MLIRValue> MLIREmitter::try_emit_tensor_op(const App* app, MLIRBlock& into) {
    auto* def = app;

    if (Axm::isa<plug::tensor::map_reduce>(app)) {
        emit_linalg_generic(app, into);
        return values_[def];
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
    if (auto get_ax = Axm::isa<plug::tensor::get>(app)) {
        // %tensor.get @(T, r, s) (arr, index)

        auto* arr_def = app->arg()->proj(2, 0);
        auto* idx_def = app->arg()->proj(2, 1);

        auto arr_val  = get_or_emit(arr_def, into);
        auto res_type = types_.convert(def->type());

        // unpack index tuple - each element is an Idx literal, cast to index
        std::vector<MLIRValue> indices;
        if (auto sigma = idx_def->type()->isa<Sigma>()) {
            size_t n = sigma->num_ops();
            for (size_t i = 0; i < n; ++i) {
                auto elem     = idx_def->proj(n, i);
                auto elem_val = get_or_emit(elem, into);
                indices.push_back(types_.to_index(elem_val, into, fresh_name("%idx")));
            }
        } else if (auto arr = idx_def->type()->isa<Arr>()) {
            if (auto n = Lit::isa(arr->arity())) {
                for (size_t i = 0; i < *n; ++i) {
                    auto elem_val = get_or_emit(idx_def->proj(*n, i), into);
                    indices.push_back(types_.to_index(elem_val, into, fresh_name("%idx")));
                }
            }
        } else {
            // rank-1: bare Idx
            auto elem_val = get_or_emit(idx_def, into);
            indices.push_back(types_.to_index(elem_val, into, fresh_name("%idx")));
        }

        MLIRValue result{fresh_name(def), res_type};
        into.ops.emplace_back(std::make_unique<TensorExtractOp>(result, arr_val, std::move(indices)));
        return result;
    }

    return std::nullopt;
}
void MLIREmitter::emit_linalg_generic(const App* app, MLIRBlock& into) {
    // New currying chain (depth 8):
    //   app0->arg = is          (inputs pack)
    //   app1->arg = maps        (per-input access lams)
    //   app2->arg = map_out     (output access lam)
    //   app3->arg = (f, init)
    //   app4->arg = (Tis,Ris,Sis) [skip, implicit type args]
    //   app5->arg = (So, Sr)    (output shape, full loop bounds)
    //   app6->arg = (To, Ro, Rr)
    //   app7->arg = nis
    auto* app1 = app->callee()->as<App>();
    auto* app2 = app1->callee()->as<App>();
    auto* app3 = app2->callee()->as<App>();
    auto* app4 = app3->callee()->as<App>();
    auto* app5 = app4->callee()->as<App>();
    auto* app6 = app5->callee()->as<App>();
    auto* app7 = app6->callee()->as<App>();

    auto* inputs_pack = app->arg();
    auto* maps_pack   = app1->arg();
    auto* map_out_def = app2->arg();
    auto [comb, zero] = app3->arg()->projs<2>();
    auto [So, Sr]     = app5->arg()->projs<2>();
    auto [To, Ro, Rr] = app6->arg()->projs<3>();
    auto* nis_def     = app7->arg();

    auto nis_opt = Lit::isa(nis_def);
    auto Ro_opt  = Lit::isa(Ro);
    auto Rr_opt  = Lit::isa(Rr);
    assert(nis_opt && Ro_opt && Rr_opt);
    size_t n_inputs    = *nis_opt;
    size_t ro          = *Ro_opt;
    size_t rr          = *Rr_opt;
    size_t total_loops = ro + rr;

    auto proj_input
        = [&](size_t i) -> const Def* { return n_inputs == 1 ? inputs_pack : inputs_pack->proj(n_inputs, i); };
    auto proj_map_lam = [&](size_t i) -> Lam* {
        auto* d = n_inputs == 1 ? maps_pack : maps_pack->proj(n_inputs, i);
        return d->isa_mut<Lam>();
    };
    auto* map_out = map_out_def->isa_mut<Lam>();
    assert(map_out);

    // ── Result type from So and To ────────────────────────────────────────
    std::vector<std::optional<int64_t>> res_shape;
    for (size_t i = 0; i < ro; ++i) {
        auto dim = ro == 1 ? So : So->proj(ro, i);
        if (auto lit = Lit::isa(dim))
            res_shape.push_back(static_cast<int64_t>(*lit));
        else
            res_shape.push_back(std::nullopt);
    }
    auto res_elem_type = types_.convert(To);
    MLIRTensorType res_tensor;
    res_tensor.shape = res_shape;
    res_tensor.elem  = std::make_shared<MLIRTypeNode>(res_elem_type);
    MLIRType res_type{std::move(res_tensor)};

    // ── Inputs ────────────────────────────────────────────────────────────
    std::vector<MLIRValue> ins;
    for (size_t i = 0; i < n_inputs; ++i)
        ins.push_back(get_or_emit(proj_input(i), into));

    // ── Output buffer ─────────────────────────────────────────────────────
    std::string buf_name = fresh_name(app) + ".buf";
    MLIRValue out_buf{buf_name, res_type};
    into.ops.emplace_back(std::make_unique<TensorEmptyOp>(out_buf));
    std::vector<MLIRValue> outs{out_buf};

    // ── Affine maps from access lams ──────────────────────────────────────

    std::vector<std::string> indexing_maps;
    for (size_t i = 0; i < n_inputs; ++i)
        indexing_maps.push_back(lam_to_affine_map(proj_map_lam(i), total_loops));
    indexing_maps.push_back(lam_to_affine_map(map_out, total_loops));

    // ── Iterator types ────────────────────────────────────────────────────
    std::vector<std::string> iterator_types;
    for (size_t i = 0; i < ro; ++i)
        iterator_types.push_back("parallel");
    for (size_t i = 0; i < rr; ++i)
        iterator_types.push_back("reduction");

    // ── Body seeding (unchanged — path-based strategy) ────────────────────
    std::vector<MLIRValue> body_args;
    auto* body_lam = comb->isa_mut<Lam>();

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
