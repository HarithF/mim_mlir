#include <format>

#include <mim/lam.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/affine/autogen.h>
#include <mim/plug/mem/mem.h>

#include "mlir/mlir_emitter.h"
#include "mlir/ops/scf.h"

namespace mim::mlir_be {

void MLIREmitter::emit_body(Lam* lam, MLIRBlock& into) {
    assert(lam->is_set() && "lam body not set");
    auto* app = lam->body()->isa<App>();
    assert(app && "expected App at lam body");
    auto* callee = app->callee();
    auto* arg    = app->arg();

    // func.return
    if (is_return_callee(callee, curr_ret_var_)) {
        std::vector<MLIRValue> ret_vals;
        if (auto sigma = arg->type()->isa<Sigma>()) {
            for (size_t i = 0; i < sigma->num_ops(); ++i) {
                if (Axm::isa<plug::mem::M>(sigma->op(i))) continue;
                auto v = get_or_emit(arg->proj(sigma->num_ops(), i), into);
                if (!v.empty()) ret_vals.push_back(v);
            }
        } else if (arg->type()->isa<Arr>()) {
            auto v = get_or_emit(arg, into);
            if (!v.empty()) ret_vals.push_back(v);
        } else if (!Axm::isa<plug::mem::M>(arg->type())) {
            auto v = get_or_emit(arg, into);
            if (!v.empty()) ret_vals.push_back(v);
        }
        into.ops.emplace_back(std::make_unique<FuncReturnOp>(std::move(ret_vals)));
        return;
    }

    if (Axm::isa<plug::affine::For>(app)) {
        emit_affine_for(app, into);
        return;
    }

    if (auto vals = try_emit_cond_branch(app, into)) {
        into.ops.emplace_back(std::make_unique<FuncReturnOp>(std::move(*vals)));
        return;
    }

    std::cerr << "unhandled callee: " << callee->node_name() << " sym='" << callee->sym().str() << "'\n";
    assert(false && "unhandled callee in emit_body");
}

void MLIREmitter::emit_affine_for(const App* app, MLIRBlock& into) {
    auto for_ax = Axm::isa<plug::affine::For>(app);
    assert(for_ax);

    auto [body_def, exit_def, loop_args] = for_ax->uncurry_args<3>();
    auto* body_lam                       = body_def->isa_mut<Lam>();
    auto* exit_lam                       = exit_def->isa_mut<Lam>();
    assert(body_lam && exit_lam);

    // collect iv and iter_args from body_lam domain
    MLIRValue iv_val;
    std::vector<MLIRValue> acc_bb;

    auto collect_body_var = [&](const Def* op_type, const Def* var_op) {
        if (Axm::isa<plug::mem::M>(op_type)) return;
        if (op_type->isa<Pi>()) return;
        auto sym = var_op->sym().str();
        MLIRValue v{"%" + (sym.empty() ? std::format("v{}", name_counter_++) : sym), types_.convert(op_type)};
        values_[var_op] = v;
        if (iv_val.empty())
            iv_val = v;
        else
            acc_bb.push_back(v);
    };

    if (auto sigma = body_lam->type()->dom()->isa<Sigma>())
        for (size_t i = 0; i < sigma->num_ops(); ++i)
            collect_body_var(sigma->op(i), body_lam->var()->proj(sigma->num_ops(), i));
    else
        collect_body_var(body_lam->type()->dom(), body_lam->var());

    // collect result values from exit_lam domain
    std::vector<MLIRValue> result_vals;

    auto collect_exit_var = [&](const Def* op_type, const Def* var_op) {
        if (Axm::isa<plug::mem::M>(op_type)) return;
        MLIRValue res{"%" + std::format("for_res{}", name_counter_++), types_.convert(op_type)};
        values_[var_op] = res;
        result_vals.push_back(res);
    };

    if (auto sigma = exit_lam->type()->dom()->isa<Sigma>())
        for (size_t i = 0; i < sigma->num_ops(); ++i)
            collect_exit_var(sigma->op(i), exit_lam->var()->proj(sigma->num_ops(), i));
    else
        collect_exit_var(exit_lam->type()->dom(), exit_lam->var());

    // emit lb, ub, step, init
    size_t n_total = 3 + acc_bb.size();
    auto lb_val    = get_or_emit(loop_args->proj(n_total, 0), into);
    auto ub_val    = get_or_emit(loop_args->proj(n_total, 1), into);
    auto step_val  = get_or_emit(loop_args->proj(n_total, 2), into);

    auto lb_idx   = types_.to_index(lb_val, into, fresh_name("%lb.idx"));
    auto ub_idx   = types_.to_index(ub_val, into, fresh_name("%ub.idx"));
    auto step_idx = types_.to_index(step_val, into, fresh_name("%step.idx"));

    // iv is always index in scf.for
    MLIRType orig_iv_type = iv_val.type;
    MLIRValue iv_idx{iv_val.name + ".idx", MLIRType{MLIRIndexType{}}};

    std::vector<MLIRValue> init_vals;
    for (size_t i = 0; i < acc_bb.size(); ++i)
        init_vals.push_back(get_or_emit(loop_args->proj(n_total, 3 + i), into));

    auto* for_op = new SCFForOp(result_vals, iv_idx, lb_idx, ub_idx, step_idx, acc_bb, init_vals);

    // cast iv back to original type at the start of the body if needed
    auto& body_bb = for_op->body().entry();
    if (!std::holds_alternative<MLIRIndexType>(orig_iv_type)) {
        MLIRValue iv_int{iv_val.name, orig_iv_type};
        body_bb.ops.emplace_back(std::make_unique<IndexCastOp>(iv_int, iv_idx));
        // re-seed all vars that were mapped to iv_val with the cast value
        for (auto& [d, v] : values_)
            if (v.name == iv_val.name) v = iv_int;
    }

    emit_for_body(body_lam, body_bb);
    into.ops.emplace_back(for_op);
    emit_body(exit_lam, into);
}

void MLIREmitter::emit_for_body(Lam* body_lam, MLIRBlock& body_bb) {
    assert(body_lam->is_set());
    auto* app = body_lam->body()->isa<App>();
    assert(app && "expected App in for body");
    auto* callee = app->callee();

    if (is_return_callee(callee, body_lam->ret_var())) {
        std::vector<MLIRValue> yield_vals;

        auto* arg = app->arg();
        if (auto sigma = arg->type()->isa<Sigma>()) {
            for (size_t i = 0; i < sigma->num_ops(); ++i) {
                if (Axm::isa<plug::mem::M>(sigma->op(i))) continue;
                auto v = get_or_emit(arg->proj(sigma->num_ops(), i), body_bb);
                if (!v.empty()) yield_vals.push_back(v);
            }
        } else if (!Axm::isa<plug::mem::M>(arg->type())) {
            auto v = get_or_emit(arg, body_bb);
            if (!v.empty()) yield_vals.push_back(v);
        }
        body_bb.ops.emplace_back(std::make_unique<SCFYieldOp>(std::move(yield_vals)));
        return;
    }

    assert(false && "unhandled callee in emit_for_body");
}

std::optional<std::tuple<MLIRValue, Lam*, Lam*>> MLIREmitter::detect_cond_branch(const App* app, MLIRBlock& into) {
    auto* ex = app->callee()->isa<Extract>();
    if (!ex) return std::nullopt;

    auto hit = select_tuple_as_bool(ex);
    if (!hit) return std::nullopt;
    auto [false_def, true_def] = *hit;

    auto* lam_f = false_def->isa_mut<Lam>(); // ff = else
    auto* lam_t = true_def->isa_mut<Lam>();  // tt = then

    if (!lam_f || !lam_t) return std::nullopt;
    return std::make_tuple(get_or_emit(ex->index(), into), lam_t, lam_f);
}

std::optional<std::vector<MLIRValue>> MLIREmitter::try_emit_cond_branch(const App* app, MLIRBlock& into) {
    auto hit = detect_cond_branch(app, into);
    if (!hit) return std::nullopt;
    auto [cond, then_lam, else_lam] = *hit;
    return emit_scf_if(cond, then_lam, else_lam, into);
}

std::vector<MLIRValue> MLIREmitter::emit_scf_if(MLIRValue cond, Lam* then_lam, Lam* else_lam, MLIRBlock& into) {
    MLIRBlock then_bb, else_bb;
    auto then_vals = emit_branch_values(then_lam, then_bb);
    auto else_vals = emit_branch_values(else_lam, else_bb);

    then_bb.ops.emplace_back(std::make_unique<SCFYieldOp>(then_vals));
    else_bb.ops.emplace_back(std::make_unique<SCFYieldOp>(else_vals));

    std::vector<MLIRValue> results;
    for (auto& v : then_vals)
        results.push_back(MLIRValue{fresh_name("%if_res"), v.type});

    into.ops.emplace_back(std::make_unique<SCFIfOp>(results, cond, std::move(then_bb), std::move(else_bb)));
    return results;
}

std::vector<MLIRValue> MLIREmitter::emit_branch_values(Lam* lam, MLIRBlock& into) {
    assert(lam->is_set());
    auto* app = lam->body()->isa<App>();
    assert(app);
    auto* callee = app->callee();
    auto* arg    = app->arg();

    if (is_return_callee(callee, curr_ret_var_)) {
        std::vector<MLIRValue> vals;
        if (auto sigma = arg->type()->isa<Sigma>()) {
            for (size_t i = 0; i < sigma->num_ops(); ++i) {
                if (Axm::isa<plug::mem::M>(sigma->op(i))) continue;
                auto v = get_or_emit(arg->proj(sigma->num_ops(), i), into);
                if (!v.empty()) vals.push_back(v);
            }
        } else if (!Axm::isa<plug::mem::M>(arg->type())) {
            auto v = get_or_emit(arg, into);
            if (!v.empty()) vals.push_back(v);
        }
        return vals;
    }

    // Nested dispatch
    if (auto vals = try_emit_cond_branch(app, into)) return *vals;

    assert(false && "unsupported branch terminator");
    return {};
}

} // namespace mim::mlir_be
