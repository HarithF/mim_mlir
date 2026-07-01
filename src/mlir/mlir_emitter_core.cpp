// Top-level driver (run/emit_func), argument seeding, the emit_def leaf
// dispatcher, and the scalar-arithmetic dispatch helper (try_emit_arith).

#include <format>
#include <functional>

#include <mim/lam.h>
#include <mim/tuple.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/affine/autogen.h>
#include <mim/plug/core/core.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>
#include <mim/plug/tensor/autogen.h>
#include <mim/plug/tensor/tensor.h>

#include "mlir/mlir_emitter.h"
#include "mlir/ops/arith.h"

namespace mim::mlir_be {

// ----- main functions -----

void MLIREmitter::run() {
    clf_.run();

    MLIRRegion module;
    for (auto& [sym, def] : world_.externals())
        if (auto lam = def->isa_mut<Lam>())
            if (clf_.kind_of(lam) == LamKind::Function) emit_func(lam, module.entry());

    os_ << "module {\n";
    Printer p{os_};
    p.indent();
    p.print_region(module);
    p.dedent();
    os_ << "}\n";
}

// ---- emitters ----

void MLIREmitter::emit_func(Lam* lam, MLIRBlock& into) {
    values_.clear();
    name_counter_ = 0;
    curr_ret_var_ = lam->ret_var();

    std::vector<MLIRValue> args;
    auto dom = lam->type()->dom();
    if (auto sigma = dom->isa<Sigma>())
        for (size_t i = 0; i < sigma->num_ops(); ++i)
            seed_dom_op(lam->var()->proj(sigma->num_ops(), i), args);
    else
        seed_dom_op(lam->var(), args);

    std::vector<MLIRType> ret_types;
    if (auto ret_pi = lam->type()->ret_pi()) {
        auto ret_dom = ret_pi->dom();
        if (auto sigma = ret_dom->isa<Sigma>()) {
            for (auto op : sigma->ops()) {
                if (Axm::isa<plug::mem::M>(op)) continue;
                ret_types.push_back(types_.convert(op));
            }
        } else if (!Axm::isa<plug::mem::M>(ret_dom)) {
            ret_types.push_back(types_.convert(ret_dom));
        }
    }

    auto* func_op = new FuncOp(lam->sym().str(), args, ret_types, lam->is_external());
    emit_body(lam, func_op->body().entry());
    into.ops.emplace_back(func_op);
}

MLIRValue MLIREmitter::emit_def(const Def* def, MLIRBlock& into) {
    if (def->isa<Var>()) {
        assert(false && "Var not pre-seeded in values_");
        return {};
    }

    if (auto lit = def->isa<Lit>()) {
        auto mlir_type = types_.convert(lit->type());
        auto name      = fresh_name(lit);
        MLIRAttr attr;
        if (std::holds_alternative<MLIRIndexType>(mlir_type))
            attr = IndexAttr{static_cast<int64_t>(lit->get<uint64_t>())};
        else if (std::holds_alternative<MLIRIntType>(mlir_type))
            attr = IntAttr{static_cast<int64_t>(lit->get<uint64_t>()), mlir_type};
        else if (std::holds_alternative<MLIRFloatType>(mlir_type))
            attr = FloatAttr{lit_to_double(lit), mlir_type};
        else
            assert(false && "unhandled literal type");
        MLIRValue result{name, mlir_type};
        into.ops.emplace_back(std::make_unique<ConstantOp>(result, attr));
        return result;
    }

    // App - arith & tensor ops
    if (auto app = def->isa<App>()) {
        if (auto v = try_emit_arith(app, into)) return *v;

        // mem ops — no MLIR value
        if (Axm::isa<plug::mem::M>(def->type())) return {};

        if (auto v = try_emit_tensor_op(app, into)) return *v;

        std::cerr << "unhandled App axiom: callee=" << app->callee()->node_name() << " sym='"
                  << app->callee()->sym().str() << "'\n";
        assert(false && "unhandled App in emit_def — missing try_emit_* case");
    }

    if (auto ex = def->isa<Extract>()) {
        auto sym = ex->sym().str();
        if (!sym.empty()) {
            for (auto& [d, v] : values_)
                if (v.name == "%" + sym) return v;
        }

        // index-based: resolve through projection chain
        if (auto lit_idx = Lit::isa(ex->index())) {
            size_t i     = static_cast<size_t>(*lit_idx);
            auto* tuple  = ex->tuple();
            size_t arity = 0;
            if (auto s = tuple->type()->isa<Sigma>())
                arity = s->num_ops();
            else if (auto a = tuple->type()->isa<Arr>())
                if (auto n = Lit::isa(a->arity())) arity = *n;

            if (arity > 0) {
                auto proj = tuple->proj(arity, i);
                if (proj != def) {
                    if (auto it = values_.find(proj); it != values_.end()) return it->second;
                    // recurse — proj is a different node, won't loop
                    auto v = emit_def(proj, into);
                    if (!v.empty()) return v;
                }
            }
        }
        // dynamic index: scalar value-select `(false_val, true_val)#cond` → arith.select
        if (auto v = try_emit_select(ex, into)) return *v;

        assert(false && "Extract not seeded");
        return {};
    }

    if (def->isa<Tuple>() || def->isa<Pack>()) {
        auto mlir_type = types_.convert(def->type());

        if (std::holds_alternative<MLIRTensorType>(mlir_type)) {
            auto name = fresh_name(def);
            auto& tt  = std::get<MLIRTensorType>(mlir_type);

            // Detect uniform Pack: arbitrarily nested Packs all wrapping a single Lit.
            // Avoid enumerating potentially huge uniform tensors — emit a splat instead.
            auto* uniform_lit = [&]() -> const Lit* {
                const Def* cur = def;
                while (auto pack = cur->isa<Pack>())
                    cur = pack->body();
                return cur->isa<Lit>();
            }();

            if (uniform_lit) {
                double val            = lit_to_double(uniform_lit);
                std::string val_str   = format_mlir_float(val);
                std::string dense_str = std::format("dense<{}>", val_str);
                MLIRValue result{name, mlir_type};
                into.ops.emplace_back(std::make_unique<DenseConstOp>(result, std::move(dense_str)));
                return result;
            }

            // Non-uniform: enumerate as before.
            std::function<void(const Def*, std::vector<double>&)> collect_vals;
            collect_vals = [&](const Def* d, std::vector<double>& out) {
                if (auto lit = d->isa<Lit>()) {
                    out.push_back(lit_to_double(lit));
                    return;
                }
                if (auto inner = d->isa<Tuple>()) {
                    for (size_t i = 0; i < inner->num_ops(); ++i)
                        collect_vals(inner->op(i), out);
                    return;
                }
                if (auto pack = d->isa<Pack>()) {
                    if (auto n = Lit::isa(pack->arity())) {
                        for (size_t i = 0; i < *n; ++i)
                            collect_vals(pack->body(), out);
                        return;
                    }
                }
                assert(false && "unexpected node in literal tensor");
            };

            std::vector<double> flat_vals;
            collect_vals(def, flat_vals);
            std::string dense_str = make_dense_attr(flat_vals, tt);
            MLIRValue result{name, mlir_type};
            into.ops.emplace_back(std::make_unique<DenseConstOp>(result, std::move(dense_str)));
            return result;
        }
    }

    std::cerr << "unhandled def: " << def->node_name() << " sym='" << def->sym().str() << "'"
              << " type=" << def->type()->node_name() << "\n";
    assert(false && "unhandled def in emit_def");
    return {};
}

std::optional<MLIRValue> MLIREmitter::try_emit_select(const Extract* ex, MLIRBlock& into) {
    auto hit = select_tuple_as_bool(ex);
    if (!hit) return std::nullopt;
    auto [false_def, true_def] = *hit;

    // both elements must be plain values, not Lams — that's the control-flow case instead
    if (false_def->isa_mut<Lam>() || true_def->isa_mut<Lam>()) return std::nullopt;

    auto false_val = get_or_emit(false_def, into);
    auto true_val  = get_or_emit(true_def, into);
    auto cond_val  = get_or_emit(ex->index(), into);

    auto t = types_.convert(ex->type());
    MLIRValue result{fresh_name(ex), t};
    into.ops.emplace_back(std::make_unique<SelectOp>(result, cond_val, true_val, false_val));
    return result;
}

std::optional<MLIRValue> MLIREmitter::try_emit_arith(const App* app, MLIRBlock& into) {
    namespace core = plug::core;
    auto* def      = app;

    if (auto wrap = Axm::isa<core::wrap>(app)) {
        auto [a, b]      = wrap->args<2>([this, &into](auto d) { return get_or_emit(d, into); });
        auto result_type = types_.convert(def->type());
        BinaryIntOp::Kind kind;
        switch (wrap.id()) {
            case core::wrap::add: kind = BinaryIntOp::Kind::Add; break;
            case core::wrap::sub: kind = BinaryIntOp::Kind::Sub; break;
            case core::wrap::mul: kind = BinaryIntOp::Kind::Mul; break;
            case core::wrap::shl: kind = BinaryIntOp::Kind::Shl; break;
            default: assert(false && "unhandled core.wrap op");
        }
        MLIRValue result{fresh_name(def), result_type};
        into.ops.emplace_back(std::make_unique<BinaryIntOp>(result, kind, a, b));
        return result;
    }

    if (auto div = Axm::isa<core::div>(app)) {
        auto [a, b]      = div->args<2>([this, &into](auto d) { return get_or_emit(d, into); });
        auto result_type = types_.convert(def->type());
        BinaryIntOp::Kind kind;
        switch (div.id()) {
            case core::div::sdiv: kind = BinaryIntOp::Kind::DivS; break;
            case core::div::udiv: kind = BinaryIntOp::Kind::DivU; break;
            case core::div::srem: kind = BinaryIntOp::Kind::RemS; break;
            case core::div::urem: kind = BinaryIntOp::Kind::RemU; break;
            default: assert(false && "unhandled core.div op");
        }
        MLIRValue result{fresh_name(def), result_type};
        into.ops.emplace_back(std::make_unique<BinaryIntOp>(result, kind, a, b));
        return result;
    }

    if (auto arith = Axm::isa<plug::math::arith>(app)) {
        auto [a, b]      = arith->args<2>([this, &into](auto d) { return get_or_emit(d, into); });
        auto result_type = types_.convert(def->type());
        BinaryFloatOp::Kind kind;
        switch (arith.id()) {
            case plug::math::arith::add: kind = BinaryFloatOp::Kind::Add; break;
            case plug::math::arith::sub: kind = BinaryFloatOp::Kind::Sub; break;
            case plug::math::arith::mul: kind = BinaryFloatOp::Kind::Mul; break;
            case plug::math::arith::div: kind = BinaryFloatOp::Kind::Div; break;
            case plug::math::arith::rem: kind = BinaryFloatOp::Kind::Rem; break;
            default: assert(false && "unhandled math.arith op");
        }
        MLIRValue result{fresh_name(def), result_type};
        into.ops.emplace_back(std::make_unique<BinaryFloatOp>(result, kind, a, b));
        return result;
    }

    if (auto icmp = Axm::isa<core::icmp>(app)) {
        auto [a, b] = icmp->args<2>([this, &into](auto d) { return get_or_emit(d, into); });
        MLIRType i1{MLIRIntType{1}};
        CmpiOp::Pred pred;
        switch (icmp.id()) {
            case core::icmp::e: pred = CmpiOp::Pred::Eq; break;
            case core::icmp::ne: pred = CmpiOp::Pred::Ne; break;
            case core::icmp::sl: pred = CmpiOp::Pred::Slt; break;
            case core::icmp::sle: pred = CmpiOp::Pred::Sle; break;
            case core::icmp::sg: pred = CmpiOp::Pred::Sgt; break;
            case core::icmp::sge: pred = CmpiOp::Pred::Sge; break;
            case core::icmp::ul: pred = CmpiOp::Pred::Ult; break;
            case core::icmp::ule: pred = CmpiOp::Pred::Ule; break;
            case core::icmp::ug: pred = CmpiOp::Pred::Ugt; break;
            case core::icmp::uge: pred = CmpiOp::Pred::Uge; break;
            default: assert(false && "unhandled core.icmp pred");
        }
        MLIRValue result{fresh_name(def), i1};
        into.ops.emplace_back(std::make_unique<CmpiOp>(result, pred, a, b));
        return result;
    }

    // math::tri → MathUnaryOp
    if (auto tri = Axm::isa<plug::math::tri>(def)) {
        auto a = get_or_emit(app->arg(), into);
        auto t = types_.convert(def->type());
        MathUnaryOp::Kind kind;
        switch (tri.id()) {
            case plug::math::tri::tanh: kind = MathUnaryOp::Kind::Tanh; break;
            case plug::math::tri::sin: kind = MathUnaryOp::Kind::Sin; break;
            case plug::math::tri::cos: kind = MathUnaryOp::Kind::Cos; break;
            default: assert(false && "unhandled math.tri");
        }
        MLIRValue result{fresh_name(def), t};
        into.ops.emplace_back(std::make_unique<MathUnaryOp>(result, kind, a));
        return result;
    }

    // math::extrema → BinaryFloatOp
    if (auto extr = Axm::isa<plug::math::extrema>(def)) {
        auto [a, b] = extr->args<2>([this, &into](auto d) { return get_or_emit(d, into); });
        auto t      = types_.convert(extr->type());
        BinaryFloatOp::Kind kind;
        switch (extr.id()) {
            case plug::math::extrema::fmax: kind = BinaryFloatOp::Kind::MaxNum; break;
            case plug::math::extrema::ieee754max: kind = BinaryFloatOp::Kind::Maximum; break;
            case plug::math::extrema::fmin: kind = BinaryFloatOp::Kind::MinNum; break;
            case plug::math::extrema::ieee754min: kind = BinaryFloatOp::Kind::Minimum; break;
            default: assert(false && "unhandled math.extrema");
        }
        MLIRValue result{fresh_name(def), t};
        into.ops.emplace_back(std::make_unique<BinaryFloatOp>(result, kind, a, b));
        return result;
    }

    if (auto nat_op = Axm::isa<core::nat>(app)) {
        auto* arg        = app->arg();
        auto a           = get_or_emit(arg->proj(2, 0), into);
        auto b           = get_or_emit(arg->proj(2, 1), into);
        auto result_type = types_.convert(def->type());
        BinaryIntOp::Kind kind;
        switch (nat_op.id()) {
            case core::nat::add: kind = BinaryIntOp::Kind::Add; break;
            case core::nat::mul: kind = BinaryIntOp::Kind::Mul; break;
            default: assert(false && "unhandled core.nat op");
        }
        MLIRValue result{fresh_name(def), result_type};
        into.ops.emplace_back(std::make_unique<BinaryIntOp>(result, kind, a, b));
        return result;
    }

    // math::is_finite
    if (auto isf = Axm::isa<plug::math::is_finite>(app)) {
        auto a = get_or_emit(app->arg(), into);
        MLIRType i1{MLIRIntType{1}};
        MLIRValue result{fresh_name(def), i1};
        into.ops.emplace_back(std::make_unique<MathIsFiniteOp>(result, a));
        return result;
    }

    // math::abs
    if (Axm::isa<plug::math::abs>(app)) {
        auto a = get_or_emit(app->arg(), into);
        auto t = types_.convert(def->type());
        MLIRValue result{fresh_name(def), t};
        into.ops.emplace_back(std::make_unique<MathUnaryOp>(result, MathUnaryOp::Kind::Abs, a));
        return result;
    }

    // math::cmp (float comparisons)
    if (auto cmp = Axm::isa<plug::math::cmp>(app)) {
        auto [a, b] = cmp->args<2>([this, &into](auto d) { return get_or_emit(d, into); });
        MLIRType i1{MLIRIntType{1}};
        CmpfOp::Pred pred;
        switch (cmp.id()) {
            case plug::math::cmp::e: pred = CmpfOp::Pred::Oeq; break;
            case plug::math::cmp::ne: pred = CmpfOp::Pred::One; break;
            case plug::math::cmp::l: pred = CmpfOp::Pred::Olt; break;
            case plug::math::cmp::le: pred = CmpfOp::Pred::Ole; break;
            case plug::math::cmp::g: pred = CmpfOp::Pred::Ogt; break;
            case plug::math::cmp::ge: pred = CmpfOp::Pred::Oge; break;
            case plug::math::cmp::ul: pred = CmpfOp::Pred::Ult; break;
            case plug::math::cmp::ule: pred = CmpfOp::Pred::Ule; break;
            case plug::math::cmp::ug: pred = CmpfOp::Pred::Ugt; break;
            case plug::math::cmp::uge: pred = CmpfOp::Pred::Uge; break;
            case plug::math::cmp::une: pred = CmpfOp::Pred::Une; break;
            case plug::math::cmp::ue: pred = CmpfOp::Pred::Ueq; break;
            case plug::math::cmp::o: pred = CmpfOp::Pred::Ord; break;
            case plug::math::cmp::u: pred = CmpfOp::Pred::Uno; break;
            default: assert(false && "unhandled math.cmp pred");
        }
        MLIRValue result{fresh_name(def), i1};
        into.ops.emplace_back(std::make_unique<CmpfOp>(result, pred, a, b));
        return result;
    }

    // math::exp (exp/log variants — 'lbb' etc. are sub-tag combinations)
    if (Axm::isa<plug::math::exp>(app)) {
        auto a = get_or_emit(app->arg(), into);
        auto t = types_.convert(def->type());
        MLIRValue result{fresh_name(def), t};
        into.ops.emplace_back(std::make_unique<MathUnaryOp>(result, MathUnaryOp::Kind::Exp, a));
        return result;
    }
    return std::nullopt;
}

} // namespace mim::mlir_be
