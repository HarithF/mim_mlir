// Top-level driver (run/emit_func), argument seeding, the emit_def leaf
// dispatcher, and the scalar-arithmetic dispatch helper (try_emit_arith).

#include <format>
#include <functional>
#include <map>
#include <set>

#include <mim/lam.h>
#include <mim/tuple.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/affine/autogen.h>
#include <mim/plug/core/core.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>
#include <mim/plug/tensor/autogen.h>
#include <mim/plug/tensor/tensor.h>

#include "mim/util/util.h"

#include "mlir/mlir_emitter.h"

namespace mim::mlir_be {

// ----- Helpers -------
std::string MLIREmitter::fresh_name(const Def* def) {
    auto sym = def->sym().str();
    if (!sym.empty() && sym[0] != '_') return "%" + sym;
    return std::format("%v{}", name_counter_++);
}
std::string MLIREmitter::fresh_name(std::string prefix) { return prefix + std::to_string(name_counter_++); }

bool MLIREmitter::is_return_callee(const Def* c, const Def* ret_var) {
    if (c == ret_var) return true;
    if (auto ex = c->isa<Extract>()) return ex->sym().str() == "return";
    return false;
}

MLIRValue MLIREmitter::get_or_emit(const Def* def, MLIRBlock& into) {
    if (auto it = values_.find(def); it != values_.end()) return it->second;
    auto val     = emit_def(def, into);
    values_[def] = val;
    return val;
}

// Recursively unpack op into individual MLIR argsv
void MLIREmitter::seed_dom_op(const Def* op, std::vector<MLIRValue>& args) {
    if (Axm::isa<plug::mem::M>(op->type())) return;
    if (op->type()->isa<Pi>()) return;

    if (auto sigma = op->type()->isa<Sigma>()) {
        for (size_t i = 0; i < sigma->num_ops(); ++i)
            seed_dom_op(op->proj(sigma->num_ops(), i), args);
        return;
    }
    if (auto arr = op->type()->isa<Arr>()) {
        if (auto n = Lit::isa(arr->arity())) {
            bool is_arg_tuple  = false;
            size_t check_limit = std::min(*n, (uint64_t)4);
            for (size_t i = 0; i < check_limit; ++i) {
                auto sym = op->proj(*n, i)->sym().str();
                if (!sym.empty() && sym[0] != '_') {
                    is_arg_tuple = true;
                    break;
                }
            }
            if (is_arg_tuple) {
                for (size_t i = 0; i < *n; ++i)
                    seed_dom_op(op->proj(*n, i), args);
                return;
            }
        }
    }

    MLIRValue v{fresh_name(op), types_.convert(op->type())};
    args.push_back(v);
    values_[op] = v;
}

// Recursively walks the var tree and seeds any unseeded leaf by sym match.
void MLIREmitter::seed_var_tree(const Def* d) {
    if (values_.contains(d)) return;
    if (Axm::isa<plug::mem::M>(d->type())) return;
    if (d->type()->isa<Pi>()) return;

    if (auto sigma = d->type()->isa<Sigma>()) {
        for (size_t i = 0; i < sigma->num_ops(); ++i)
            seed_var_tree(d->proj(sigma->num_ops(), i));
    } else if (auto arr = d->type()->isa<Arr>()) {
        if (auto n = Lit::isa(arr->arity()))
            for (size_t i = 0; i < *n; ++i)
                seed_var_tree(d->proj(*n, i));
    }

    if (!values_.contains(d)) {
        auto sym = d->sym().str();
        if (!sym.empty()) {
            for (auto& [seeded, v] : values_)
                if (v.name == "%" + sym) {
                    values_[d] = v;
                    return;
                }
        }
    }
}
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
            attr = FloatAttr{lit->get<double>(), mlir_type};
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
        assert(false && "Extract not seeded");
        return {};
    }

    if (auto tup = def->isa<Tuple>()) {
        auto mlir_type = types_.convert(def->type());
        auto name      = fresh_name(def);

        std::function<void(const Def*, std::vector<double>&)> collect_vals;
        collect_vals = [&](const Def* d, std::vector<double>& out) {
            if (auto lit = d->isa<Lit>()) {
                out.push_back(mim::bitcast_resize<double>(lit->get<u64>()));
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

        auto& tt = std::get<MLIRTensorType>(mlir_type);
        std::vector<double> flat_vals;
        collect_vals(tup, flat_vals);
        std::string dense_str = make_dense_attr(flat_vals, tt);

        MLIRValue result{name, mlir_type};
        into.ops.emplace_back(std::make_unique<DenseConstOp>(result, std::move(dense_str)));
        return result;
    }

    std::cerr << "unhandled def: " << def->node_name() << " sym='" << def->sym().str() << "'"
              << " type=" << def->type()->node_name() << "\n";
    assert(false && "unhandled def in emit_def");
    return {};
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

    return std::nullopt;
}

} // namespace mim::mlir_be
