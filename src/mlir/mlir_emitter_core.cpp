//
// Top-level driver (run/emit_func), argument seeding, the emit_def leaf
// dispatcher, and the scalar-arithmetic dispatch helper (try_emit_arith).

#include <cstdint>
#include <cstdlib>

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
#include "mlir/ops/memref.h"
#include "mlir/ops/tensor_util.h"

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
    names_.clear();
    used_names_.clear();
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
        else if (auto ft = std::get_if<MLIRFloatType>(&mlir_type))
            attr = FloatAttr{lit_to_double(lit->get<uint64_t>(), ft->bits), mlir_type};

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

        auto [axm, curry, trip] = Axm::get(app);
        std::cerr << "unhandled App axiom: callee=" << app->callee()->node_name() << " sym='"
                  << app->callee()->sym().str() << "' axm='" << (axm ? axm->sym().str() : "<none>")
                  << "' curry=" << (int)curry << " type=" << app->type() << '\n';
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
        // Walk up an Extract chain of literal indices
        if (std::holds_alternative<MLIRIntType>(types_.convert(ex->type()))
            || std::holds_alternative<MLIRFloatType>(types_.convert(ex->type()))
            || std::holds_alternative<MLIRIndexType>(types_.convert(ex->type()))) {
            std::vector<size_t> indices; // innermost-first
            const Def* cur = ex;
            while (auto step = cur->isa<Extract>()) {
                auto lit = Lit::isa(step->index());
                if (!lit) break;
                indices.push_back(static_cast<size_t>(*lit));
                cur = step->tuple();
                if (auto it = values_.find(cur); it != values_.end()) {
                    if (std::holds_alternative<MLIRTensorType>(it->second.type)) {
                        std::reverse(indices.begin(), indices.end());
                        std::vector<MLIRValue> idx_vals;
                        idx_vals.reserve(indices.size());
                        for (size_t idx : indices) {
                            MLIRValue idx_v{fresh_name("%c"), MLIRType{MLIRIndexType{}}};
                            into.ops.emplace_back(
                                std::make_unique<ConstantOp>(idx_v, IndexAttr{static_cast<int64_t>(idx)}));
                            idx_vals.push_back(idx_v);
                        }
                        MLIRValue result{fresh_name(ex), types_.convert(ex->type())};
                        into.ops.emplace_back(
                            std::make_unique<TensorExtractOp>(result, it->second, std::move(idx_vals)));
                        return result;
                    }
                    break;
                }
            }
        }

        // A runtime index into a tensor is MimIR's n-ary extract, one index per axis: tensor.extract.
        if (!Lit::isa(ex->index()) && std::holds_alternative<MLIRTensorType>(types_.convert(ex->tuple()->type()))
            && !std::holds_alternative<MLIRTensorType>(types_.convert(ex->type()))) {
            auto arr_val = get_or_emit(ex->tuple(), into);
            auto* idx    = ex->index();
            size_t rank  = 1;
            if (auto sigma = idx->type()->isa<Sigma>())
                rank = sigma->num_ops();
            else if (auto arr = idx->type()->isa<Arr>())
                if (auto n = Lit::isa(arr->arity())) rank = *n;

            std::vector<MLIRValue> indices;
            for (size_t i = 0; i < rank; ++i) {
                auto elem_val = get_or_emit(rank == 1 ? idx : idx->proj(rank, i), into);
                indices.push_back(types_.to_index(elem_val, into, fresh_name("%idx")));
            }
            MLIRValue result{fresh_name(ex), types_.convert(ex->type())};
            into.ops.emplace_back(std::make_unique<TensorExtractOp>(result, arr_val, std::move(indices)));
            return result;
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

            // Uniform Pack: all elements are the same Lit — emit a splat.
            {
                const Def* cur = def;
                while (auto pack = cur->isa<Pack>())
                    cur = pack->elem();
                if (auto lit = cur->isa<Lit>()) {
                    std::string dense_str = make_dense_splat(lit->get<u64>(), tt);
                    MLIRValue result{name, mlir_type};
                    into.ops.emplace_back(std::make_unique<DenseConstOp>(result, std::move(dense_str)));
                    return result;
                }
                // A Pack of a computed scalar has no dense attribute to print; splat it at runtime.
                if (cur != def) {
                    auto val = get_or_emit(cur, into);
                    MLIRValue buf{name + ".empty", mlir_type};
                    into.ops.emplace_back(std::make_unique<TensorEmptyOp>(buf));
                    MLIRValue result{name, mlir_type};
                    into.ops.emplace_back(std::make_unique<LinalgFillOp>(result, val, buf));
                    return result;
                }
            }

            // Non-uniform, but only when every leaf really is a literal: `make_dense_attr` needs one value per
            // element and has nothing to print for a computed one.
            {
                std::vector<uint64_t> raw;
                collect_lit_tensor(def, raw);
                size_t want = 1;
                bool statik = true;
                for (auto d : tt.shape)
                    if (d)
                        want *= static_cast<size_t>(*d);
                    else
                        statik = false;
                if (statik && raw.size() == want) {
                    auto dense_str = make_dense_attr(raw, tt);
                    MLIRValue result{name, mlir_type};
                    into.ops.emplace_back(std::make_unique<DenseConstOp>(result, std::move(dense_str)));
                    return result;
                }
            }

            // A Tuple of computed tensors is a stack along a new leading axis (a q/k/v split arrives this way):
            // grow each element by that axis and concatenate along it.
            if (auto tup = def->isa<Tuple>(); tup && tt.shape.size() > 1) {
                std::vector<MLIRValue> grown;
                for (size_t i = 0; i < tup->num_ops(); ++i) {
                    auto val = get_or_emit(tup->op(i), into);
                    if (val.empty() || !std::holds_alternative<MLIRTensorType>(val.type)) return {};

                    auto& src = std::get<MLIRTensorType>(val.type);
                    if (src.shape.size() + 1 != tt.shape.size()) return {};

                    MLIRTensorType one = src;
                    one.shape.insert(one.shape.begin(), 1);
                    // Source axis 0 spans the inserted axis and its own; the rest map across one-to-one.
                    std::vector<std::vector<int64_t>> reassoc{
                        {0, 1}
                    };
                    for (size_t d = 2; d < tt.shape.size(); ++d)
                        reassoc.push_back({static_cast<int64_t>(d)});

                    MLIRValue up{fresh_name(name + ".lift"), MLIRType{std::move(one)}};
                    into.ops.emplace_back(std::make_unique<TensorExpandShapeOp>(up, val, std::move(reassoc)));
                    grown.push_back(up);
                }
                MLIRValue result{name, mlir_type};
                into.ops.emplace_back(std::make_unique<TensorConcatOp>(result, std::move(grown), 0));
                return result;
            }
            return {};
        }
        return {};
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

namespace {

/// The `fastmath<…>` clause for a `%math` op's mode, empty when it permits nothing.
/// MLIR's `arith.fastmath` flag names mirror LLVM's, which is what `math::Mode` already encodes, so the
/// flags map across one-to-one.
std::string fastmath_clause(const Def* mode) {
    auto lit = Lit::isa(mode);
    if (!lit) return {};
    auto m = static_cast<plug::math::Mode>(*lit);

    // `reassoc` is withheld by default: it lets LLVM vectorize a reduction along whatever axis the loop
    // nest offers, and for a windowed (conv) access that axis is strided — the result is a gather per
    // iteration, which is slower than the scalar accumulation it replaces.
    if (auto* env = std::getenv("MIMIR_MLIR_FASTMATH")) {
        if (std::string_view{env} == "none") return {};
        if (std::string_view{env} != "full") m = m & ~plug::math::Mode::reassoc;
    } else {
        m = m & ~plug::math::Mode::reassoc;
    }

    if (m == plug::math::Mode::none) return {};
    if (m == plug::math::Mode::fast) return "fastmath<fast>";

    std::string flags;
    auto add = [&](plug::math::Mode f, std::string_view name) {
        if (fe::has_flag(m, f)) flags += (flags.empty() ? "" : ",") + std::string(name);
    };
    add(plug::math::Mode::nnan, "nnan");
    add(plug::math::Mode::ninf, "ninf");
    add(plug::math::Mode::nsz, "nsz");
    add(plug::math::Mode::arcp, "arcp");
    add(plug::math::Mode::contract, "contract");
    add(plug::math::Mode::afn, "afn");
    add(plug::math::Mode::reassoc, "reassoc");
    return flags.empty() ? std::string{} : "fastmath<" + flags + ">";
}

} // namespace

std::optional<MLIRValue> MLIREmitter::try_emit_arith(const App* app, MLIRBlock& into) {
    namespace core = plug::core;
    auto* def      = app;

    if (auto wrap = Axm::isa<core::wrap>(app)) {
        auto [a, b] = app->args<2>([this, &into](auto d) { return get_or_emit(d, into); });

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
        // auto [a, b]      = div->args<2>([this, &into](auto d) { return get_or_emit(d, into); });
        auto [a, b] = div->args<2>([this, &into](auto d) { return get_or_emit(d, into); });

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
        auto [mode, _ab] = arith->uncurry_args<2>();

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
        into.ops.emplace_back(std::make_unique<BinaryFloatOp>(result, kind, a, b, fastmath_clause(mode)));
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
        into.ops.emplace_back(std::make_unique<MathUnaryOp>(result, kind, a, fastmath_clause(tri->decurry()->arg())));
        return result;
    }

    // math::extrema → BinaryFloatOp
    if (auto extr = Axm::isa<plug::math::extrema>(def)) {
        auto [a, b] = extr->args<2>([this, &into](auto d) { return get_or_emit(d, into); });

        auto t = types_.convert(extr->type());
        BinaryFloatOp::Kind kind;
        switch (extr.id()) {
            case plug::math::extrema::fmax: kind = BinaryFloatOp::Kind::MaxNum; break;
            case plug::math::extrema::ieee754max: kind = BinaryFloatOp::Kind::Maximum; break;
            case plug::math::extrema::fmin: kind = BinaryFloatOp::Kind::MinNum; break;
            case plug::math::extrema::ieee754min: kind = BinaryFloatOp::Kind::Minimum; break;
            default: assert(false && "unhandled math.extrema");
        }
        MLIRValue result{fresh_name(def), t};
        into.ops.emplace_back(
            std::make_unique<BinaryFloatOp>(result, kind, a, b, fastmath_clause(extr->decurry()->arg())));
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

    if (auto rt = Axm::isa<plug::math::rt>(def)) {
        auto a = get_or_emit(app->arg(), into);
        auto t = types_.convert(def->type());
        MathUnaryOp::Kind kind;
        switch (rt.id()) {
            case plug::math::rt::sq: kind = MathUnaryOp::Kind::Sqrt; break;
            case plug::math::rt::cb: kind = MathUnaryOp::Kind::Cbrt; break;
            default: assert(false && "unhandled math.rt");
        }
        MLIRValue result{fresh_name(def), t};
        into.ops.emplace_back(std::make_unique<MathUnaryOp>(result, kind, a));
        return result;
    }

    // exp10 and the unused sub-tags have no math-dialect counterpart and fall through to the caller's diagnostic.
    if (auto e = Axm::isa<plug::math::exp>(def)) {
        std::optional<MathUnaryOp::Kind> kind;
        switch (e.id()) {
            case plug::math::exp::exp: kind = MathUnaryOp::Kind::Exp; break;
            case plug::math::exp::exp2: kind = MathUnaryOp::Kind::Exp2; break;
            case plug::math::exp::log: kind = MathUnaryOp::Kind::Log; break;
            case plug::math::exp::log2: kind = MathUnaryOp::Kind::Log2; break;
            case plug::math::exp::log10: kind = MathUnaryOp::Kind::Log10; break;
            default: break;
        }
        if (kind) {
            auto a = get_or_emit(app->arg(), into);
            auto t = types_.convert(def->type());
            MLIRValue result{fresh_name(def), t};
            into.ops.emplace_back(std::make_unique<MathUnaryOp>(result, *kind, a));
            return result;
        }
    }
    return std::nullopt;
}

} // namespace mim::mlir_be
