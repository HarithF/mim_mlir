//
// Affine expression construction (from `%affine` index expressions), folding, range analysis, and printing.

#include "mlir/ops/affine.h"

#include <algorithm>
#include <format>

#include <mim/tuple.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/affine/autogen.h>

namespace mim::mlir_be {

namespace {

std::optional<int64_t> const_val(const AffineExprRef& e) {
    if (auto c = std::get_if<AffineConst>(&e->expr)) return c->value;
    return {};
}

/// Binding strength, so str() can drop the parens MLIR does not need: `+`/`-` bind loosest, then the
/// multiplicative operators, and leaves never need parens.
int precedence(const AffineExpr& e) {
    if (auto b = std::get_if<AffineBin>(&e)) return (b->op == AffineBinOp::Add || b->op == AffineBinOp::Sub) ? 1 : 2;
    return 3;
}

std::string_view spelling(AffineBinOp op) {
    switch (op) {
        case AffineBinOp::Add: return "+";
        case AffineBinOp::Sub: return "-";
        case AffineBinOp::Mul: return "*";
        case AffineBinOp::FloorDiv: return "floordiv";
        case AffineBinOp::CeilDiv: return "ceildiv";
        case AffineBinOp::Mod: return "mod";
    }
    return "?";
}

/// True when @p e is a multiple of @p c for every value of its dims — enough to fold `(x + y) floordiv c` by
/// splitting off the part that divides evenly.
bool divides(const AffineExprNode& e, int64_t c) {
    if (c == 0) return false;
    if (auto k = std::get_if<AffineConst>(&e.expr)) return k->value % c == 0;
    if (auto b = std::get_if<AffineBin>(&e.expr)) {
        switch (b->op) {
            case AffineBinOp::Mul: {
                // One side is constant by construction.
                if (auto k = const_val(b->rhs)) return *k % c == 0 || divides(*b->lhs, c);
                if (auto k = const_val(b->lhs)) return *k % c == 0 || divides(*b->rhs, c);
                return false;
            }
            case AffineBinOp::Add:
            case AffineBinOp::Sub: return divides(*b->lhs, c) && divides(*b->rhs, c);
            default: return false;
        }
    }
    return false;
}

AffineExprRef make(AffineExpr e) { return std::make_shared<const AffineExprNode>(std::move(e)); }

AffineExprRef bin(AffineBinOp op, AffineExprRef a, AffineExprRef b) {
    return make(AffineBin{op, std::move(a), std::move(b)});
}

} // namespace

// ----- printing & queries -----

std::string AffineExprNode::str() const {
    if (auto d = std::get_if<AffineDim>(&expr)) return std::format("d{}", d->pos);
    if (auto c = std::get_if<AffineConst>(&expr)) return std::to_string(c->value);

    auto& b   = std::get<AffineBin>(expr);
    auto prec = precedence(expr);
    auto wrap = [&](const AffineExprRef& child, bool rhs) {
        auto s = child->str();
        // Parenthesize a looser child, and any binary rhs — `a - (b - c)` and `a floordiv (b * c)` do not
        // re-associate.
        auto child_prec = precedence(child->expr);
        if (child_prec < prec || (rhs && child_prec <= prec && std::holds_alternative<AffineBin>(child->expr)))
            return "(" + s + ")";
        return s;
    };

    return std::format("{} {} {}", wrap(b.lhs, false), spelling(b.op), wrap(b.rhs, true));
}

void AffineExprNode::used_dims(std::vector<size_t>& out) const {
    if (auto d = std::get_if<AffineDim>(&expr)) {
        if (std::ranges::find(out, d->pos) == out.end()) out.push_back(d->pos);
        return;
    }
    if (auto b = std::get_if<AffineBin>(&expr)) {
        b->lhs->used_dims(out);
        b->rhs->used_dims(out);
    }
}

// ----- range analysis -----

AffineRange aff_range(const AffineExprRef& e, const AffineExtents& extents) {
    // Wide enough to mean "unknown" while leaving room for the arithmetic below.
    constexpr int64_t Unknown = int64_t{1} << 40;

    if (auto d = std::get_if<AffineDim>(&e->expr)) {
        if (d->pos < extents.size())
            if (auto n = extents[d->pos]) return {0, *n > 0 ? *n - 1 : 0};
        return {0, Unknown};
    }
    if (auto c = std::get_if<AffineConst>(&e->expr)) return {c->value, c->value};

    auto& b = std::get<AffineBin>(e->expr);
    auto l  = aff_range(b.lhs, extents);
    auto r  = aff_range(b.rhs, extents);

    switch (b.op) {
        case AffineBinOp::Add: return {l.lo + r.lo, l.hi + r.hi};
        case AffineBinOp::Sub: return {l.lo - r.hi, l.hi - r.lo};
        case AffineBinOp::Mul: {
            std::array cands{l.lo * r.lo, l.lo * r.hi, l.hi * r.lo, l.hi * r.hi};
            return {std::ranges::min(cands), std::ranges::max(cands)};
        }
        case AffineBinOp::FloorDiv:
        case AffineBinOp::CeilDiv: {
            if (r.lo != r.hi || r.lo <= 0) return {0, Unknown};
            auto adjust = b.op == AffineBinOp::CeilDiv ? r.lo - 1 : 0;
            return {l.lo / r.lo, (l.hi + adjust) / r.lo};
        }
        case AffineBinOp::Mod:
            if (r.lo != r.hi || r.lo <= 0) return {0, Unknown};
            return {0, r.lo - 1};
    }
    return {0, Unknown};
}

// ----- smart constructors -----

AffineExprRef aff_dim(size_t pos) { return make(AffineDim{pos}); }
AffineExprRef aff_const(int64_t value) { return make(AffineConst{value}); }

AffineExprRef aff_add(AffineExprRef a, AffineExprRef b) {
    auto ka = const_val(a);
    auto kb = const_val(b);
    if (ka && kb) return aff_const(*ka + *kb);
    if (ka && *ka == 0) return b;
    if (kb && *kb == 0) return a;
    // A negative constant factor reads better as a subtraction.
    if (auto mul = std::get_if<AffineBin>(&b->expr); mul && mul->op == AffineBinOp::Mul)
        if (auto k = const_val(mul->rhs); k && *k < 0) return aff_sub(std::move(a), aff_mul(mul->lhs, -*k));
    return bin(AffineBinOp::Add, std::move(a), std::move(b));
}

AffineExprRef aff_sub(AffineExprRef a, AffineExprRef b) {
    auto ka = const_val(a);
    auto kb = const_val(b);
    if (ka && kb) return aff_const(*ka - *kb);
    if (kb && *kb == 0) return a;
    return bin(AffineBinOp::Sub, std::move(a), std::move(b));
}

AffineExprRef aff_mul(AffineExprRef a, int64_t c) {
    if (c == 0) return aff_const(0);
    if (c == 1) return a;
    if (auto ka = const_val(a)) return aff_const(*ka * c);
    // Collapse nested scaling so strides multiply out instead of nesting.
    if (auto b = std::get_if<AffineBin>(&a->expr); b && b->op == AffineBinOp::Mul)
        if (auto k = const_val(b->rhs)) return aff_mul(b->lhs, *k * c);
    return bin(AffineBinOp::Mul, std::move(a), aff_const(c));
}

AffineExprRef aff_floordiv(AffineExprRef a, int64_t c, const AffineExtents& extents) {
    if (c == 1) return a;
    if (c <= 0) return bin(AffineBinOp::FloorDiv, std::move(a), aff_const(c));
    if (auto ka = const_val(a)) return aff_const(*ka / c);

    // Within the loop domain the quotient is always 0.
    auto range = aff_range(a, extents);
    if (range.lo >= 0 && range.hi < c) return aff_const(0);

    if (auto b = std::get_if<AffineBin>(&a->expr)) {
        // `x * c floordiv c → x`, and more generally peel a factor off an even multiple.
        if (b->op == AffineBinOp::Mul)
            if (auto k = const_val(b->rhs); k && *k % c == 0) return aff_mul(b->lhs, *k / c);

        // `(hi + lo) floordiv c → hi floordiv c` when `c` divides `hi` and `lo` cannot carry into it.
        // This is what makes a delinearize-of-linearize round trip collapse back to the plain loop dim.
        if (b->op == AffineBinOp::Add) {
            for (auto [even, rest] : {
                     std::pair{b->lhs, b->rhs},
                     std::pair{b->rhs, b->lhs}
            }) {
                if (!divides(*even, c)) continue;
                auto rest_range = aff_range(rest, extents);
                if (rest_range.lo >= 0 && rest_range.hi < c) return aff_floordiv(even, c, extents);
            }
        }
    }
    return bin(AffineBinOp::FloorDiv, std::move(a), aff_const(c));
}

AffineExprRef aff_ceildiv(AffineExprRef a, int64_t c, const AffineExtents& extents) {
    if (c == 1) return a;
    if (c > 0)
        if (auto ka = const_val(a)) return aff_const((*ka + c - 1) / c);
    return bin(AffineBinOp::CeilDiv, std::move(a), aff_const(c));
}

AffineExprRef aff_mod(AffineExprRef a, int64_t c, const AffineExtents& extents) {
    if (c == 1) return aff_const(0);
    if (c <= 0) return bin(AffineBinOp::Mod, std::move(a), aff_const(c));
    if (auto ka = const_val(a)) return aff_const(((*ka % c) + c) % c);

    // The loop never reaches the modulus, so the wrap never happens. `%tensor.repeat` relies on this: its read map
    // is `o#d mod s_in#d`, and dropping the `mod` is what keeps `d` recoverable for `linalg.generic`.
    auto range = aff_range(a, extents);
    if (range.lo >= 0 && range.hi < c) return a;

    // An even multiple of the modulus always leaves remainder 0.
    if (divides(*a, c)) return aff_const(0);

    if (auto b = std::get_if<AffineBin>(&a->expr); b && b->op == AffineBinOp::Add) {
        // Drop whichever addend is a multiple of `c`: `(x * c + y) mod c → y mod c`.
        for (auto [even, rest] : {
                 std::pair{b->lhs, b->rhs},
                 std::pair{b->rhs, b->lhs}
        })
            if (divides(*even, c)) return aff_mod(rest, c, extents);
    }
    return bin(AffineBinOp::Mod, std::move(a), aff_const(c));
}

// ----- translation from %affine -----

namespace {

/// Reads the literal extents of a `«n; Nat»` shape operand.
std::optional<std::vector<int64_t>> shape_lits(const Def* s) {
    auto n = Lit::isa(s->arity());
    if (!n) return {};
    std::vector<int64_t> out;
    for (size_t i = 0; i < *n; ++i) {
        auto e = Lit::isa(s->proj(*n, i));
        if (!e) return {};
        out.push_back(static_cast<int64_t>(*e));
    }
    return out;
}

/// Axis @p idx of `%affine.delinearize (lin, s)`, which is `(lin floordiv ∏_{j>idx} s#j) mod s#idx`.
AffineExprRef
delinearize_axis(const App* dl, u64 idx, const std::vector<const Def*>& params, const AffineExtents& extents) {
    auto [lin, s] = dl->args<2>();
    auto sizes    = shape_lits(s);
    if (!sizes || idx >= sizes->size()) return {};

    auto lin_expr = affine_expr(lin, params, extents);
    if (!lin_expr) return {};

    int64_t stride = 1;
    for (size_t j = idx + 1; j < sizes->size(); ++j)
        stride *= (*sizes)[j];

    auto res = aff_floordiv(lin_expr, stride, extents);
    // The leading axis needs no wrap: the linear index never reaches its extent times the stride.
    return idx == 0 ? res : aff_mod(res, (*sizes)[idx], extents);
}

} // namespace

AffineExprRef affine_expr(const Def* def, const std::vector<const Def*>& params, const AffineExtents& extents) {
    for (size_t j = 0; j < params.size(); ++j)
        if (def == params[j]) return aff_dim(j);

    if (auto lit = Lit::isa(def)) return aff_const(static_cast<int64_t>(*lit));

    // `%affine.delinearize` yields a tuple, so one axis normally arrives as an Extract of it.
    if (auto ex = def->isa<Extract>()) {
        auto idx = Lit::isa(ex->index());
        if (!idx) return {};
        if (auto dl = Axm::isa<plug::affine::delinearize>(ex->tuple()))
            return delinearize_axis(dl, *idx, params, extents);
        return {};
    }

    auto app = def->isa<App>();
    if (!app) return {};

    // A rank-1 delinearize is its own single axis: projecting an arity-1 def is the identity, so no Extract is
    // interposed. This is the shape of `%tensor.reshape`'s read map whenever the input is rank 1.
    if (auto dl = Axm::isa<plug::affine::delinearize>(app)) return delinearize_axis(dl, 0, params, extents);

    if (auto c = Axm::isa<plug::affine::constant>(app)) {
        auto lit = Lit::isa(c->arg());
        return lit ? aff_const(static_cast<int64_t>(*lit)) : nullptr;
    }

    if (auto o = Axm::isa<plug::affine::op>(app)) {
        if (o.id() == plug::affine::op::neg) {
            auto a = affine_expr(o->arg(), params, extents);
            return a ? aff_sub(aff_const(0), a) : nullptr;
        }

        auto [x, y] = o->args<2>();
        auto a      = affine_expr(x, params, extents);
        auto b      = affine_expr(y, params, extents);
        if (!a || !b) return {};

        switch (o.id()) {
            case plug::affine::op::add: return aff_add(a, b);
            case plug::affine::op::sub: return aff_sub(a, b);
            default: return {}; // %affine.op.mul is a lam, so it never shows up as an Axm App
        }
    }

    if (auto o = Axm::isa<plug::affine::semiop>(app)) {
        auto [x, c] = o->args<2>();
        auto a      = affine_expr(x, params, extents);
        auto c_lit  = Lit::isa(c);
        if (!a || !c_lit) return {};

        auto k = static_cast<int64_t>(*c_lit);
        switch (o.id()) {
            case plug::affine::semiop::mul: return aff_mul(a, k);
            case plug::affine::semiop::floordiv: return aff_floordiv(a, k, extents);
            case plug::affine::semiop::ceildiv: return aff_ceildiv(a, k, extents);
            case plug::affine::semiop::mod: return aff_mod(a, k, extents);
            default: return {};
        }
    }

    // `%affine.linearize (idxs, s) = Σ_i idxs#i · ∏_{j>i} s#j` (row-major).
    if (auto ln = Axm::isa<plug::affine::linearize>(app)) {
        auto [idxs, s] = ln->args<2>();
        auto sizes     = shape_lits(s);
        if (!sizes) return {};

        auto sum       = aff_const(0);
        int64_t stride = 1;
        for (size_t i = sizes->size(); i-- > 0;) {
            auto term = affine_expr(idxs->proj(sizes->size(), i), params, extents);
            if (!term) return {};

            // An extent-1 loop contributes nothing. Dropping those terms is what lets a reshape that only inserts
            // unit axes (`«c»` → `«1, c, 1, 1»`, how conv biases are broadcast) come out as the plain `d` it is,
            // instead of a sum over dims that are pinned to 0 anyway.
            auto range = aff_range(term, extents);
            if (range.lo != 0 || range.hi != 0) sum = aff_add(sum, aff_mul(term, stride));
            stride *= (*sizes)[i];
        }
        return sum;
    }

    return {};
}

} // namespace mim::mlir_be
