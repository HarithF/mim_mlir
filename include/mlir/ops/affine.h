#pragma once
#include <cstdint>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <mim/def.h>
#include <mim/lam.h>

#include "mlir/printer.h"
#include "mlir/region_tree.h"

namespace mim::mlir_be {

// ----- Affine expressions -----

/// A typed MLIR affine expression.
struct AffineExprNode;
using AffineExprRef = std::shared_ptr<const AffineExprNode>;

/// Loop dimension `d<pos>`.
struct AffineDim {
    size_t pos;
};

struct AffineConst {
    int64_t value;
};

/// MLIR's affine binary operators.
enum class AffineBinOp { Add, Sub, Mul, FloorDiv, CeilDiv, Mod };

struct AffineBin {
    AffineBinOp op;
    AffineExprRef lhs, rhs;
};

using AffineExpr = std::variant<AffineDim, AffineConst, AffineBin>;

/// Inclusive range of the values an expression can take, used to justify folds that only hold inside the loop
/// domain (`d mod n → d` when the loop over `d` never reaches `n`).
struct AffineRange {
    int64_t lo, hi;
};

struct AffineExprNode {
    AffineExpr expr;

    template<class T>
    AffineExprNode(T t)
        : expr(std::move(t)) {}

    /// Renders this expression in MLIR `affine_map` syntax, parenthesized only where precedence requires it.
    std::string str() const;

    /// Appends the position of every AffineDim occurring in this expression.
    void used_dims(std::vector<size_t>& out) const;
};

/// Bounds for the loop dimensions: `extents[j]` confines `d<j>` to `[0, *extents[j])`.
/// An absent entry means the bound is unknown, which widens AffineRange and so blocks the folds that rely on it.
using AffineExtents = std::vector<std::optional<int64_t>>;

/// @name Smart constructors
/// These fold as they build. Folding is not cosmetic: an unfolded `* 1` or `mod 1` leaves a loop dim unrecoverable
/// and makes `linalg.generic` reject the whole map.
///@{
AffineExprRef aff_dim(size_t pos);
AffineExprRef aff_const(int64_t value);
AffineExprRef aff_add(AffineExprRef a, AffineExprRef b);
AffineExprRef aff_sub(AffineExprRef a, AffineExprRef b);
AffineExprRef aff_mul(AffineExprRef a, int64_t c);
AffineExprRef aff_floordiv(AffineExprRef a, int64_t c, const AffineExtents& extents);
AffineExprRef aff_ceildiv(AffineExprRef a, int64_t c, const AffineExtents& extents);
AffineExprRef aff_mod(AffineExprRef a, int64_t c, const AffineExtents& extents);
///@}

/// Range of the values @p e can take given the loop @p extents.
AffineRange aff_range(const AffineExprRef& e, const AffineExtents& extents);

/// Translates the `%affine` index expression @p def into an AffineExpr.
AffineExprRef affine_expr(const Def* def, const std::vector<const Def*>& params, const AffineExtents& extents);

} // namespace mim::mlir_be
