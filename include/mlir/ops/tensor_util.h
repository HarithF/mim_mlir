#pragma once
#include <cmath>
#include <cstddef>
#include <cstring>

#include <functional>

#include <mim/def.h>
#include <mim/lam.h>
#include <mim/tuple.h>

#include <mim/plug/affine/affine.h>
#include <mim/plug/affine/autogen.h>

#include "mlir/ops/affine.h"
#include "mlir/region_tree.h"

namespace mim::mlir_be {

inline std::string format_mlir_float(double v) {
    if (std::isnan(v)) return "0x7FC00000";
    if (std::isinf(v)) return v > 0 ? "0x7F800000" : "0xFF800000";
    return (v == std::floor(v)) ? std::format("{:.1f}", v) : std::format("{}", v);
}

inline double lit_to_double(uint64_t raw, uint32_t bits) {
    if (bits == 32) {
        uint32_t b32 = static_cast<uint32_t>(raw);
        float f;
        std::memcpy(&f, &b32, sizeof(f));
        return static_cast<double>(f);
    }
    double d;
    std::memcpy(&d, &raw, sizeof(d));
    return d;
}

inline std::string format_lit(uint64_t raw, const MLIRType& elem_type) {
    if (auto ft = std::get_if<MLIRFloatType>(&elem_type)) return format_mlir_float(lit_to_double(raw, ft->bits));
    return std::to_string(static_cast<int64_t>(raw));
}

// Recursively collect every leaf Lit payload from a Mim Tuple/Pack tensor Def
// into a flat row-major vector of raw u64 values.
inline void collect_lit_tensor(const mim::Def* d, std::vector<uint64_t>& out) {
    if (auto lit = d->isa<mim::Lit>()) {
        out.push_back(lit->get<u64>());
        return;
    }
    if (auto tup = d->isa<mim::Tuple>()) {
        for (size_t i = 0; i < tup->num_ops(); ++i)
            collect_lit_tensor(tup->op(i), out);
        return;
    }
    if (auto pack = d->isa<mim::Pack>()) {
        if (auto n = mim::Lit::isa(pack->arity())) {
            for (size_t i = 0; i < *n; ++i)
                collect_lit_tensor(pack->body(), out);
            return;
        }
    }
    assert(false && "unexpected node in literal tensor");
}

inline std::string make_dense_attr(const std::vector<uint64_t>& vals, const MLIRTensorType& tt) {
    const MLIRType& elem = tt.elem->type;
    std::vector<size_t> dims;
    for (auto& d : tt.shape)
        dims.push_back(d ? static_cast<size_t>(*d) : 0);

    std::function<std::string(size_t, size_t&)> emit_nested;
    emit_nested = [&](size_t dim_idx, size_t& flat_idx) -> std::string {
        if (dim_idx == dims.size() - 1) {
            std::string s = "[";
            for (size_t i = 0; i < dims[dim_idx]; ++i) {
                if (i) s += ", ";
                // print as integer if whole number, float otherwise
                uint64_t v = vals[flat_idx++];
                s += format_lit(v, elem);
            }
            return s + "]";
        }
        std::string s = "[";
        for (size_t i = 0; i < dims[dim_idx]; ++i) {
            if (i) s += ", ";
            s += emit_nested(dim_idx + 1, flat_idx);
        }
        return s + "]";
    };

    size_t idx = 0;
    return "dense<" + emit_nested(0, idx) + ">";
}

inline std::string make_dense_splat(uint64_t raw, const MLIRTensorType& tt) {
    return std::format("dense<{}>", format_lit(raw, tt.elem->type));
}

inline bool is_affine_mod(const Def* d, const Def*& value, const Def*& modulus) {
    auto app = d->isa<App>();
    if (!app) return false;

    auto semiop = Axm::isa<plug::affine::semiop>(app);
    if (!semiop || semiop.id() != plug::affine::semiop::mod) return false;
    std::tie(value, modulus) = app->arg()->projs<2>();
    return true;
}

inline std::optional<size_t> find_driving_param(const Def* d, const std::vector<const Def*>& params) {
    for (size_t j = 0; j < params.size(); ++j)
        if (d == params[j]) return j;

    const Def *value, *modulus;
    if (is_affine_mod(d, value, modulus)) {
        if (auto m = Lit::isa(modulus); m && *m == 1) return std::nullopt; // degenerate: always 0
        return find_driving_param(value, params);
    }

    std::optional<size_t> found;
    for (auto op : d->ops()) {
        auto sub = find_driving_param(op, params);
        if (!sub) continue;
        if (found && *found != *sub) return std::nullopt; // ambiguous: mixes >1 param
        found = sub;
    }
    return found;
}

struct AffineMapInfo {
    std::string str;
    /// Loop dims this map exposes as a bare `d<j>`, i.e. not wrapped in any arithmetic.
    /// `linalg.generic` recovers the loop nest by inverting the concatenated maps, so a dim that only ever occurs
    /// inside an expression (`d2 * 2 + d4`) does not count as recovered.
    std::vector<size_t> bare_dims;
};

/// Renders a `%tensor.map_reduce` access lam as an MLIR `affine_map` over @p total_loops loop dims.
/// @p loop_extents bounds those dims (see AffineExtents); it lets the affine folder discard the `mod`/`floordiv`
/// terms that the loop domain makes redundant.
inline AffineMapInfo lam_to_affine_map(Lam* lam, size_t total_loops, const AffineExtents& loop_extents) {
    assert(lam && lam->is_set());

    // infer actual param count
    size_t actual_params = 0;
    auto dom             = lam->type()->dom();
    if (auto sigma = dom->isa<Sigma>())
        actual_params = sigma->num_ops();
    else if (auto arr = dom->isa<Arr>())
        actual_params = Lit::isa(arr->arity()) ? *Lit::isa(arr->arity()) : 0;
    else
        actual_params = 1;

    // dim string uses total_loops
    std::string dim_str;
    for (size_t i = 0; i < total_loops; ++i)
        dim_str += (i ? ", " : "") + std::format("d{}", i);

    // collect actual params using actual_params
    std::vector<const Def*> params;
    if (actual_params == 1)
        params.push_back(lam->var());
    else
        for (size_t i = 0; i < actual_params; ++i)
            params.push_back(lam->var()->proj(actual_params, i));

    const Def* result_def = lam->body();
    assert(result_def);

    // One output position per input axis; a lam returning a bare index has exactly one.
    std::vector<const Def*> results;
    if (auto sigma = result_def->type()->isa<Sigma>()) {
        size_t n = sigma->num_ops();
        for (size_t i = 0; i < n; ++i)
            results.push_back(result_def->proj(n, i));
    } else if (auto arr = result_def->type()->isa<Arr>()) {
        if (auto n = Lit::isa(arr->arity()))
            for (size_t i = 0; i < *n; ++i)
                results.push_back(result_def->proj(*n, i));
    } else {
        results.push_back(result_def);
    }

    std::string result_str;
    std::vector<size_t> bare_dims;
    for (size_t i = 0; i < results.size(); ++i) {
        if (i) result_str += ", ";

        if (auto e = affine_expr(results[i], params, loop_extents)) {
            result_str += e->str();
            if (auto d = std::get_if<AffineDim>(&e->expr)) bare_dims.push_back(d->pos);
            continue;
        }

        // Outside the affine grammar. Fall back to the old "which single loop var drives this position" guess, which
        // is right for pure projections and broadcasts but silently wrong for anything with real index arithmetic —
        // so say so rather than emitting a plausible-looking map.
        std::cerr << "mlir: cannot render access map position " << i << " of '" << lam->sym().str()
                  << "' as an affine expression; falling back to a driving-parameter guess\n";
        if (auto j = find_driving_param(results[i], params)) {
            result_str += std::format("d{}", *j);
            bare_dims.push_back(*j);
        } else {
            result_str += "0";
        }
    }

    return {std::format("affine_map<({}) -> ({})>", dim_str, result_str), std::move(bare_dims)};
}

} // namespace mim::mlir_be
