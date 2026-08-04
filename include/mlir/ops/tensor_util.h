#pragma once
#include <cmath>
#include <cstddef>
#include <cstring>

#include <functional>

#include <mim/def.h>
#include <mim/lam.h>
#include <mim/tuple.h>

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

inline std ::string lam_to_affine_map(Lam* lam, size_t total_loops) {
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

    const Def* result_def = nullptr;
    if (auto* body_app = lam->body()->isa<App>()) {
        result_def = body_app->arg();
    } else {
        // direct style: body IS the result (e.g. lam with `as o` pattern)
        result_def = lam->body();
    }

    assert(result_def);

    std::string result_str;

    std::vector<const Def*> results;
    bool is_direct_param = false;
    for (size_t j = 0; j < params.size(); ++j) {
        if (result_def == params[j]) {
            result_str += std::format("d{}", j);
            is_direct_param = true;
            break;
        }
    }
    if (!is_direct_param) {
        // unpack as tuple/sigma/arr
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

        for (size_t i = 0; i < results.size(); ++i) {
            if (i) result_str += ", ";
            bool found = false;
            for (size_t j = 0; j < params.size(); ++j) {
                if (results[i] == params[j]) {
                    result_str += std::format("d{}", j);
                    found = true;
                    break;
                }
            }
            if (!found) result_str += "0"; // broadcast
        }
    }

    return std::format("affine_map<({}) -> ({})>", dim_str, result_str);
}

} // namespace mim::mlir_be
