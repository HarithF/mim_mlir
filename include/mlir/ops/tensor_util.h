#pragma once
#include <cmath>
#include <cstring>

#include <functional>

#include <mim/def.h>
#include <mim/tuple.h>

#include "mlir/region_tree.h"

namespace mim::mlir_be {

inline std::string format_mlir_float(double v) {
    if (std::isnan(v)) return "0x7FC00000";
    if (std::isinf(v)) return v > 0 ? "0x7F800000" : "0xFF800000";
    return (v == std::floor(v)) ? std::format("{:.1f}", v) : std::format("{}", v);
}

inline std::string format_lit(uint64_t raw, const MLIRType& elem_type) {
    if (std::holds_alternative<MLIRFloatType>(elem_type)) {
        double val;
        // Mim's parser stores a bare-integer literal ascribed to a float type
        // (e.g. `1: F32`) as the raw integer value rather than the IEEE bit
        // pattern of that floating value (which requires `1.0: F32`). This
        // produces extremely small subnormal magnitudes that essentially never
        // occur as legitimate constants in real programs. Detect via the f64
        // exponent and correct.

        std::memcpy(&val, &raw, sizeof(val));
        if ((raw & 0x7FF0000000000000ull) == 0 && raw != 0) val = static_cast<double>(raw);
        return format_mlir_float(val);
    }
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
                double v = vals[flat_idx++];
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

} // namespace mim::mlir_be
