
#pragma once
#include <mim/plug/affine/affine.h>
#include <mim/plug/affine/autogen.h>
#include <mim/plug/core/core.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>
#include <mim/plug/tensor/autogen.h>
#include <mim/plug/tensor/tensor.h>

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

double MLIREmitter::lit_to_double(const Lit* lit) {
    auto mlir_type = types_.convert(lit->type());
    auto& ft       = std::get<MLIRFloatType>(mlir_type);
    uint64_t raw   = lit->get<u64>();

    if (ft.bits == 16) {
        assert(false && "f16 literal-to-double not yet implemented");
        return 0.0;
    }

    double val;
    std::memcpy(&val, &raw, sizeof(val));

    // Mim's parser stores a bare-integer literal ascribed to a float type
    // (e.g. `1: F32`) as the raw integer value rather than the IEEE bit
    // pattern of that floating value (which requires `1.0: F32`). This
    // produces extremely small subnormal magnitudes that essentially never
    // occur as legitimate constants in real programs. Detect via the f64
    // exponent and correct.
    uint64_t exp_bits         = raw & 0x7FF0000000000000ull;
    bool is_subnormal_nonzero = (exp_bits == 0) && (raw != 0);
    if (is_subnormal_nonzero) return static_cast<double>(raw); // reinterpret as the intended integer value

    return val;
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
} // namespace mim::mlir_be
