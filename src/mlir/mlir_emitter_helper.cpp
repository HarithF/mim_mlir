#include <mim/plug/affine/affine.h>
#include <mim/plug/affine/autogen.h>
#include <mim/plug/core/core.h>
#include <mim/plug/math/math.h>
#include <mim/plug/mem/mem.h>
#include <mim/plug/tensor/autogen.h>
#include <mim/plug/tensor/tensor.h>

#include "mlir/mlir_emitter.h"
#include "mlir/ops/tensor_util.h"

namespace mim::mlir_be {

// ----- Helpers -------
/// Memoized per Def: call sites derive several SSA names from one Def (`%x`, `%x.buf`, `%x.buf.out`) and rely on
/// the base staying stable across calls.
/// Distinct Defs never share a name: Mim shadows freely (`let a = …; let a = …`), so the same Sym can name many
/// Defs, and MLIR requires one definition per SSA name.
std::string MLIREmitter::fresh_name(const Def* def) {
    if (auto it = names_.find(def); it != names_.end()) return it->second;

    std::string name;
    auto sym = def->sym().str();
    if (!sym.empty() && sym[0] != '_') {
        name = "%" + std::string(sym);
        for (int i = 1; !used_names_.insert(name).second; ++i) name = std::format("%{}_{}", sym, i);
    } else {
        // The counter already makes these unique.
        name = std::format("%v{}", name_counter_++);
        used_names_.insert(name);
    }

    names_[def] = name;
    return name;
}
std::string MLIREmitter::fresh_name(std::string prefix) { return prefix + std::to_string(name_counter_++); }

bool MLIREmitter::is_return_callee(const Def* c, const Def* ret_var) {
    if (c == ret_var) return true;
    if (auto ex = c->isa<Extract>()) return ex->sym().str() == "return";
    return false;
}

MLIRValue MLIREmitter::wrap_as_tensor(const Def* input, MLIRValue in_val, MLIRBlock& into) {
    MLIRTensorType scalar_tensor_t;
    scalar_tensor_t.shape = {}; // rank 0
    scalar_tensor_t.elem  = std::make_shared<MLIRTypeNode>(in_val.type);
    MLIRType wrapped_t{std::move(scalar_tensor_t)};

    auto lit = input->isa<Lit>();
    assert(lit && "scalar broadcast input must be a literal — non-literal scalar splat not yet handled");

    auto dense_str = make_dense_splat(lit->get<u64>(), std::get<MLIRTensorType>(wrapped_t));
    MLIRValue wrapped{fresh_name(input) + ".splat", wrapped_t};
    into.ops.emplace_back(std::make_unique<DenseConstOp>(wrapped, std::move(dense_str)));
    return wrapped;
}

MLIRValue MLIREmitter::get_or_emit(const Def* def, MLIRBlock& into) {
    if (auto it = values_.find(def); it != values_.end()) return it->second;
    auto val     = emit_def(def, into);
    values_[def] = val;
    return val;
}

// Recursively unpack op into individual MLIR args.
// Heterogeneous Sigma → recurse into each element.
// Arr → seed the whole tuple as a single tensor value; individual elements are
// pulled out lazily via `tensor.extract` in emit_def when the body needs them.
void MLIREmitter::seed_dom_op(const Def* op, std::vector<MLIRValue>& args) {
    if (Axm::isa<plug::mem::M>(op->type())) return;
    if (op->type()->isa<Pi>()) return;

    if (auto sigma = op->type()->isa<Sigma>()) {
        for (size_t i = 0; i < sigma->num_ops(); ++i)
            seed_dom_op(op->proj(sigma->num_ops(), i), args);
        return;
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
