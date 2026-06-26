#include "mim/world.h"

#include "mim/plug/mlir/mlir.h"

namespace mim::plug::mlir {

const Def* normalize_const(const Def* type, const Def*, const Def* arg) {
    auto& world = type->world();
    return world.lit(world.type_idx(arg), 42);
}

MIM_mlir_NORMALIZER_IMPL

} // namespace mim::plug::mlir
