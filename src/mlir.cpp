#include "mim/plug/mlir/mlir.h"

#include <mim/pass.h>
#include <mim/plugin.h>

using namespace mim;

/// Registers normalizers as well as Phase%s and Pass%es for the Axm%s of this Plugin.
extern "C" MIM_EXPORT Plugin mim_get_plugin() {
    return {"mlir", MIM_VERSION, plug::mlir::register_normalizers, nullptr};
}
