#include "mim/plug/mlir/mlir.h"

#include <fstream>
#include <string>

#include <mim/config.h>
#include <mim/pass.h>
#include <mim/plugin.h>

#include "mlir/mlir_emitter.h"

namespace mim::mlir_be {

using namespace std::string_literals;

/// Pipeline phase for `%mlir.emit`.
/// Writes the MLIR of the parcially optimized world to `<world>.mlir` or `a.mlir`.
class Emit : public Phase {
public:
    Emit(World& world, flags_t annex)
        : Phase(world, annex) {}

    void start() override {
        auto name = world().name() ? std::string(world().name().view()) : "a"s;
        auto ofs  = std::ofstream(name + ".mlir"s);
        MLIREmitter emitter{world(), ofs};
        emitter.run();
    }
};

} // namespace mim::mlir_be

using namespace mim;

/// Registers normalizers as well as Phase%s and Pass%es for the Axm%s of this Plugin.
static void reg_stages(Flags2Stages& stages) { Stage::hook<plug::mlir::emit, mlir_be::Emit>(stages); }
extern "C" MIM_EXPORT Plugin mim_get_plugin() { return {"mlir", MIM_VERSION, nullptr, reg_stages}; }
