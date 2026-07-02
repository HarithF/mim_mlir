#pragma once
#include <optional>
#include <ostream>
#include <set>
#include <string>

#include <mim/def.h>
#include <mim/lam.h>
#include <mim/world.h>

#include "mlir/lam_classifier.h"
#include "mlir/ops/arith.h"
#include "mlir/ops/func.h"
#include "mlir/ops/linalg.h"
#include "mlir/ops/math.h"
#include "mlir/ops/memref.h"
#include "mlir/ops/scf.h"
#include "mlir/printer.h"
#include "mlir/region_tree.h"
#include "mlir/type_converter.h"

namespace mim::mlir_be {

class MLIREmitter {
public:
    MLIREmitter(World& world, std::ostream& os)
        : world_(world)
        , os_(os)
        , types_(world)
        , clf_(world) {}

    void run();

private:
    // ===== mlir_emitter_core.cpp =====
    void emit_func(Lam* lam, MLIRBlock& into);

    // ------ leaf def emitter ---------
    MLIRValue emit_def(const Def* def, MLIRBlock& into);
    MLIRValue get_or_emit(const Def* def, MLIRBlock& into);

    // Scalar arithmetic dispatch
    std::optional<MLIRValue> try_emit_arith(const App* app, MLIRBlock& into);
    std::optional<MLIRValue> try_emit_select(const Extract* ex, MLIRBlock& into);

    // ----- helpers  ---------
    std::string fresh_name(const Def* def);
    std::string fresh_name(std::string prefix);
    bool is_return_callee(const Def* c, const Def* ret_var);
    MLIRValue wrap_as_tensor(const Def* input, MLIRValue in_val, MLIRBlock& into);
    double lit_to_double(const Lit* lit);

    //  -------arg seeding -----------
    void seed_dom_op(const Def* op, std::vector<MLIRValue>& args);
    void seed_var_tree(const Def* d);

    // ===== mlir_emitter_control_flow.cpp =====
    void emit_body(Lam* lam, MLIRBlock& into);
    void emit_affine_for(const App* app, MLIRBlock& into);
    void emit_for_body(Lam* body_lam, MLIRBlock& body_bb);

    std::optional<std::tuple<MLIRValue, Lam*, Lam*>> detect_cond_branch(const App* app, MLIRBlock& into);
    std::optional<std::vector<MLIRValue>> try_emit_cond_branch(const App* app, MLIRBlock& into);
    std::vector<MLIRValue> emit_scf_if(MLIRValue cond, Lam* then_lam, Lam* else_lam, MLIRBlock& into);
    std::vector<MLIRValue> emit_branch_values(Lam* lam, MLIRBlock& into);

    // ===== mlir_emitter_linalg.cpp =====

    std::optional<MLIRValue> try_emit_tensor_op(const App* app, MLIRBlock& into);
    void emit_linalg_generic(const App* mr_app, MLIRBlock& into);
    void emit_linalg_body(Lam* body_lam, MLIRBlock& body_bb);

    World& world_;
    std::ostream& os_;
    TypeConverter types_;
    LamClassifier clf_;

    DefMap<MLIRValue> values_;

    const Def* curr_ret_var_ = nullptr;
    int name_counter_        = 0;
};

} // namespace mim::mlir_be
