#pragma once
#include "mlir/printer.h"
#include "mlir/region_tree.h"

namespace mim::mlir_be {

class TensorEmptyOp : public MLIROp {
public:
    TensorEmptyOp(MLIRValue result)
        : MLIROp({std::move(result)}, {}) {}

    void print(Printer& p) const override {
        p.line("{} = tensor.empty() : {}", results_[0].name, print_type(results_[0].type));
    }
};

/// `%r = tensor.extract %src[%i0, %i1, ...] : tensor<...>` where operands[0] is
/// the source tensor and the remaining operands are the (index-typed) indices.
class TensorExtractOp : public MLIROp {
public:
    TensorExtractOp(MLIRValue result, MLIRValue src, std::vector<MLIRValue> indices)
        : MLIROp({std::move(result)}, [&] {
            std::vector<MLIRValue> ops;
            ops.reserve(1 + indices.size());
            ops.push_back(std::move(src));
            for (auto& i : indices) ops.push_back(std::move(i));
            return ops;
        }()) {}

    void print(Printer& p) const override {
        std::string idx_list;
        for (size_t i = 1; i < operands_.size(); ++i) {
            if (i > 1) idx_list += ", ";
            idx_list += operands_[i].name;
        }
        p.line("{} = tensor.extract {}[{}] : {}", results_[0].name, operands_[0].name, idx_list,
               print_type(operands_[0].type));
    }
};

} // namespace mim::mlir_be
