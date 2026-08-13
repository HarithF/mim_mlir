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

/// `%r = tensor.collapse_shape %src [[0, 1], [2]] : tensor<…> into tensor<…>`.
/// Each reassociation group is merged into one output axis; an empty group list collapses to rank 0.
class TensorCollapseShapeOp : public MLIROp {
public:
    TensorCollapseShapeOp(MLIRValue result, MLIRValue src, std::vector<std::vector<int64_t>> reassoc)
        : MLIROp({std::move(result)}, {std::move(src)})
        , reassoc_(std::move(reassoc)) {}

    void print(Printer& p) const override {
        std::string groups;
        for (size_t g = 0; g < reassoc_.size(); ++g) {
            groups += (g ? ", [" : "[");
            for (size_t i = 0; i < reassoc_[g].size(); ++i) groups += (i ? ", " : "") + std::to_string(reassoc_[g][i]);
            groups += "]";
        }
        p.line("{} = tensor.collapse_shape {} [{}] : {} into {}", results_[0].name, operands_[0].name, groups,
               print_type(operands_[0].type), print_type(results_[0].type));
    }

private:
    std::vector<std::vector<int64_t>> reassoc_;
};

/// `%r = tensor.pad %src low[…] high[…] { ^bb0(…): tensor.yield %v } : tensor<…> to tensor<…>`.
/// operands[0] is the source tensor, operands[1] the scalar pad value.
/// The region is always the trivial constant-pad body, so it is printed inline instead of being modelled
/// as an MLIRRegion.
class TensorPadOp : public MLIROp {
public:
    TensorPadOp(MLIRValue result,
                MLIRValue src,
                MLIRValue value,
                std::vector<int64_t> low,
                std::vector<int64_t> high,
                std::vector<std::string> block_args)
        : MLIROp({std::move(result)}, {std::move(src), std::move(value)})
        , low_(std::move(low))
        , high_(std::move(high))
        , block_args_(std::move(block_args)) {}

    void print(Printer& p) const override {
        auto list = [](const std::vector<int64_t>& xs) {
            std::string s;
            for (size_t i = 0; i < xs.size(); ++i) s += (i ? ", " : "") + std::to_string(xs[i]);
            return s;
        };

        // One index-typed block arg per axis; unused here, but required by the op's region signature.
        std::string block_args;
        for (size_t i = 0; i < block_args_.size(); ++i) block_args += (i ? ", " : "") + block_args_[i] + ": index";

        p.line("{} = tensor.pad {} low[{}] high[{}] {{", results_[0].name, operands_[0].name, list(low_), list(high_));
        p.indent();
        p.line("^bb0({}):", block_args);
        p.indent();
        p.line("tensor.yield {} : {}", operands_[1].name, print_type(operands_[1].type));
        p.dedent();
        p.dedent();
        p.line("}} : {} to {}", print_type(operands_[0].type), print_type(results_[0].type));
    }

private:
    std::vector<int64_t> low_, high_;
    std::vector<std::string> block_args_;
};

} // namespace mim::mlir_be
