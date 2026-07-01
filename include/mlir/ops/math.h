#pragma once
#include "mlir/printer.h"
#include "mlir/region_tree.h"

namespace mim::mlir_be {

class MathUnaryOp : public MLIROp {
public:
    enum class Kind {
        Tanh, // math.tanh
        Sin,  // math.sin
        Cos,  // math.cos
        Exp,  // math.exp
        Log,  // math.log
        Sqrt, // math.sqrt
        Abs,  // math.absf
    };

    MathUnaryOp(MLIRValue result, Kind kind, MLIRValue operand)
        : MLIROp({std::move(result)}, {std::move(operand)})
        , kind_(kind) {}

    void print(Printer& p) const override {
        p.line("{} = {} {} : {}", results_[0].name, mnemonic(kind_), operands_[0].name, print_type(results_[0].type));
    }

private:
    static std::string_view mnemonic(Kind k) {
        switch (k) {
            case Kind::Tanh: return "math.tanh";
            case Kind::Sin: return "math.sin";
            case Kind::Cos: return "math.cos";
            case Kind::Exp: return "math.exp";
            case Kind::Log: return "math.log";
            case Kind::Sqrt: return "math.sqrt";
            case Kind::Abs: return "math.absf";
        }
        return "?";
    }

    Kind kind_;
};

class MathIsFiniteOp : public MLIROp {
public:
    MathIsFiniteOp(MLIRValue result, MLIRValue operand)
        : MLIROp({std::move(result)}, {std::move(operand)}) {}
    void print(Printer& p) const override {
        p.line("{} = math.isfinite {} : {}", results_[0].name, operands_[0].name, print_type(operands_[0].type));
    }
};

} // namespace mim::mlir_be
