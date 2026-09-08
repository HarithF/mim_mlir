#pragma once
#include "mlir/printer.h"
#include "mlir/region_tree.h"

namespace mim::mlir_be {

class MathUnaryOp : public MLIROp {
public:
    enum class Kind {
        Tanh,
        Sin,
        Cos,
        Exp,
        Exp2,
        Log,
        Log2,
        Log10,
        Sqrt,
        Cbrt,
        Abs,
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
            case Kind::Exp2: return "math.exp2";
            case Kind::Log: return "math.log";
            case Kind::Log2: return "math.log2";
            case Kind::Log10: return "math.log10";
            case Kind::Sqrt: return "math.sqrt";
            case Kind::Cbrt: return "math.cbrt";
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
