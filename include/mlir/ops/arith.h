#pragma once
#include <cmath>

#include <string_view>

#include "mlir/printer.h"
#include "mlir/region_tree.h"

namespace mim::mlir_be {

class ConstantOp : public MLIROp {
public:
    ConstantOp(MLIRValue result, MLIRAttr value)
        : MLIROp({std::move(result)}, {})
        , value_(std::move(value)) {}

    void print(Printer& p) const override { p.line("{} = arith.constant {}", results_[0].name, print_attr(value_)); }

private:
    MLIRAttr value_;
};

class BinaryIntOp : public MLIROp {
public:
    enum class Kind {
        Add,
        Sub,
        Mul,
        DivS,
        DivU,
        RemS,
        RemU,
        And,
        Or,
        Xor,
        Shl,
        ShrS,
        ShrU,
    };

    BinaryIntOp(MLIRValue result, Kind kind, MLIRValue lhs, MLIRValue rhs)
        : MLIROp({std::move(result)}, {std::move(lhs), std::move(rhs)})
        , kind_(kind) {}

    void print(Printer& p) const override {
        p.line("{} = arith.{} {}, {} : {}", results_[0].name, mnemonic(kind_), operands_[0].name, operands_[1].name,
               print_type(results_[0].type));
    }

private:
    static std::string_view mnemonic(Kind k) {
        switch (k) {
            case Kind::Add: return "addi";
            case Kind::Sub: return "subi";
            case Kind::Mul: return "muli";
            case Kind::DivS: return "divsi";
            case Kind::DivU: return "divui";
            case Kind::RemS: return "remsi";
            case Kind::RemU: return "remui";
            case Kind::And: return "andi";
            case Kind::Or: return "ori";
            case Kind::Xor: return "xori";
            case Kind::Shl: return "shli";
            case Kind::ShrS: return "shrsi";
            case Kind::ShrU: return "shrui";
        }
        return "?";
    }

    Kind kind_;
};

class BinaryFloatOp : public MLIROp {
public:
    enum class Kind {
        Add,
        Sub,
        Mul,
        Div,
        Rem,
        MaxNum,  // arith.maxnumf
        Maximum, // arith.maximumf  (IEEE 754)
        MinNum,  // arith.minnumf
        Minimum, // arith.minimumf  (IEEE 754)
    };

    BinaryFloatOp(MLIRValue result, Kind kind, MLIRValue lhs, MLIRValue rhs)
        : MLIROp({std::move(result)}, {std::move(lhs), std::move(rhs)})
        , kind_(kind) {}

    void print(Printer& p) const override {
        p.line("{} = arith.{} {}, {} : {}", results_[0].name, mnemonic(kind_), operands_[0].name, operands_[1].name,
               print_type(results_[0].type));
    }

private:
    static std::string_view mnemonic(Kind k) {
        switch (k) {
            case Kind::Add: return "addf";
            case Kind::Sub: return "subf";
            case Kind::Mul: return "mulf";
            case Kind::Div: return "divf";
            case Kind::Rem: return "remf";
            case Kind::MaxNum: return "maxnumf";
            case Kind::Maximum: return "maximumf";
            case Kind::MinNum: return "minnumf";
            case Kind::Minimum: return "minimumf";
        }
        return "?";
    }

    Kind kind_;
};

class SelectOp : public MLIROp {
public:
    SelectOp(MLIRValue result, MLIRValue cond, MLIRValue true_val, MLIRValue false_val)
        : MLIROp({std::move(result)}, {std::move(cond), std::move(true_val), std::move(false_val)}) {}

    void print(Printer& p) const override {
        p.line("{} = arith.select {}, {}, {} : {}", results_[0].name, operands_[0].name, operands_[1].name,
               operands_[2].name, print_type(results_[0].type));
    }
};

class CmpiOp : public MLIROp {
public:
    enum class Pred {
        Eq,
        Ne,
        Slt,
        Sle,
        Sgt,
        Sge,
        Ult,
        Ule,
        Ugt,
        Uge,
    };

    CmpiOp(MLIRValue result, Pred pred, MLIRValue lhs, MLIRValue rhs)
        : MLIROp({std::move(result)}, {std::move(lhs), std::move(rhs)})
        , pred_(pred) {}

    // Trailing type is the OPERAND type, not the i1 result.
    void print(Printer& p) const override {
        p.line("{} = arith.cmpi {}, {}, {} : {}", results_[0].name, mnemonic(pred_), operands_[0].name,
               operands_[1].name, print_type(operands_[0].type));
    }

private:
    static std::string_view mnemonic(Pred pred) {
        switch (pred) {
            case Pred::Eq: return "eq";
            case Pred::Ne: return "ne";
            case Pred::Slt: return "slt";
            case Pred::Sle: return "sle";
            case Pred::Sgt: return "sgt";
            case Pred::Sge: return "sge";
            case Pred::Ult: return "ult";
            case Pred::Ule: return "ule";
            case Pred::Ugt: return "ugt";
            case Pred::Uge: return "uge";
        }
        return "?";
    }

    Pred pred_;
};

class CmpfOp : public MLIROp {
public:
    enum class Pred { Oeq, One, Olt, Ole, Ogt, Oge, Ult, Ule, Ugt, Uge, Une, Ueq, Ord, Uno };
    CmpfOp(MLIRValue result, Pred pred, MLIRValue a, MLIRValue b)
        : MLIROp({std::move(result)}, {std::move(a), std::move(b)})
        , pred_(pred) {}
    void print(Printer& p) const override {
        p.line("{} = arith.cmpf {}, {}, {} : {}", results_[0].name, mnemonic(pred_), operands_[0].name,
               operands_[1].name, print_type(operands_[0].type));
    }

private:
    static std::string_view mnemonic(Pred p) {
        switch (p) {
            case Pred::Oeq: return "oeq";
            case Pred::One: return "one";
            case Pred::Olt: return "olt";
            case Pred::Ole: return "ole";
            case Pred::Ogt: return "ogt";
            case Pred::Oge: return "oge";
            case Pred::Ult: return "ult";
            case Pred::Ule: return "ule";
            case Pred::Ugt: return "ugt";
            case Pred::Uge: return "uge";
            case Pred::Une: return "une";
            case Pred::Ueq: return "ueq";
            case Pred::Ord: return "ord";
            case Pred::Uno: return "uno";
        }
        return "?";
    }
    Pred pred_;
};

class IndexCastOp : public MLIROp {
public:
    IndexCastOp(MLIRValue result, MLIRValue operand)
        : MLIROp({std::move(result)}, {std::move(operand)}) {}

    void print(Printer& p) const override {
        p.line("{} = arith.index_cast {} : {} to {}", results_[0].name, operands_[0].name,
               print_type(operands_[0].type), print_type(results_[0].type));
    }
};

class DenseConstOp : public MLIROp {
public:
    DenseConstOp(MLIRValue result, std::string attr)
        : MLIROp({std::move(result)}, {})
        , attr_(std::move(attr)) {}

    void print(Printer& p) const override {
        p.line("{} = arith.constant {} : {}", results_[0].name, attr_, print_type(results_[0].type));
    }

private:
    std::string attr_;
};

} // namespace mim::mlir_be
