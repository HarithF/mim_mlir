#pragma once
#include <optional>
#include <utility>

#include <mim/def.h>
#include <mim/tuple.h>

#include "mlir/printer.h"
#include "mlir/region_tree.h"

namespace mim::mlir_be {

class SCFYieldOp : public MLIROp {
public:
    SCFYieldOp()
        : MLIROp({}, {}) {}

    explicit SCFYieldOp(std::vector<MLIRValue> vals)
        : MLIROp({}, std::move(vals)) {}

    void print(Printer& p) const override {
        if (operands_.empty()) {
            p.line("scf.yield");
            return;
        }
        std::string vals, types;
        for (auto& v : operands_) {
            vals += (vals.empty() ? "" : ", ") + v.name;
            types += (types.empty() ? "" : ", ") + print_type(v.type);
        }
        p.line("scf.yield {} : {}", vals, types);
    }
};

class SCFForOp : public MLIROp {
public:
    // Mode (1): no iter args
    SCFForOp(MLIRValue iv, MLIRValue lb, MLIRValue ub, MLIRValue step)
        : MLIROp({}, {std::move(lb), std::move(ub), std::move(step)})
        , iv_(std::move(iv)) {
        body_.entry().args.push_back(iv_);
    }

    // Mode (2): iter args + results
    SCFForOp(std::vector<MLIRValue> results,
             MLIRValue iv,
             MLIRValue lb,
             MLIRValue ub,
             MLIRValue step,
             std::vector<MLIRValue> iter_args_bb,
             std::vector<MLIRValue> iter_args_init)
        : MLIROp(std::move(results),
                 [&] {
                     std::vector<MLIRValue> ops = {lb, ub, step};
                     for (auto& v : iter_args_init)
                         ops.push_back(v);
                     return ops;
                 }())
        , iv_(std::move(iv))
        , iter_args_(std::move(iter_args_bb)) {
        body_.entry().args.push_back(iv_);
        for (auto& a : iter_args_)
            body_.entry().args.push_back(a);
    }

    MLIRRegion& body() { return body_; }

    void print(Printer& p) const override {
        std::string lhs;
        for (auto& r : results_)
            lhs += (lhs.empty() ? "" : ", ") + r.name;
        if (!lhs.empty()) lhs += " = ";

        std::string iter_str;
        for (size_t i = 0; i < iter_args_.size(); ++i)
            iter_str += (iter_str.empty() ? "" : ", ") + iter_args_[i].name + " = " + operands_[3 + i].name;

        // return-type suffix: "-> (T0, T1)"
        std::string ret_types;
        for (auto& r : results_)
            ret_types += (ret_types.empty() ? "" : ", ") + print_type(r.type);

        if (iter_str.empty()) {
            p.line("{}scf.for {} = {} to {} step {} {{", lhs, iv_.name, operands_[0].name, operands_[1].name,
                   operands_[2].name);
        } else {
            p.line("{}scf.for {} = {} to {} step {} iter_args({}) -> ({}) {{", lhs, iv_.name, operands_[0].name,
                   operands_[1].name, operands_[2].name, iter_str, ret_types);
        }
        p.indent();
        p.print_region(body_);
        p.dedent();
        p.line("}}");
    }

private:
    MLIRValue iv_;
    std::vector<MLIRValue> iter_args_;
    MLIRRegion body_;
};

class SCFIfOp : public MLIROp {
public:
    SCFIfOp(std::vector<MLIRValue> results, MLIRValue cond, MLIRBlock then_block, MLIRBlock else_block)
        : MLIROp(std::move(results), {std::move(cond)})
        , then_block_(std::move(then_block))
        , else_block_(std::move(else_block)) {}

    void print(Printer& p) const override {
        std::string lhs;
        for (auto& r : results_)
            lhs += (lhs.empty() ? "" : ", ") + r.name;
        if (!lhs.empty()) lhs += " = ";

        std::string ret_types;
        for (auto& r : results_)
            ret_types += (ret_types.empty() ? "" : ", ") + print_type(r.type);

        if (ret_types.empty())
            p.line("{}scf.if {} {{", lhs, operands_[0].name);
        else
            p.line("{}scf.if {} -> ({}) {{", lhs, operands_[0].name, ret_types);

        p.indent();
        for (auto& op : then_block_.ops)
            op->print(p);
        p.dedent();
        p.line("}} else {{");
        p.indent();
        for (auto& op : else_block_.ops)
            op->print(p);
        p.dedent();
        p.line("}}");
    }

private:
    MLIRBlock then_block_, else_block_;
};

inline std::optional<std::pair<const Def*, const Def*>> select_tuple_as_bool(const Extract* ex) {
    auto* tup = ex->tuple()->isa<Tuple>();
    if (!tup || tup->num_ops() != 2) return std::nullopt;
    return std::make_pair(tup->op(0), tup->op(1)); // (false_val, true_val)
}

} // namespace mim::mlir_be
