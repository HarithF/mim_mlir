#include <algorithm>
#include <functional>
#include <ranges>

#include <mim/lam.h>
#include <mim/tuple.h>

#include <mim/plug/mem/mem.h>
#include <mim/plug/tensor/autogen.h>
#include <mim/plug/tensor/tensor.h>

#include "mlir/mlir_emitter.h"
#include "mlir/ops/tensor_util.h"

namespace mim::mlir_be {

std::optional<MLIRValue> MLIREmitter::try_emit_tensor_op(const App* app, MLIRBlock& into) {
    auto* def = app;

    if (Axm::isa<plug::tensor::map_reduce_post>(app)) {
        emit_linalg_generic(app, into);
        return values_[def];
    }

    if (auto bc = Axm::isa<plug::tensor::broadcast>(app)) {
        // mirrors lower_broadcast extraction exactly
        auto [s_in, s_out, input] = bc->arg()->projs<3>();
        auto callee               = bc->callee()->as<App>();
        auto [T, r]               = callee->args<2>();

        auto r_lit = Lit::isa(r);
        assert(r_lit && "broadcast rank must be literal");
        auto r_nat = *r_lit;

        // derive broadcast dimensions: where s_in dim == 1
        std::vector<int64_t> bcast_dims;
        for (size_t i = 0; i < r_nat; ++i) {
            auto dim_in = s_in->proj(r_nat, i);
            if (auto lit = Lit::isa(dim_in); lit && *lit == 1) bcast_dims.push_back(static_cast<int64_t>(i));
        }

        auto in_val   = get_or_emit(input, into);
        auto out_type = types_.convert(def->type());

        if (!std::holds_alternative<MLIRTensorType>(in_val.type)) in_val = wrap_as_tensor(input, in_val, into);

        // `linalg.broadcast` treats `dimensions` as the *added* axes, so it wants the input without them
        // (input rank + dimensions == init rank). %tensor.broadcast instead preserves rank, and whether the size-1
        // axes are physically present depends on the source: `%tensor.broadcast ((1, 32), (4, 32), b)` may be given a
        // rank-1 `b: «32; T»`, or a rank-2 one that was %tensor.reshape'd to «1, 32; T» first. Collapse them away
        // when they are there.
        if (auto tt = std::get_if<MLIRTensorType>(&in_val.type);
            tt && tt->shape.size() == r_nat && !bcast_dims.empty()) {
            // One group per kept axis, absorbing the unit axes to its left; the last group also takes the trailing
            // ones. Groups must be contiguous, and each contributes `∏ extents` — i.e. the kept extent — to the result.
            std::vector<std::vector<int64_t>> reassoc;
            MLIRTensorType collapsed;
            collapsed.elem = tt->elem;
            for (size_t i = 0; i < r_nat; ++i) {
                if (reassoc.empty()) reassoc.emplace_back();
                reassoc.back().push_back(static_cast<int64_t>(i));
                // A kept axis closes its group, unless no kept axis follows.
                if (std::ranges::find(bcast_dims, static_cast<int64_t>(i)) == bcast_dims.end()) {
                    collapsed.shape.push_back(tt->shape[i]);
                    if (collapsed.shape.size() < r_nat - bcast_dims.size()) reassoc.emplace_back();
                }
            }
            // All axes are size 1: the whole thing collapses to a rank-0 tensor, spelled with no groups.
            if (collapsed.shape.empty()) reassoc.clear();

            MLIRValue collapsed_val{fresh_name(def) + ".collapsed", MLIRType{std::move(collapsed)}};
            into.ops.emplace_back(std::make_unique<TensorCollapseShapeOp>(collapsed_val, in_val, std::move(reassoc)));
            in_val = collapsed_val;
        }

        // tensor.empty for output buffer
        std::string buf_name = fresh_name(def) + ".buf";
        MLIRValue out_buf{buf_name, out_type};
        into.ops.emplace_back(std::make_unique<TensorEmptyOp>(out_buf));

        MLIRValue result{fresh_name(def), out_type};
        into.ops.emplace_back(std::make_unique<LinalgBroadcastOp>(result, in_val, out_buf, std::move(bcast_dims)));
        return result;
    }
    if (auto pd = Axm::isa<plug::tensor::pad>(app)) {
        // %tensor.pad @(T, r) s_in (mode, lo, hi) (input, value)
        auto [input, value] = pd->args<2>();
        auto* lohi_app      = pd->callee()->as<App>();
        auto [mode, lo, hi] = lohi_app->args<3>();
        auto [T, r]         = lohi_app->callee()->as<App>()->callee()->as<App>()->args<2>();

        auto mode_lit = Lit::isa(mode);
        assert(mode_lit && "pad mode must be literal");
        // `tensor.pad` fills the halo from its region, which cannot express a per-axis clamped read.
        assert(*mode_lit == 0 && "only constant padding (mode 0) is supported by the MLIR backend");

        auto r_lit = Lit::isa(r);
        assert(r_lit && "pad rank must be literal");
        auto r_nat = *r_lit;

        auto extents = [&](const Def* d) {
            std::vector<int64_t> xs;
            for (size_t i = 0; i < r_nat; ++i) {
                auto lit = Lit::isa(r_nat == 1 ? d : d->proj(r_nat, i));
                assert(lit && "pad amounts must be literal");
                xs.push_back(static_cast<int64_t>(*lit));
            }
            return xs;
        };

        auto in_val = get_or_emit(input, into);
        if (!std::holds_alternative<MLIRTensorType>(in_val.type)) in_val = wrap_as_tensor(input, in_val, into);
        auto val_val = get_or_emit(value, into);

        // The pad region is not isolated from above, so its block args live in the function's SSA namespace.
        std::vector<std::string> block_args;
        for (size_t i = 0; i < r_nat; ++i)
            block_args.push_back(fresh_name("%pad_i"));

        MLIRValue result{fresh_name(def), types_.convert(def->type())};
        into.ops.emplace_back(
            std::make_unique<TensorPadOp>(result, in_val, val_val, extents(lo), extents(hi), std::move(block_args)));
        return result;
    }

    if (auto cat = Axm::isa<plug::tensor::concat>(app)) {
        // %tensor.concat @(T, nis, r) ax @Sis is
        auto [TnisR, ax, Sis] = cat->callee()->as<App>()->uncurry_args<3>();
        auto [T, nis, r]      = TnisR->projs<3>();

        auto nis_lit = Lit::isa(nis);
        assert(nis_lit && "concat input count must be literal");
        auto ax_lit = Lit::isa(ax);
        assert(ax_lit && "concat axis must be literal");
        auto n_inputs = *nis_lit;

        // `tensor.concat` takes the inputs as a variadic operand list and derives the joined extent itself
        std::vector<MLIRValue> ins;
        for (size_t i = 0; i < n_inputs; ++i) {
            auto* in    = n_inputs == 1 ? app->arg() : app->arg()->proj(n_inputs, i);
            auto in_val = get_or_emit(in, into);
            if (!std::holds_alternative<MLIRTensorType>(in_val.type)) in_val = wrap_as_tensor(in, in_val, into);
            ins.push_back(std::move(in_val));
        }

        MLIRValue result{fresh_name(def), types_.convert(def->type())};
        into.ops.emplace_back(std::make_unique<TensorConcatOp>(result, std::move(ins), static_cast<int64_t>(*ax_lit)));
        return result;
    }

    if (auto get_ax = Axm::isa<plug::tensor::get>(app)) {
        // %tensor.get @(T, r, s) (arr, index)

        auto* arr_def = app->arg()->proj(2, 0);
        auto* idx_def = app->arg()->proj(2, 1);

        auto arr_val  = get_or_emit(arr_def, into);
        auto res_type = types_.convert(def->type());

        // unpack index tuple - each element is an Idx literal, cast to index
        std::vector<MLIRValue> indices;
        if (auto sigma = idx_def->type()->isa<Sigma>()) {
            size_t n = sigma->num_ops();
            for (size_t i = 0; i < n; ++i) {
                auto elem     = idx_def->proj(n, i);
                auto elem_val = get_or_emit(elem, into);
                indices.push_back(types_.to_index(elem_val, into, fresh_name("%idx")));
            }
        } else if (auto arr = idx_def->type()->isa<Arr>()) {
            if (auto n = Lit::isa(arr->arity())) {
                for (size_t i = 0; i < *n; ++i) {
                    auto elem_val = get_or_emit(idx_def->proj(*n, i), into);
                    indices.push_back(types_.to_index(elem_val, into, fresh_name("%idx")));
                }
            }
        } else {
            // rank-1: bare Idx
            auto elem_val = get_or_emit(idx_def, into);
            indices.push_back(types_.to_index(elem_val, into, fresh_name("%idx")));
        }

        MLIRValue result{fresh_name(def), res_type};
        into.ops.emplace_back(std::make_unique<TensorExtractOp>(result, arr_val, std::move(indices)));
        return result;
    }

    return std::nullopt;
}

/// Re-materializes the size-1 axes that dropped out of @p val's Mim type, so its rank matches @p want.
///
/// `«1; T»` *is* `T` in MimIR (World::seq folds arity-1 sequences away), so a rank-4 tensor whose trailing axes are
/// singletons — a 1x1 convolution weight `«cout, cin, 1, 1; T»`, say — is typed `«cout, cin; T»` and converts to a
/// rank-2 `tensor`. The tensor ops carry the true shape separately (`Sis`, `s_in`, …), and the affine access maps are
/// written against *that* rank, so an operand taken straight from the type is short by however many unit axes
/// collapsed. `tensor.expand_shape` puts them back; it is pure metadata, so the function signature — and with it the
/// ABI the caller was compiled against — stays as it was.
/// A `%tensor.map_reduce` access map may cover only a prefix of an input axis — slicing a 48-wide tensor down to
/// its first 32 columns reaches the emitter as an identity map over a 32-wide loop. `linalg.generic` requires each
/// map's image to be exactly the operand's shape, so materialize the prefix with `tensor.extract_slice`.
MLIRValue MLIREmitter::narrow_to_map(MLIRValue val,
                                     const AffineMapInfo& info,
                                     const AffineExtents& loop_extents,
                                     MLIRBlock& into) {
    if (val.empty() || !std::holds_alternative<MLIRTensorType>(val.type)) return val;

    auto shape = std::get<MLIRTensorType>(val.type).shape;
    if (shape.size() != info.pos_dims.size()) return val;

    auto want = shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        auto dim = info.pos_dims[i];
        if (!dim || *dim >= loop_extents.size()) continue;
        auto extent = loop_extents[*dim];
        if (!extent || !shape[i]) continue;
        if (*extent < *shape[i]) want[i] = *extent;
    }
    if (want == shape) return val;

    MLIRTensorType sliced_t;
    sliced_t.shape = std::move(want);
    sliced_t.elem  = std::get<MLIRTensorType>(val.type).elem;
    MLIRValue sliced{val.name + ".slice", MLIRType{std::move(sliced_t)}};
    into.ops.emplace_back(std::make_unique<TensorExtractSliceOp>(sliced, std::move(val)));
    return sliced;
}

MLIRValue MLIREmitter::restore_unit_axes(const Def* def,
                                         MLIRValue val,
                                         const std::vector<std::optional<int64_t>>& want,
                                         MLIRBlock& into) {
    // An operand that failed to emit at all is somebody else's diagnostic; do not compound it here.
    if (val.empty()) return val;

    // A tensor all of whose axes are 1 collapses to its element type; wrap it back up into the rank-0 tensor that
    // `tensor.expand_shape` can grow. Anything else non-tensor is not this bug — leave it for its own diagnostic.
    if (!std::holds_alternative<MLIRTensorType>(val.type)) {
        auto is_scalar
            = std::holds_alternative<MLIRFloatType>(val.type) || std::holds_alternative<MLIRIntType>(val.type);
        if (want.empty() || !is_scalar) return val;
        val = wrap_as_tensor(def, std::move(val), into);
    }

    auto have = std::get<MLIRTensorType>(val.type).shape;
    if (have == want) return val;

    if (have.size() > want.size()) {
        std::cerr << "mlir: operand " << val.name << " has rank " << have.size() << " but its access map expects "
                  << want.size() << "; not a collapsed-unit-axis mismatch, leaving it alone\n";
        return val;
    }

    // One group per source axis, each closed by the axis it maps to; the inserted unit axes join the group to their
    // left, except leading ones, which have no group yet and so join the first. `have` is matched greedily against
    // `want`: an axis that agrees consumes a source axis, anything else must be an inserted singleton.
    std::vector<std::vector<int64_t>> reassoc;
    size_t src = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        // A rank-0 source splits into no groups at all — `[]` — and every result axis is an inserted 1.
        if (!have.empty()) {
            if (reassoc.empty()) reassoc.emplace_back();
            reassoc.back().push_back(static_cast<int64_t>(i));
        }
        if (src < have.size() && have[src] == want[i]) {
            if (++src < have.size()) reassoc.emplace_back();
        } else if (want[i] != 1) {
            std::cerr << "mlir: cannot reconcile operand " << val.name << " with the shape its access map expects; "
                      << "axis " << i << " is neither a match nor a collapsed size-1 axis\n";
            return val;
        }
    }
    if (src != have.size()) {
        std::cerr << "mlir: cannot reconcile operand " << val.name << " with the shape its access map expects; "
                  << have.size() - src << " of its axes went unmatched\n";
        return val;
    }

    MLIRTensorType expanded;
    expanded.shape = want;
    expanded.elem  = std::get<MLIRTensorType>(val.type).elem;

    // The same Def can feed several `linalg.generic`s, each expanding it for itself, so the name needs a
    // disambiguator of its own — `fresh_name(def)` is memoized and hands out the same base every time.
    auto name = fresh_name(def) + ".expanded";
    for (int i = 1; !used_names_.insert(name).second; ++i)
        name = std::format("{}.expanded_{}", fresh_name(def), i);

    MLIRValue result{std::move(name), MLIRType{std::move(expanded)}};
    into.ops.emplace_back(std::make_unique<TensorExpandShapeOp>(result, std::move(val), std::move(reassoc)));
    return result;
}

void MLIREmitter::emit_linalg_generic(const App* app, MLIRBlock& into) {
    // Currying chain (depth 8):
    //   app0->arg = (is, post_is)                    (inputs pack)
    //   app1->arg = (maps, post_maps)                (per-input access lams)
    //   app2->arg = map_out                          (output access lam)
    //   app3->arg = (f, init, post)
    //   app4->arg = (Tis,Ris,Sis,Tps,Rps,Sps)   (implicit per-input/post element type, rank, shape)
    //   app5->arg = (So, Sr, sched)                  (output shape, full loop bounds)
    //   app6->arg = (To, Tp, Ro, Rn, TSched)
    //   app7->arg = (nis, nps)
    auto* app1 = app->callee()->as<App>();
    auto* app2 = app1->callee()->as<App>();
    auto* app3 = app2->callee()->as<App>();
    auto* app4 = app3->callee()->as<App>();
    auto* app5 = app4->callee()->as<App>();
    auto* app6 = app5->callee()->as<App>();
    auto* app7 = app6->callee()->as<App>();

    auto [inputs_pack, post_inputs_pack] = app->arg()->projs<2>();
    auto [maps_pack, post_maps_pack]     = app1->arg()->projs<2>();
    auto* map_out_def                    = app2->arg();
    auto [comb, zero, post]              = app3->arg()->projs<3>();
    auto [So, Sr, sched]                 = app5->arg()->projs<3>();
    auto [To, Tp, Ro, Rn, TSched]        = app6->arg()->projs<5>();
    auto [nis_def, nps_def]              = app7->arg()->projs<2>();
    auto [Tis, Ris, Sis, Tps, Rps, Sps]  = app4->arg()->projs<6>();

    auto nis_opt = Lit::isa(nis_def);
    auto nps_opt = Lit::isa(nps_def);
    auto Ro_opt  = Lit::isa(Ro);
    auto Rn_opt  = Lit::isa(Rn);
    assert(nis_opt && nps_opt && Ro_opt && Rn_opt);
    size_t n_inputs    = *nis_opt;
    size_t n_post      = *nps_opt;
    size_t ro          = *Ro_opt;
    size_t total_loops = *Rn_opt;
    size_t rr          = total_loops - ro;

    auto proj_input
        = [&](size_t i) -> const Def* { return n_inputs == 1 ? inputs_pack : inputs_pack->proj(n_inputs, i); };
    auto proj_map_lam = [&](size_t i) -> Lam* {
        auto* d = n_inputs == 1 ? maps_pack : maps_pack->proj(n_inputs, i);
        return d->isa_mut<Lam>();
    };
    auto* map_out = map_out_def->isa_mut<Lam>();
    assert(map_out);

    // ── Loop extents from Sr ──────────────────────────────────────────────
    // `Sr` gives the bounds of all ro + rr loops. The affine folder uses them to drop `mod`/`floordiv` terms the
    // loop domain makes redundant, which is what keeps every loop dim recoverable from the emitted maps.
    AffineExtents loop_extents;
    for (size_t i = 0; i < total_loops; ++i) {
        auto dim = total_loops == 1 ? Sr : Sr->proj(total_loops, i);
        if (auto lit = Lit::isa(dim))
            loop_extents.push_back(static_cast<int64_t>(*lit));
        else
            loop_extents.push_back(std::nullopt);
    }

    // ── Result type from So and To ────────────────────────────────────────
    std::vector<std::optional<int64_t>> res_shape;
    for (size_t i = 0; i < ro; ++i) {
        auto dim = ro == 1 ? So : So->proj(ro, i);
        if (auto lit = Lit::isa(dim))
            res_shape.push_back(static_cast<int64_t>(*lit));
        else
            res_shape.push_back(std::nullopt);
    }
    auto res_elem_type = types_.convert(To);
    MLIRTensorType res_tensor;
    res_tensor.shape = res_shape;
    res_tensor.elem  = std::make_shared<MLIRTypeNode>(res_elem_type);
    MLIRType res_type{std::move(res_tensor)};

    // ── Inputs ────────────────────────────────────────────────────────────
    // `Ris`/`Sis` state each input's rank and extents, which is what the access maps are written against. The Mim
    // type cannot be trusted for the rank: its size-1 axes have collapsed away (see restore_unit_axes).
    // The access maps are needed here, before the operands are finalized: a map may cover only a prefix of an
    // input axis, and the operand has to be narrowed to match. `lam_to_affine_map` is pure, so computing it early
    // costs nothing and the results are reused for `indexing_maps` below.
    std::vector<AffineMapInfo> in_map_info;
    for (size_t i = 0; i < n_inputs; ++i)
        in_map_info.push_back(lam_to_affine_map(proj_map_lam(i), total_loops, loop_extents));

    std::vector<MLIRValue> ins;
    for (size_t i = 0; i < n_inputs; ++i) {
        auto* in    = proj_input(i);
        auto in_val = get_or_emit(in, into);

        if (auto ris = Lit::isa(n_inputs == 1 ? Ris : Ris->proj(n_inputs, i))) {
            auto* Sis_i = n_inputs == 1 ? Sis : Sis->proj(n_inputs, i);
            std::vector<std::optional<int64_t>> in_shape;
            for (size_t j = 0; j < *ris; ++j) {
                auto dim = Lit::isa(*ris == 1 ? Sis_i : Sis_i->proj(*ris, j));
                in_shape.push_back(dim ? std::optional<int64_t>(static_cast<int64_t>(*dim)) : std::nullopt);
            }
            in_val = restore_unit_axes(in, std::move(in_val), in_shape, into);
        }

        in_val = narrow_to_map(std::move(in_val), in_map_info[i], loop_extents, into);
        ins.push_back(std::move(in_val));
    }

    // ── Output buffer ─────────────────────────────────────────────────────

    auto base       = fresh_name(app);
    auto needs_init = !zero->isa<Bot>();

    MLIRValue out_buf{base + (needs_init ? ".empty" : ".buf"), res_type};
    into.ops.emplace_back(std::make_unique<TensorEmptyOp>(out_buf));

    if (needs_init) {
        auto init_val = get_or_emit(zero, into);
        MLIRValue filled{base + ".buf", res_type};
        into.ops.emplace_back(std::make_unique<LinalgFillOp>(filled, init_val, out_buf));
        out_buf = std::move(filled);
    }
    std::vector<MLIRValue> outs{out_buf};

    // ── Affine maps from access lams ──────────────────────────────────────

    std::vector<std::string> indexing_maps;
    std::vector<size_t> bare_dims;
    auto add_map = [&](const AffineMapInfo& info) {
        indexing_maps.push_back(info.str);
        for (auto d : info.bare_dims)
            if (std::ranges::find(bare_dims, d) == bare_dims.end()) bare_dims.push_back(d);
    };

    for (size_t i = 0; i < n_inputs; ++i)
        add_map(in_map_info[i]);
    auto out_map = lam_to_affine_map(map_out, total_loops, loop_extents);
    add_map(out_map);

    // ── Shape-only operand for loop dims no map exposes ───────────────────
    // `linalg.generic` inverts the concatenated maps to recover the loop nest, so a dim occurring only inside index
    // arithmetic (a window offset like `d2 * 2 + d4`) leaves the op unverifiable. MLIR's own `linalg.pooling_*` ops
    // solve this by taking the kernel as a shape-only `ins` operand; %tensor.pool has no such operand (the window is
    // a literal), so synthesize one. Its element is never read — it exists purely to name the dims.
    std::vector<size_t> missing;
    for (size_t i = 0; i < total_loops; ++i)
        if (std::ranges::find(bare_dims, i) == bare_dims.end()) missing.push_back(i);

    std::optional<MLIRValue> shape_arg;
    if (!missing.empty()) {
        MLIRTensorType shape_tensor;
        shape_tensor.elem = std::make_shared<MLIRTypeNode>(res_elem_type);
        std::string dims;
        for (size_t i = 0; i < missing.size(); ++i) {
            shape_tensor.shape.push_back(loop_extents[missing[i]]);
            dims += (i ? ", " : "") + std::format("d{}", missing[i]);
        }

        MLIRValue shape_buf{fresh_name(app) + ".shape", MLIRType{std::move(shape_tensor)}};
        into.ops.emplace_back(std::make_unique<TensorEmptyOp>(shape_buf));
        ins.push_back(shape_buf);

        // Slot the map in ahead of the output map: `indexing_maps` must follow ins-then-outs order.
        std::string dim_str;
        for (size_t i = 0; i < total_loops; ++i)
            dim_str += (i ? ", " : "") + std::format("d{}", i);
        indexing_maps.back() = std::format("affine_map<({}) -> ({})>", dim_str, dims);
        indexing_maps.push_back(out_map.str);

        shape_arg = MLIRValue{fresh_name("%shape_"), res_elem_type};
    }

    // ── Iterator types ────────────────────────────────────────────────────
    std::vector<std::string> iterator_types;
    for (size_t i = 0; i < ro; ++i)
        iterator_types.push_back("parallel");
    for (size_t i = 0; i < rr; ++i)
        iterator_types.push_back("reduction");

    // ── Body ──────────────────────────────────────────────────────────────
    auto* body_lam = comb->isa_mut<Lam>();
    auto seed      = seed_linalg_args(body_lam, res_elem_type);

    std::vector<MLIRValue> body_args = seed.ins;
    // Between the real inputs and the accumulator, matching the ins-then-outs block-arg order.
    if (shape_arg) body_args.push_back(*shape_arg);
    body_args.push_back(seed.acc);

    bind_linalg_args(body_lam, seed);

    auto* op = new LinalgGenericOp(ins, outs, indexing_maps, iterator_types, body_args);
    emit_linalg_body_scoped(body_lam, op->body().entry());

    values_[app] = op->result();
    into.ops.emplace_back(op);

    if (n_post == 0) return;

    // ── Epilogue ──────────────────────────────────────────────────────────
    // `post` runs once per output cell, after the reduction has been folded away, so it cannot sit in the reduction
    // body — it becomes a second, all-parallel `linalg.generic` over the `Ro` output coordinates. `post_maps` are
    // already stated over exactly those coordinates.
    auto* post_lam = post->isa_mut<Lam>();
    assert(post_lam && "epilogue must be a lam");

    auto post_elem_type = types_.convert(Tp);
    MLIRTensorType post_tensor;
    post_tensor.shape = res_shape;
    post_tensor.elem  = std::make_shared<MLIRTypeNode>(post_elem_type);
    MLIRType post_type{std::move(post_tensor)};

    std::string out_dims;
    for (size_t i = 0; i < ro; ++i)
        out_dims += (i ? ", " : "") + std::format("d{}", i);
    auto identity_map = std::format("affine_map<({}) -> ({})>", out_dims, out_dims);

    // The identity map on the folded result exposes every output dim as a bare `d<i>`, so the op stays invertible
    // whatever arithmetic the epilogue maps use — no shape-only operand is ever needed here.

    std::vector<MLIRValue> post_ins{op->result()};
    std::vector<std::string> post_indexing_maps{identity_map};
    for (size_t j = 0; j < n_post; ++j) {
        auto* in  = n_post == 1 ? post_inputs_pack : post_inputs_pack->proj(n_post, j);
        auto* map = n_post == 1 ? post_maps_pack : post_maps_pack->proj(n_post, j);
        auto* Rp  = n_post == 1 ? Rps : Rps->proj(n_post, j);
        auto* Sp  = n_post == 1 ? Sps : Sps->proj(n_post, j);

        auto in_val = get_or_emit(in, into);

        // Size-1 axes collapse out of a tensor's Mim type, so an epilogue input declared `«1, 32; T»` can arrive as
        // a rank-1 `«32; T»`; those positions have to drop out of its access map. But a value the emitter itself
        // materialized keeps them — a previous epilogue's result is built at the full `res_shape` — so which axes
        // are really gone follows from the operand's own rank, not from the declared shape alone.
        auto rp = Lit::isa(Rp);
        assert(rp && "epilogue input rank must be literal");
        std::vector<std::optional<int64_t>> declared;
        for (size_t i = 0; i < *rp; ++i) {
            auto lit = Lit::isa(*rp == 1 ? Sp : Sp->proj(*rp, i));
            declared.push_back(lit ? std::optional<int64_t>(static_cast<int64_t>(*lit)) : std::nullopt);
        }

        auto* in_tensor = std::get_if<MLIRTensorType>(&in_val.type);
        auto have       = in_tensor ? in_tensor->shape : std::vector<std::optional<int64_t>>{};

        std::vector<size_t> omit;
        size_t src = 0;
        for (size_t i = 0; i < declared.size(); ++i) {
            if (src < have.size() && have[src] == declared[i]) {
                ++src;
            } else if (declared[i] == 1) {
                omit.push_back(i);
            } else {
                std::cerr << "mlir: epilogue input " << in_val.name << " axis " << i
                          << " is neither present in the operand nor a collapsed size-1 axis\n";
                break;
            }
        }
        if (src != have.size())
            std::cerr << "mlir: epilogue input " << in_val.name << " has rank " << have.size()
                      << " that does not align with its declared rank " << *rp << "\n";

        post_ins.push_back(in_val);
        post_indexing_maps.push_back(lam_to_affine_map(map->isa_mut<Lam>(), ro, res_shape, omit).str);
    }
    post_indexing_maps.push_back(identity_map);

    MLIRValue post_buf{fresh_name(app) + ".post", post_type};
    into.ops.emplace_back(std::make_unique<TensorEmptyOp>(post_buf));

    auto post_seed = seed_linalg_args(post_lam, res_elem_type);

    // The folded accumulator is an `ins` operand here, not the carried `outs` one, so it leads the block args; the
    // `outs` arg is only written.
    std::vector<MLIRValue> post_body_args{post_seed.acc};
    for (const auto& a : post_seed.ins)
        post_body_args.push_back(a);
    post_body_args.push_back(MLIRValue{fresh_name("%post_"), post_elem_type});

    bind_linalg_args(post_lam, post_seed);

    auto* post_op = new LinalgGenericOp(post_ins, {post_buf}, post_indexing_maps,
                                        std::vector<std::string>(ro, "parallel"), post_body_args);
    emit_linalg_body_scoped(post_lam, post_op->body().entry());

    values_[app] = post_op->result();
    into.ops.emplace_back(post_op);
}

MLIREmitter::LinalgBodySeed MLIREmitter::seed_linalg_args(Lam* body_lam, const MLIRType& acc_type) {
    LinalgBodySeed seed;
    auto body_var_type = body_lam->var()->type();

    std::vector<size_t> arg_path;
    const Def* arg_type = body_var_type;
    if (auto sigma = body_var_type->isa<Sigma>()) {
        for (size_t i = 0; i < sigma->num_ops(); ++i) {
            if (!sigma->op(i)->isa<Pi>() && !Axm::isa<plug::mem::M>(sigma->op(i))) {
                arg_path.push_back(i);
                arg_type = sigma->op(i);
                break;
            }
        }
    }

    auto put = [&](std::vector<size_t> p, MLIRValue v) { seed.paths[std::move(p)] = std::move(v); };

    auto plan_ins = [&](const Def* ins_t, std::vector<size_t> ins_path) {
        if (auto s = ins_t->isa<Sigma>()) {
            for (size_t i = 0; i < s->num_ops(); ++i) {
                auto p = ins_path;
                p.push_back(i);
                MLIRValue v{fresh_name("%in_"), types_.convert(s->op(i))};
                seed.ins.push_back(v);
                put(std::move(p), v);
            }
        } else if (auto a = ins_t->isa<Arr>()) {
            if (auto n = Lit::isa(a->arity())) {
                for (size_t i = 0; i < *n; ++i) {
                    auto p = ins_path;
                    p.push_back(i);
                    MLIRValue v{fresh_name("%in_"), types_.convert(a->body())};
                    seed.ins.push_back(v);
                    put(std::move(p), v);
                }
            }
        } else if (!ins_t->isa<Pi>()) {
            MLIRValue v{fresh_name("%in_"), types_.convert(ins_t)};
            seed.ins.push_back(v);
            put(std::move(ins_path), v);
        }
    };

    if (auto s = arg_type->isa<Sigma>()) {
        // [acc, ins]
        auto acc_p = arg_path;
        acc_p.push_back(0);
        auto ins_p = arg_path;
        ins_p.push_back(1);
        plan_ins(s->op(1), std::move(ins_p));
        seed.acc = MLIRValue{fresh_name("%acc_"), acc_type};
        put(std::move(acc_p), seed.acc);
    } else if (auto a = arg_type->isa<Arr>()) {
        // «2; T» from a [T, «1; T»] singleton collapse: [acc, single_input]
        if (auto n = Lit::isa(a->arity()); n && *n == 2) {
            auto acc_p = arg_path;
            acc_p.push_back(0);
            auto ins_p = arg_path;
            ins_p.push_back(1);
            MLIRValue v{fresh_name("%in_"), types_.convert(a->body())};
            seed.ins.push_back(v);
            put(std::move(ins_p), v);
            seed.acc = MLIRValue{fresh_name("%acc_"), acc_type};
            put(std::move(acc_p), seed.acc);
        } else {
            seed.acc = MLIRValue{fresh_name("%acc_"), acc_type};
            put(arg_path, seed.acc);
        }
    } else {
        seed.acc = MLIRValue{fresh_name("%acc_"), acc_type};
        put(arg_path, seed.acc);
    }

    return seed;
}

void MLIREmitter::bind_linalg_args(Lam* body_lam, const LinalgBodySeed& seed) {
    auto type_arity = [](const Def* t) -> size_t {
        if (auto s = t->isa<Sigma>()) return s->num_ops();
        if (auto a = t->isa<Arr>())
            if (auto n = Lit::isa(a->arity())) return *n;
        return 0;
    };
    auto nav_path = [&](const Def* root, const std::vector<size_t>& path) -> const Def* {
        const Def* cur = root;
        for (size_t i : path) {
            size_t arity = type_arity(cur->type());
            if (arity == 0) return nullptr;
            cur = cur->proj(arity, i);
        }
        return cur;
    };

    for (auto& [path, val] : seed.paths)
        if (auto d = nav_path(body_lam->var(), path)) values_[d] = val;
}

void MLIREmitter::emit_linalg_body_scoped(Lam* body_lam, MLIRBlock& body_bb) {
    DefSet pre_body_keys;
    for (auto& [d, _] : values_)
        pre_body_keys.insert(d);

    emit_linalg_body(body_lam, body_lam->ret_var(), body_bb);

    std::vector<const Def*> body_added;
    for (auto& [d, _] : values_)
        if (!pre_body_keys.contains(d)) body_added.push_back(d);
    for (auto* d : body_added) {
        values_.erase(d);
        // Drop the memoized name too: the Def has to be re-emitted in the next body that needs it, and that
        // second definition needs a name of its own. `used_names_` keeps the old one reserved.
        names_.erase(d);
    }
}

void MLIREmitter::emit_linalg_body(Lam* body_lam, const Def* ret_var, MLIRBlock& body_bb) {
    assert(body_lam->is_set());
    auto* app = body_lam->body()->isa<App>();
    assert(app);
    auto* callee = app->callee();
    auto* arg    = app->arg();

    if (is_return_callee(callee, ret_var)) {
        std::vector<MLIRValue> yield_vals;
        if (!Axm::isa<plug::mem::M>(arg->type())) {
            auto v = get_or_emit(arg, body_bb);
            if (!v.empty()) yield_vals.push_back(v);
        }
        body_bb.ops.emplace_back(std::make_unique<LinalgYieldOp>(std::move(yield_vals)));
        return;
    }

    // Local continuation call
    if (auto* local_lam = callee->isa_mut<Lam>()) {
        if (local_lam->is_set() && !local_lam->is_external()) {
            // Seed the local lam's parameter(s)
            auto dom = local_lam->type()->dom();
            if (auto sigma = dom->isa<Sigma>()) {
                for (size_t i = 0; i < sigma->num_ops(); ++i) {
                    if (Axm::isa<plug::mem::M>(sigma->op(i))) continue;
                    auto var_op = local_lam->var()->proj(sigma->num_ops(), i);
                    auto v      = get_or_emit(arg->proj(sigma->num_ops(), i), body_bb);
                    if (!v.empty()) values_[var_op] = v;
                }
            } else if (!dom->isa<Pi>() && !Axm::isa<plug::mem::M>(dom)) {
                // single scalar argument
                auto v = get_or_emit(arg, body_bb);
                if (!v.empty()) values_[local_lam->var()] = v;
            }
            // Recurse into the local lam's body in the same block
            emit_linalg_body(local_lam, local_lam->ret_var() ? local_lam->ret_var() : ret_var, body_bb);
            return;
        }
    }

    auto [axm, curry, trip] = Axm::get(app);
    std::cerr << "unhandled callee in emit_linalg_body: " << callee->node_name() << " sym='" << callee->sym().str()
              << "' axm='" << (axm ? axm->sym().str() : "<none>") << "' type=" << callee->type() << "\n";
    assert(false && "unhandled callee in emit_linalg_body");
}

} // namespace mim::mlir_be
