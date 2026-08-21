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
        // %tensor.get @(T, r, s) (index, arr)
        auto [idx_def, arr_def] = app->arg()->projs<2>();

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
void MLIREmitter::emit_linalg_generic(const App* app, MLIRBlock& into) {
    // %tensor.map_reduce_post currying chain (depth 8):
    //   app0->arg = (is, post_is)     (inputs pack, epilogue inputs pack)
    //   app1->arg = (maps, post_maps) (per-input access lams, per-epilogue-input access lams)
    //   app2->arg = map_out           (output access lam)
    //   app3->arg = (f, init, post)
    //   app4->arg = (Tis, Ris, Sis, Tps, Rps, Sps) [skip, implicit type args]
    //   app5->arg = (So, Sr, sched)   (output shape, full loop bounds; sched drives the native lowering only)
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
    (void)sched, (void)TSched; // the schedule chooser only steers the native (ll) lowering

    auto nis_opt = Lit::isa(nis_def);
    auto nps_opt = Lit::isa(nps_def);
    auto Ro_opt  = Lit::isa(Ro);
    auto Rn_opt  = Lit::isa(Rn);
    assert(nis_opt && nps_opt && Ro_opt && Rn_opt);
    size_t n_inputs    = *nis_opt;
    size_t n_post      = *nps_opt;
    size_t ro          = *Ro_opt;
    size_t total_loops = *Rn_opt;

    auto proj_input
        = [&](size_t i) -> const Def* { return n_inputs == 1 ? inputs_pack : inputs_pack->proj(n_inputs, i); };
    auto proj_map_lam = [&](size_t i) -> Lam* {
        auto* d = n_inputs == 1 ? maps_pack : maps_pack->proj(n_inputs, i);
        return d->isa_mut<Lam>();
    };
    auto* map_out = map_out_def->isa_mut<Lam>();
    assert(map_out);

    // ── Collapsed input axes ──────────────────────────────────────────────
    // A literal size-1 axis collapses out of a tensor's (nested array) type («1, n; T» IS «n; T»), so
    // the converted MLIR operand is missing it — but a fused (read-through) access map still yields a
    // coordinate for it. Mask those result positions out so map rank and operand rank agree.
    auto [Tis, Ris, Sis, Tps, Rps, Sps] = app4->arg()->projs<6>();
    (void)Tis, (void)Ris, (void)Tps, (void)Rps;
    auto collapsed_axes = [](const Def* shape) -> std::optional<std::vector<bool>> {
        auto rank = Lit::isa(shape->arity());
        if (!rank) return std::nullopt;
        std::vector<bool> keep(*rank, true);
        bool any = false;
        for (size_t d = 0; d < *rank; ++d) {
            auto extent = Lit::isa(*rank == 1 ? shape : shape->proj(*rank, d));
            if (extent && *extent == 1) keep[d] = false, any = true;
        }
        if (!any) return std::nullopt;
        return keep;
    };

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

    // ── Drop unit-extent loop dims ────────────────────────────────────────
    // The neutral tile of the schedule selectors is a strip-mine by 1, which leaves size-1 axes in
    // the domain. The native lowering folds them away, and so does this backend: such a loop runs
    // exactly once, so every use of its dim reads as 0 and the dim itself can go.
    std::vector<std::optional<size_t>> dim_map(total_loops);
    size_t n_kept = 0, ro_kept = 0;
    for (size_t i = 0; i < total_loops; ++i) {
        bool drop = loop_extents[i] && *loop_extents[i] == 1;
        if (!drop) {
            dim_map[i] = n_kept++;
            if (i < ro) ++ro_kept;
        }
    }
    auto* dim_map_p = n_kept == total_loops ? nullptr : &dim_map;

    AffineExtents kept_extents;
    for (size_t i = 0; i < total_loops; ++i)
        if (dim_map[i]) kept_extents.push_back(loop_extents[i]);
    size_t n_loops = n_kept;
    size_t rr_kept = n_loops - ro_kept;

    // ── Result type from So and To ────────────────────────────────────────
    // Literal size-1 axes of `So` collapse out of the result's (nested array) type, so they must
    // collapse out of the emitted tensor type and the write map as well.
    auto out_keep = collapsed_axes(So);
    AffineExtents so_extents; // per original output axis, for rendering the post access maps
    std::vector<std::optional<int64_t>> res_shape;
    for (size_t i = 0; i < ro; ++i) {
        auto dim = ro == 1 ? So : So->proj(ro, i);
        auto lit = Lit::isa(dim);
        auto ext = lit ? std::optional<int64_t>(static_cast<int64_t>(*lit)) : std::nullopt;
        so_extents.push_back(ext);
        if (!out_keep || (*out_keep)[i]) res_shape.push_back(ext);
    }
    auto res_elem_type = types_.convert(To);
    MLIRTensorType res_tensor;
    res_tensor.shape = res_shape;
    res_tensor.elem  = std::make_shared<MLIRTypeNode>(res_elem_type);
    MLIRType res_type{std::move(res_tensor)};

    // ── Inputs ────────────────────────────────────────────────────────────
    std::vector<MLIRValue> ins;
    for (size_t i = 0; i < n_inputs; ++i)
        ins.push_back(get_or_emit(proj_input(i), into));

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

    for (size_t i = 0; i < n_inputs; ++i) {
        auto* S_i = n_inputs == 1 ? Sis : Sis->proj(n_inputs, i);
        auto keep = collapsed_axes(S_i);
        add_map(lam_to_affine_map(proj_map_lam(i), total_loops, loop_extents, dim_map_p, keep ? &*keep : nullptr));
    }
    auto out_map = lam_to_affine_map(map_out, total_loops, loop_extents, dim_map_p, out_keep ? &*out_keep : nullptr);
    add_map(out_map);

    // ── Shape-only operand for loop dims no map exposes ───────────────────
    // `linalg.generic` inverts the concatenated maps to recover the loop nest, so a dim occurring only inside index
    // arithmetic (a window offset like `d2 * 2 + d4`) leaves the op unverifiable. MLIR's own `linalg.pooling_*` ops
    // solve this by taking the kernel as a shape-only `ins` operand; %tensor.pool has no such operand (the window is
    // a literal), so synthesize one. Its element is never read — it exists purely to name the dims.
    std::vector<size_t> missing;
    for (size_t i = 0; i < n_loops; ++i)
        if (std::ranges::find(bare_dims, i) == bare_dims.end()) missing.push_back(i);

    std::optional<MLIRValue> shape_arg;
    if (!missing.empty()) {
        MLIRTensorType shape_tensor;
        shape_tensor.elem = std::make_shared<MLIRTypeNode>(res_elem_type);
        std::string dims;
        for (size_t i = 0; i < missing.size(); ++i) {
            shape_tensor.shape.push_back(kept_extents[missing[i]]);
            dims += (i ? ", " : "") + std::format("d{}", missing[i]);
        }

        MLIRValue shape_buf{fresh_name(app) + ".shape", MLIRType{std::move(shape_tensor)}};
        into.ops.emplace_back(std::make_unique<TensorEmptyOp>(shape_buf));
        ins.push_back(shape_buf);

        // Slot the map in ahead of the output map: `indexing_maps` must follow ins-then-outs order.
        std::string dim_str;
        for (size_t i = 0; i < n_loops; ++i)
            dim_str += (i ? ", " : "") + std::format("d{}", i);
        indexing_maps.back() = std::format("affine_map<({}) -> ({})>", dim_str, dims);
        indexing_maps.push_back(out_map.str);

        shape_arg = MLIRValue{fresh_name("%shape_"), res_elem_type};
    }

    // ── Iterator types ────────────────────────────────────────────────────
    std::vector<std::string> iterator_types;
    for (size_t i = 0; i < ro_kept; ++i)
        iterator_types.push_back("parallel");
    for (size_t i = 0; i < rr_kept; ++i)
        iterator_types.push_back("reduction");

    // ── Body seeding (path-based strategy) ────────────────────────────────
    auto* body_lam = comb->isa_mut<Lam>();
    auto plan      = plan_mr_body(body_lam);

    MLIRValue acc_val{fresh_name("%acc_"), res_elem_type};
    plan.vals[plan.first_path] = acc_val;

    std::vector<MLIRValue> body_args = plan.ins_args;
    // Between the real inputs and the accumulator, matching the ins-then-outs block-arg order.
    if (shape_arg) body_args.push_back(*shape_arg);
    body_args.push_back(acc_val);

    auto* op = new LinalgGenericOp(ins, outs, indexing_maps, iterator_types, body_args);
    emit_mr_body(body_lam, plan.vals, op->body().entry());
    into.ops.emplace_back(op);
    MLIRValue result = op->result();

    // ── Post epilogue ─────────────────────────────────────────────────────
    // `%tensor.map_reduce` delegates to map_reduce_post with the neutral epilogue (a specialization
    // of `%tensor.id` returning the folded accumulator unchanged) — recognize it structurally so no
    // second generic is emitted for it.
    auto* post_lam   = post->isa_mut<Lam>();
    auto is_identity = [&]() -> bool {
        if (n_post != 0) return false;
        if (!post_lam || !post_lam->is_set()) return false;
        auto* papp = post_lam->body()->isa<App>();
        if (!papp || !is_return_callee(papp->callee(), post_lam->ret_var())) return false;
        auto* x = post_lam->var()->proj(post_lam->num_vars(), 0);
        // `x` is either the accumulator itself (flat dom) or the ([To, «0; …»], ret) pair whose
        // first half is the accumulator.
        return papp->arg() == x || (x->type()->isa<Sigma>() && papp->arg() == x->proj(2, 0));
    };

    if (!is_identity()) {
        assert(post_lam && post_lam->is_set() && "stuck post epilogue not supported");

        // `post` runs once per output cell, after the reduction is folded away: a second,
        // parallel-only generic over the output domain. It reads the folded tensor at identity
        // coordinates and each epilogue input through its post_map (output-cell → read coords).
        // Its domain is the kept output axes; a collapsed unit axis reads as 0 in the post maps.
        std::vector<std::optional<size_t>> post_dim_map(ro);
        size_t ro_out = 0;
        for (size_t i = 0; i < ro; ++i)
            if (!out_keep || (*out_keep)[i]) post_dim_map[i] = ro_out++;
        auto* post_dim_map_p = out_keep ? &post_dim_map : nullptr;

        std::string id_dims;
        for (size_t i = 0; i < ro_out; ++i)
            id_dims += (i ? ", " : "") + std::format("d{}", i);
        auto id_map = std::format("affine_map<({}) -> ({})>", id_dims, id_dims);

        std::vector<std::string> post_idx_maps{id_map};
        for (size_t j = 0; j < n_post; ++j) {
            auto* d  = n_post == 1 ? post_maps_pack : post_maps_pack->proj(n_post, j);
            auto* pm = d->isa_mut<Lam>();
            assert(pm && "post access map must be a lam");
            auto* S_j = n_post == 1 ? Sps : Sps->proj(n_post, j);
            auto keep = collapsed_axes(S_j);
            post_idx_maps.push_back(lam_to_affine_map(pm, ro, so_extents, post_dim_map_p, keep ? &*keep : nullptr).str);
        }
        post_idx_maps.push_back(id_map);

        std::vector<MLIRValue> post_ins{result};
        for (size_t j = 0; j < n_post; ++j) {
            auto* d = n_post == 1 ? post_inputs_pack : post_inputs_pack->proj(n_post, j);
            auto v  = get_or_emit(d, into);
            if (!std::holds_alternative<MLIRTensorType>(v.type)) v = wrap_as_tensor(d, v, into);
            post_ins.push_back(std::move(v));
        }

        auto post_elem_type = types_.convert(Tp);
        MLIRTensorType post_tensor;
        post_tensor.shape = res_shape;
        post_tensor.elem  = std::make_shared<MLIRTypeNode>(post_elem_type);
        MLIRType post_type{std::move(post_tensor)};

        MLIRValue post_buf{base + ".post.buf", post_type};
        into.ops.emplace_back(std::make_unique<TensorEmptyOp>(post_buf));

        std::vector<std::string> post_iters(ro_out, "parallel");

        auto post_plan = plan_mr_body(post_lam);
        MLIRValue folded_arg{fresh_name("%acc_"), res_elem_type};
        post_plan.vals[post_plan.first_path] = folded_arg;

        // Block args: the folded cell first, then the epilogue inputs, then the (unread) out arg.
        std::vector<MLIRValue> post_args{folded_arg};
        post_args.insert(post_args.end(), post_plan.ins_args.begin(), post_plan.ins_args.end());
        post_args.emplace_back(fresh_name("%out_"), post_elem_type);

        auto* post_op = new LinalgGenericOp(post_ins, {post_buf}, post_idx_maps, post_iters, post_args);
        emit_mr_body(post_lam, post_plan.vals, post_op->body().entry());
        into.ops.emplace_back(post_op);
        result = post_op->result();
    }

    values_[app] = result;
}

MLIREmitter::MRBodyPlan MLIREmitter::plan_mr_body(Lam* body_lam) {
    assert(body_lam && body_lam->is_set());
    MRBodyPlan plan;

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

    auto put = [&](std::vector<size_t> p, MLIRValue v) { plan.vals[std::move(p)] = std::move(v); };

    auto plan_ins = [&](const Def* ins_t, std::vector<size_t> ins_path) {
        if (auto s = ins_t->isa<Sigma>()) {
            for (size_t i = 0; i < s->num_ops(); ++i) {
                auto p = ins_path;
                p.push_back(i);
                MLIRValue v{fresh_name("%in_"), types_.convert(s->op(i))};
                plan.ins_args.push_back(v);
                put(std::move(p), v);
            }
        } else if (auto a = ins_t->isa<Arr>()) {
            if (auto n = Lit::isa(a->arity())) {
                for (size_t i = 0; i < *n; ++i) {
                    auto p = ins_path;
                    p.push_back(i);
                    MLIRValue v{fresh_name("%in_"), types_.convert(a->body())};
                    plan.ins_args.push_back(v);
                    put(std::move(p), v);
                }
            }
        } else if (!ins_t->isa<Pi>()) {
            MLIRValue v{fresh_name("%in_"), types_.convert(ins_t)};
            plan.ins_args.push_back(v);
            put(std::move(ins_path), v);
        }
    };

    if (auto s = arg_type->isa<Sigma>()) {
        // [first, ins]
        auto first_p = arg_path;
        first_p.push_back(0);
        auto ins_p = arg_path;
        ins_p.push_back(1);
        plan_ins(s->op(1), std::move(ins_p));
        plan.first_path = std::move(first_p);
    } else if (auto a = arg_type->isa<Arr>(); a && Lit::isa(a->arity()) && *Lit::isa(a->arity()) == 2) {
        // «2; T» from a [T, «1; T»] singleton collapse: [first, single_input]
        auto first_p = arg_path;
        first_p.push_back(0);
        auto ins_p = arg_path;
        ins_p.push_back(1);
        MLIRValue v{fresh_name("%in_"), types_.convert(a->body())};
        plan.ins_args.push_back(v);
        put(std::move(ins_p), v);
        plan.first_path = std::move(first_p);
    } else {
        plan.first_path = arg_path;
    }

    return plan;
}

void MLIREmitter::emit_mr_body(Lam* body_lam,
                               const std::map<std::vector<size_t>, MLIRValue>& path_vals,
                               MLIRBlock& body_bb) {
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

    for (auto& [path, val] : path_vals)
        if (auto d = nav_path(body_lam->var(), path)) values_[d] = val;
    DefSet pre_body_keys;
    for (auto& [d, _] : values_)
        pre_body_keys.insert(d);

    emit_linalg_body(body_lam, body_bb);

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

void MLIREmitter::emit_linalg_body(Lam* body_lam, MLIRBlock& body_bb) {
    assert(body_lam->is_set());
    auto* app = body_lam->body()->isa<App>();
    assert(app);
    auto* callee = app->callee();
    auto* arg    = app->arg();

    if (is_return_callee(callee, body_lam->ret_var())) {
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
            emit_linalg_body(local_lam, body_bb);
            return;
        }
    }

    std::cerr << "unhandled callee in emit_linalg_body: " << callee->sym().str() << "\n";
    assert(false && "unhandled callee in emit_linalg_body");
}

} // namespace mim::mlir_be
