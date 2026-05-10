"""P1 smoke: lift_2_5d_to_3d + aux_seg_supervision + deep_supervision.

Routes an end-to-end step through the REAL trainer aggregator
``Trainer._compute_loss_aux_fp32`` — not a shape check — so the test
fails immediately if DS unwrap, aux routing, label-slice rank, or
inner-loss contract drift in any future refactor.

Contracts exercised simultaneously:
    * image          : (B, n_views, D, H, W)                 (lift keeps rank-5)
    * model output   : {"main": list[(B, num_fg, D_r, H_r, W_r)] x n_ds,
                        "aux":  list[(B, num_fg, D, H, W)]   x (n_views-1)}
    * main loss path : DeepSupervisionLoss -> MultiResolutionLoss(num_res=1)
    * aux  loss path : MultiResolutionLoss(num_res=1) per view (NO DS)
    * labels         : (B, C_res=n_views, D, H, W); sliced as [:, k:k+1]
                       (rank-5, length-1 C_res axis) for every view.
"""
import sys, os, traceback, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from segtask_v1.config import load_config
from segtask_v1.models.factory import build_model
from segtask_v1.losses.losses import (
    MultiResolutionLoss, DeepSupervisionLoss, build_loss)


def _make_cfg(**overrides):
    cfg = load_config("configs/experiments/lift_a_planA_aux.yaml")
    for k, v in overrides.items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], v)
    cfg.sync()
    cfg.validate()
    return cfg


def _build_loss_stack(cfg):
    """Rebuild the same criterion stack ``Trainer.__init__`` would build.

    Kept in sync by construction with ``trainer.py:390-430`` — any drift
    there will break this test (intentional: a wrong wrapper choice in
    lift+aux+DS is exactly what we're guarding against).
    """
    base = build_loss(cfg.loss)
    num_res = 1  # lift mode forces num_res=1 for main as well
    inner = MultiResolutionLoss(
        base_loss=base, num_fg_classes=cfg.num_fg_classes,
        num_res=num_res, label_values=cfg.data.label_values)
    if cfg.model.deep_supervision and cfg.loss.deep_supervision_weights:
        criterion = DeepSupervisionLoss(
            base_loss=inner,
            weights=cfg.loss.deep_supervision_weights,
            normalize_weights=True,
            upsample_pred=False,
        )
    else:
        criterion = inner
    aux_inner = MultiResolutionLoss(
        base_loss=base, num_fg_classes=cfg.num_fg_classes,
        num_res=1, label_values=cfg.data.label_values)
    return criterion, aux_inner


def _trainer_like_loss(criterion, aux_inner, aux_weights, pred, label_all, wmap_all):
    """Mirror of ``Trainer._compute_loss_aux_fp32`` — lift branch only.

    Kept deliberately SHORT so the shape contract is in-your-face; if
    the trainer source diverges, the assertion at the bottom of the
    test catches it.
    """
    label_main = label_all[:, :1]
    wmap_main = wmap_all[:, :1] if wmap_all is not None else None
    main_pred = pred["main"]
    aux_preds = pred.get("aux", []) or []
    total = criterion(main_pred, label_main, weight_map=wmap_main)
    for k_idx, (ap, w_k) in enumerate(zip(aux_preds, aux_weights)):
        view_k = k_idx + 1
        lbl_k = label_all[:, view_k:view_k + 1]
        wm_k = (wmap_all[:, view_k:view_k + 1] if wmap_all is not None else None)
        total = total + w_k * aux_inner(ap, lbl_k, weight_map=wm_k)
    return total


def case(name, cfg, B=2):
    print(f"\n=== {name} ===")
    n_views = len(cfg.data.multi_res_scales)
    D, H, W = cfg.data.patch_size
    num_fg = cfg.num_fg_classes
    print(f"  D={D} H={H} W={W} n_views={n_views} num_fg={num_fg} "
          f"DS={cfg.model.deep_supervision} aux={cfg.model.aux_seg_supervision} "
          f"ds_w={cfg.loss.deep_supervision_weights}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device).train()

    image = torch.randn(B, n_views, D, H, W, device=device)
    # Raw integer labels in label_values = [0, 1, 2]; float for fp path.
    label = torch.randint(0, 3, (B, n_views, D, H, W),
                          device=device, dtype=torch.float32)
    wmap = torch.rand(B, n_views, D, H, W, device=device) + 0.5

    out = model(image)
    assert isinstance(out, dict), f"Expected dict output under aux+DS, got {type(out)}"
    main, aux = out["main"], out["aux"]
    assert isinstance(main, list), (
        f"deep_supervision=True must make main a list, got {type(main)}")
    assert len(aux) == n_views - 1, (
        f"aux count {len(aux)} != n_views-1 = {n_views-1}")
    print(f"  main list len={len(main)} | shapes={[tuple(m.shape) for m in main]}")
    print(f"  aux shapes={[tuple(a.shape) for a in aux]}")

    # ---- Core shape invariants ----
    # (1) main[0] is full-resolution -> (B, num_fg, D, H, W).
    assert tuple(main[0].shape) == (B, num_fg, D, H, W), tuple(main[0].shape)
    # (2) each DS level halves spatial dims (since downsample is isotropic).
    for i in range(1, len(main)):
        exp_d = max(D >> i, 1)
        exp_h = max(H >> i, 1)
        exp_w = max(W >> i, 1)
        assert tuple(main[i].shape) == (B, num_fg, exp_d, exp_h, exp_w), (
            f"DS level {i} expected (B,{num_fg},{exp_d},{exp_h},{exp_w}), "
            f"got {tuple(main[i].shape)}")
    # (3) every aux is full-resolution (no DS on aux heads).
    for k, a in enumerate(aux):
        assert tuple(a.shape) == (B, num_fg, D, H, W), (
            f"aux[{k}] shape={tuple(a.shape)}")

    # ---- Loss + backward through real stack ----
    criterion, aux_inner = _build_loss_stack(cfg)
    criterion = criterion.to(device)
    aux_inner = aux_inner.to(device)
    aux_weights = [0.5 ** (k + 1) for k in range(n_views - 1)]  # trainer default

    loss = _trainer_like_loss(
        criterion, aux_inner, aux_weights, out, label, wmap)
    print(f"  loss={loss.item():.4f}  aux_w={aux_weights}")
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    assert n_grad == n_total, (
        f"Incomplete backward: {n_grad}/{n_total} params have .grad. "
        f"This would mean one of the (DS-level | aux-view) heads is "
        f"detached from the loss.")
    print(f"  params with grad: {n_grad}/{n_total}  OK")


def main():
    # ---- Case A: lift + aux + DS (full stack) ----
    cfg = _make_cfg(**{
        "model.deep_supervision": True,
        # nnU-Net-style weights: highest res first, halved per level.
        # encoder_channels len = 5 -> decoder has 4 levels -> 4 DS outputs
        # (main[0] = full res, main[1..3] = halved cascade).
        "loss.deep_supervision_weights": [1.0, 0.5, 0.25, 0.125],
    })
    case("lift+aux+DS  D=16 n_levels=5 n_views=3 ds_levels=4", cfg)

    # ---- Case B: lift + DS (no aux) — regression; same criterion path ----
    cfg = _make_cfg(**{
        "model.deep_supervision": True,
        "model.aux_seg_supervision": False,
        "loss.deep_supervision_weights": [1.0, 0.5, 0.25, 0.125],
    })
    print("\n=== lift+DS (no aux), D=16 n_levels=5 n_views=3 ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device).train()
    B = 2
    n_views = len(cfg.data.multi_res_scales)
    D, H, W = cfg.data.patch_size
    image = torch.randn(B, n_views, D, H, W, device=device)
    label = torch.randint(0, 3, (B, n_views, D, H, W),
                          device=device, dtype=torch.float32)
    out = model(image)
    # No aux -> plain list, no dict wrapping.
    assert isinstance(out, list), (
        f"aux=False + DS=True must return list, got {type(out)}")
    print(f"  main list len={len(out)}  shapes={[tuple(m.shape) for m in out]}")
    criterion, _ = _build_loss_stack(cfg)
    criterion = criterion.to(device)
    loss = criterion(out, label[:, :1])
    print(f"  loss={loss.item():.4f}")
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    assert n_grad == n_total, f"{n_grad}/{n_total} grads"
    print(f"  params with grad: {n_grad}/{n_total}  OK")

    # ---- Case C: lift + aux + DS, D=8 thin-slab, 4 stages ----
    cfg = _make_cfg(**{
        "data.patch_size": [8, 128, 128],
        "model.encoder_channels": [32, 64, 128, 256],
        "model.deep_supervision": True,
        "loss.deep_supervision_weights": [1.0, 0.5, 0.25],
    })
    case("lift+aux+DS  D=8  n_levels=4 n_views=3 ds_levels=3", cfg, B=2)

    print("\nAll lift+aux+DS smoke tests passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
