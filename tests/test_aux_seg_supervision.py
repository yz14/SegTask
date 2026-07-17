"""Smoke test for 2.5D multi-FOV auxiliary segmentation supervision.

Exercises the full feature surface added by the aux_seg_supervision change:

1. Config sync/validate — both Plan A (multi_stem_proj) and Plan C
   (hierarchical) aux topologies, with custom + auto weights.
2. UNet3D forward — verifies dict {"main", "aux"} output in train mode,
   single-tensor output in eval mode, and correct (B, num_fg*D, H, W)
   shapes for every aux head.
3. Loss aggregation — runs the trainer's ``_compute_loss_aux_fp32`` style
   path on synthetic tensors and checks gradient flow into aux heads.
4. Backward compatibility — n_views==1 and aux_seg_supervision=False both
   keep the legacy single-tensor / list contract.

Run:
    conda activate torch27_env
    python test_aux_seg_supervision.py
"""

from __future__ import annotations

import sys
import traceback
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from taskcore.config.core import (
    Config, DataConfig, ModelConfig, LossConfig, TrainConfig, AugConfig,
    PredictConfig,
)
from segtask_v1.losses.losses import SliceChannelLoss, build_loss
from taskcore.models.factory import build_model


def _make_cfg(
    *,
    multi_res_scales,
    stem_fusion_mode: str,
    aux_seg_supervision: bool,
    aux_weights=None,
    aux_head_mode: str = "linear",
    deep_supervision: bool = False,
    patch_size=(8, 64, 64),
    encoder_channels=(16, 32, 64, 128),
):
    """Build a minimal 2.5D Config that passes sync/validate."""
    cfg = Config(
        data=DataConfig(
            image_dir="dummy",
            label_dir="dummy",
            label_values=[0, 1, 2],
            num_classes=3,
            patch_size=list(patch_size),
            patch_mode="2_5d",
            multi_res_scales=list(multi_res_scales),
        ),
        augment=AugConfig(enabled=False),
        model=ModelConfig(
            encoder_channels=list(encoder_channels),
            blocks_per_level=1,
            stem_fusion_mode=stem_fusion_mode,
            aux_seg_supervision=aux_seg_supervision,
            aux_head_mode=aux_head_mode,
            deep_supervision=deep_supervision,
            stem_mode="conv3",
            decoder_type="unet",
        ),
        loss=LossConfig(
            name="dice_bce",
            slice_loss_reduction="per_volume",
            aux_supervision_weights=list(aux_weights) if aux_weights else [],
        ),
        train=TrainConfig(epochs=1, output_dir=str(ROOT / "outputs" / "tmp")),
        predict=PredictConfig(),
    )
    cfg.sync()
    cfg.validate()
    return cfg


def _make_inputs(cfg: Config, device):
    """Synthesise (image_2d, label_views, wmap_views) on device.

    Mirrors the trainer's post-augment, post-_squeeze_2_5d_keep_views
    layout: image is rank-4, labels/wmaps are rank-5 (B, C_res, D, H, W).
    """
    B = 2
    C_res = len(cfg.data.multi_res_scales)
    D, H, W = cfg.data.patch_size
    fg_vals = cfg.data.label_values[1:]
    image = torch.randn(B, C_res * D, H, W, device=device)
    # Random integer labels with values from label_values.
    label_views = torch.randint(
        0, len(cfg.data.label_values), (B, C_res, D, H, W),
        device=device, dtype=torch.long)
    # Force at least one fg voxel per (sample, view) so dice doesn't hit
    # smooth ≈ 1 and zero out the gradient.
    label_views[:, :, 0, 0, 0] = fg_vals[0]
    wmap_views = torch.ones(B, C_res, D, H, W, device=device)
    return image, label_views, wmap_views


def _check_aux_forward(cfg, label, expected_n_aux, device):
    """Build model, forward in train+eval, assert dict / shape contract."""
    model = build_model(cfg).to(device)
    image, label_views, wmap_views = _make_inputs(cfg, device)
    B = image.shape[0]
    num_fg = cfg.num_fg_classes
    D, H, W = cfg.data.patch_size
    expected_main_shape = (B, num_fg * D, H, W)

    # ---- training mode ---------------------------------------------------
    model.train()
    out_train = model(image)
    if expected_n_aux > 0:
        assert isinstance(out_train, dict), (
            f"[{label}] train output must be dict when aux enabled; "
            f"got {type(out_train).__name__}")
        main = out_train["main"]
        aux = out_train["aux"]
        assert isinstance(aux, list) and len(aux) == expected_n_aux, (
            f"[{label}] expected {expected_n_aux} aux outputs; got "
            f"{len(aux) if isinstance(aux, list) else aux}")
        for k, ao in enumerate(aux, start=1):
            assert tuple(ao.shape) == expected_main_shape, (
                f"[{label}] aux head {k} shape {tuple(ao.shape)} "
                f"!= expected {expected_main_shape}")
    else:
        assert not isinstance(out_train, dict), (
            f"[{label}] no-aux config must NOT return dict; got dict.")
        main = out_train

    # main_path can be tensor or DS list — collapse to tensor.
    main_tensor = main[0] if isinstance(main, list) else main
    assert tuple(main_tensor.shape) == expected_main_shape, (
        f"[{label}] main shape {tuple(main_tensor.shape)} "
        f"!= expected {expected_main_shape}")

    # ---- eval mode -------------------------------------------------------
    model.eval()
    with torch.no_grad():
        out_eval = model(image)
    assert not isinstance(out_eval, dict), (
        f"[{label}] eval mode must NOT return dict; got dict.")
    eval_tensor = out_eval[0] if isinstance(out_eval, list) else out_eval
    assert tuple(eval_tensor.shape) == expected_main_shape, (
        f"[{label}] eval shape {tuple(eval_tensor.shape)} "
        f"!= expected {expected_main_shape}")

    # ---- training-time loss + backward ----------------------------------
    if expected_n_aux > 0:
        model.train()
        out = model(image)
        # Build losses analogous to Trainer.
        base_loss = build_loss(cfg.loss)
        sc_loss = SliceChannelLoss(
            base_loss=base_loss,
            num_fg_classes=num_fg,
            num_slices=D,
            label_values=cfg.data.label_values,
            reduction=cfg.loss.slice_loss_reduction,
        )
        # Aux weights — mimic Trainer auto-fill.
        aux_w = cfg.loss.aux_supervision_weights or [
            0.5 ** (k + 1) for k in range(expected_n_aux)
        ]
        # Loss = main(view0) + Σ w_k aux(view k).
        main_path = out["main"]
        if isinstance(main_path, list):
            main_pred = main_path[0]
        else:
            main_pred = main_path
        loss = sc_loss(
            main_pred.float(), label_views[:, 0],
            weight_map=wmap_views[:, 0].float(),
        )
        for k_idx, ap in enumerate(out["aux"]):
            loss = loss + aux_w[k_idx] * sc_loss(
                ap.float(), label_views[:, k_idx + 1],
                weight_map=wmap_views[:, k_idx + 1].float(),
            )
        loss.backward()

        # Verify each aux head received gradient (non-zero somewhere).
        for k, head in enumerate(model.aux_heads, start=1):
            grads = [p.grad for p in head.parameters() if p.grad is not None]
            assert grads, f"[{label}] aux head {k} got no gradients"
            assert any(g.abs().sum().item() > 0 for g in grads), (
                f"[{label}] aux head {k} gradients are all zero")

    # Param count sanity — aux heads should add a few thousand params.
    pc = model.param_count()
    return pc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}, torch={torch.__version__}")

    cases = [
        # (label, kwargs, expected_n_aux)
        ("legacy n_views=1, aux off",
         dict(multi_res_scales=[1.0], stem_fusion_mode="multi_stem_proj",
              aux_seg_supervision=False),
         0),
        ("Plan A (multi_stem_proj), aux off",
         dict(multi_res_scales=[1.0, 1.5, 2.0],
              stem_fusion_mode="multi_stem_proj",
              aux_seg_supervision=False),
         0),
        ("Plan A (multi_stem_proj), aux on, default weights",
         dict(multi_res_scales=[1.0, 1.5, 2.0],
              stem_fusion_mode="multi_stem_proj",
              aux_seg_supervision=True),
         2),
        ("Plan A (multi_stem_proj), aux on, custom weights, +DS",
         dict(multi_res_scales=[1.0, 1.5, 2.0],
              stem_fusion_mode="multi_stem_proj",
              aux_seg_supervision=True,
              aux_weights=[0.4, 0.2],
              deep_supervision=True),
         2),
        ("Plan A (shared_stem), aux on",
         dict(multi_res_scales=[1.0, 1.5],
              stem_fusion_mode="shared_stem",
              aux_seg_supervision=True),
         1),
        ("Plan C (hierarchical), aux on, head=linear",
         dict(multi_res_scales=[1.0, 1.5, 2.0],
              stem_fusion_mode="hierarchical",
              aux_seg_supervision=True,
              aux_head_mode="linear"),
         2),
        ("Plan C (hierarchical), aux on, head=conv",
         dict(multi_res_scales=[1.0, 1.5, 2.0],
              stem_fusion_mode="hierarchical",
              aux_seg_supervision=True,
              aux_head_mode="conv"),
         2),
        ("Plan A (multi_stem_proj), aux on, head=conv",
         dict(multi_res_scales=[1.0, 1.5],
              stem_fusion_mode="multi_stem_proj",
              aux_seg_supervision=True,
              aux_head_mode="conv"),
         1),
    ]

    n_pass = 0
    n_fail = 0
    for label, kw, expected_n_aux in cases:
        print(f"\n[case] {label}")
        try:
            cfg = _make_cfg(**kw)
            pc = _check_aux_forward(cfg, label, expected_n_aux, device)
            print(f"  OK — total_params={pc['total']/1e6:.2f}M, "
                  f"expected_aux_heads={expected_n_aux}")
            n_pass += 1
        except Exception:
            traceback.print_exc()
            n_fail += 1

    print(f"\n[smoke] passed={n_pass}, failed={n_fail}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
