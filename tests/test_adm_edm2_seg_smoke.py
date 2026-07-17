"""Smoke test for the new ADM / EDM2 segmentation arch (TODO #5).

Constructs each model directly from ``Config`` (no real data), runs a
random forward pass in both eval and train mode, and asserts:

  * eval mode without aux returns a tensor of the expected shape.
  * train mode with ``aux_seg_supervision`` returns a dict with
    ``main`` (tensor / list when DS is on) and ``aux`` (list of K-1
    tensors with the right per-view channel counts when
    ``keep_native_view_depth=True``).
  * deep supervision returns the right number of DS heads in main.

Usage (Windows powershell):
    & D:\\miniconda\\envs\\torch27_env\\python.exe \\
        d:\\codes\\work-projects\\SegTask\\test_adm_edm2_seg_smoke.py

Exits with non-zero on failure. No data dependencies.
"""

from __future__ import annotations

import logging
import sys
import traceback
from typing import Tuple

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke")


def _make_cfg(arch: str,
              n_views: int = 3,
              D: int = 8,
              H: int = 64, W: int = 64,
              encoder_channels=(16, 32, 64, 64, 64),
              aux: bool = True,
              ds: bool = False,
              keep_native_view_depth: bool = True,
              stem_fusion_mode: str = "multi_stem_proj",
              adm_linear_attention_levels=None):
    """Build a minimal :class:`Config` for the requested arch.

    Avoids depending on real data dirs by only constructing in-memory
    Config + sync(). validate() is also called to exercise the relaxed
    arch != 'unet' branch.
    """
    from taskcore.config.core import Config

    cfg = Config()
    # Data — minimal dummy paths (not loaded).
    cfg.data.image_dir = "F:/dummy/img"
    cfg.data.label_dir = "F:/dummy/lbl"
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.patch_size = [D, H, W]
    cfg.data.patch_mode = "2_5d"
    cfg.data.multi_res_scales = [1.0, 1.5, 2.0][:n_views] if n_views > 1 else [1.0]
    cfg.data.keep_native_view_depth = bool(keep_native_view_depth) and n_views > 1
    cfg.data.z_boundary_mode = (
        "edge_pad" if cfg.data.keep_native_view_depth else "stretch")
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    cfg.data.cache_mode = "memory"
    cfg.data.samples_per_volume = 1

    # Model — arch-specific.
    cfg.model.arch = arch
    cfg.model.encoder_channels = list(encoder_channels)
    cfg.model.encoder_blocks_per_stage = [2] * len(encoder_channels)
    cfg.model.decoder_blocks_per_stage = [1] * (len(encoder_channels) - 1)
    cfg.model.stem_mode = "conv3"
    cfg.model.stem_fusion_mode = stem_fusion_mode
    cfg.model.deep_supervision = bool(ds)
    cfg.model.aux_seg_supervision = bool(aux) and n_views > 1
    cfg.model.aux_head_mode = "linear"
    cfg.model.dropout = 0.0

    # Arch-specific knobs — pick small attn levels so the test stays cheap.
    if arch == "adm":
        # 5 levels → defaults to [3, 4]. For smaller H=64 (downsampled to
        # 64/16=4 at level 4 → /32 = 2 at level 4 actually with 5 stages
        # H goes 64,32,16,8,4 → fine). Constrain to [4] only to keep
        # the test fast.
        cfg.model.adm_attention_levels = [4]
        cfg.model.adm_num_heads = 2
        cfg.model.adm_num_head_channels = -1
        if adm_linear_attention_levels is not None:
            cfg.model.adm_linear_attention_levels = list(
                adm_linear_attention_levels)
            # Use head_dim=8 in the smoke test so hidden=32 stays small.
            cfg.model.adm_linear_attention_num_heads = 4
            cfg.model.adm_linear_attention_head_dim = 8
    elif arch == "edm2":
        cfg.model.edm2_attention_levels = [4]
        cfg.model.edm2_channels_per_head = 32  # match channel widths

    cfg.sync()
    cfg.validate()
    return cfg


def _make_input(cfg) -> Tuple[torch.Tensor, int, int, int]:
    """Build a random input matching ``cfg.model.in_channels``."""
    in_ch = int(cfg.model.in_channels)
    H = int(cfg.data.patch_size[1])
    W = int(cfg.data.patch_size[2])
    x = torch.randn(2, in_ch, H, W)
    return x, in_ch, H, W


def _check_main_shape(out, num_fg: int, D: int, H: int, W: int) -> torch.Tensor:
    """Pull the main tensor out of the (tensor | list | dict) contract."""
    if isinstance(out, dict):
        main = out["main"]
    else:
        main = out
    if isinstance(main, list):
        # DS path — main[0] = highest-res output.
        main_t = main[0]
    else:
        main_t = main
    expected = (2, num_fg * D, H, W)
    assert tuple(main_t.shape) == expected, (
        f"main output shape {tuple(main_t.shape)} != expected {expected}")
    return main_t


def _run_arch(arch: str, *, ds: bool, aux: bool, keep_native_view_depth: bool,
              stem_fusion_mode: str = "multi_stem_proj",
              adm_linear_attention_levels=None) -> None:
    cfg = _make_cfg(arch, ds=ds, aux=aux,
                    keep_native_view_depth=keep_native_view_depth,
                    stem_fusion_mode=stem_fusion_mode,
                    adm_linear_attention_levels=adm_linear_attention_levels)
    from taskcore.models.factory import build_model

    model = build_model(cfg)
    pc = model.param_count()
    logger.info("[%s] params total = %.2fM (ds=%s, aux=%s, native_d=%s, fusion=%s)",
                arch, pc["total"] / 1e6, ds, aux, keep_native_view_depth,
                stem_fusion_mode)

    x, in_ch, H, W = _make_input(cfg)
    D = int(cfg.data.patch_size[0])
    num_fg = int(cfg.data.num_classes - 1)
    n_views = max(len(cfg.data.multi_res_scales), 1)

    # Eval forward — must always return tensor (or list when DS, but
    # dict-returning aux is gated on training=True per the contract).
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert not isinstance(out_eval, dict), (
        f"eval forward should not return dict (aux gated on training). "
        f"got {type(out_eval).__name__}")
    _check_main_shape(out_eval, num_fg, D, H, W)
    logger.info("[%s][eval] main shape OK (%s)", arch,
                tuple((out_eval if not isinstance(out_eval, list)
                       else out_eval[0]).shape))

    # Train forward.
    model.train()
    out_train = model(x)
    if cfg.model.aux_seg_supervision and n_views > 1:
        assert isinstance(out_train, dict), (
            f"train forward with aux supervision should return dict; "
            f"got {type(out_train).__name__}")
        main = out_train["main"]
        aux_list = out_train["aux"]
        assert len(aux_list) == n_views - 1, (
            f"aux list length {len(aux_list)} != n_views-1 {n_views - 1}")
        if cfg.data.keep_native_view_depth:
            depths = list(cfg.per_view_depths)
            for k, ao in enumerate(aux_list, start=1):
                expected_ch = num_fg * depths[k]
                assert ao.shape == (2, expected_ch, H, W), (
                    f"aux[{k}] shape {tuple(ao.shape)} != "
                    f"(2, {expected_ch}, {H}, {W})")
            logger.info("[%s][train] aux native-D shapes OK: %s",
                        arch, [tuple(a.shape) for a in aux_list])
        else:
            for k, ao in enumerate(aux_list, start=1):
                assert ao.shape == (2, num_fg * D, H, W), (
                    f"aux[{k}] shape {tuple(ao.shape)} != "
                    f"(2, {num_fg * D}, {H}, {W})")
            logger.info("[%s][train] aux uniform-D shapes OK", arch)
    else:
        main = out_train

    # Deep supervision check.
    if cfg.model.deep_supervision:
        assert isinstance(main, list), (
            f"deep_supervision=True should produce a list of main outputs; "
            f"got {type(main).__name__}")
        # Existing convention: main[0] = highest res; main[1..] = lower.
        n_ds_expected = len(cfg.model.encoder_channels) - 2
        assert len(main) == 1 + n_ds_expected, (
            f"DS main length {len(main)} != 1 + n_ds_expected "
            f"{1 + n_ds_expected}")
        logger.info("[%s][train] DS main length OK (%d)", arch, len(main))
    _check_main_shape(out_train, num_fg, D, H, W)

    # Backward — check gradient flow (single fp32 mean loss, low cost).
    if isinstance(main, list):
        loss_t = sum(m.float().mean() for m in main)
    else:
        loss_t = main.float().mean()
    if isinstance(out_train, dict):
        loss_t = loss_t + sum(a.float().mean() for a in out_train["aux"])
    loss_t.backward()
    # Sanity check: at least one parameter received a gradient.
    has_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters())
    assert has_grad, "no parameter received a finite gradient on backward()"
    logger.info("[%s] backward OK (loss=%.4f)", arch, float(loss_t.item()))


def main() -> int:
    failures = 0
    cases = [
        # (arch, ds, aux, native_d, fusion)
        ("adm",  False, True,  True,  "multi_stem_proj"),
        ("adm",  True,  True,  True,  "multi_stem_proj"),
        ("adm",  False, True,  False, "multi_stem_proj"),
        ("adm",  False, True,  False, "shared_stem"),
        ("edm2", False, True,  True,  "multi_stem_proj"),
        ("edm2", True,  True,  True,  "multi_stem_proj"),
        ("edm2", False, True,  False, "multi_stem_proj"),
        ("edm2", False, True,  False, "shared_stem"),
        # Single-FOV smoke (n_views=1 → aux disabled by builder).
        # Not in the matrix — covered by the test below.
    ]
    for arch, ds, aux, native_d, fusion in cases:
        try:
            _run_arch(arch, ds=ds, aux=aux,
                      keep_native_view_depth=native_d,
                      stem_fusion_mode=fusion)
        except Exception:
            failures += 1
            logger.error("FAIL: arch=%s ds=%s aux=%s native_d=%s fusion=%s\n%s",
                         arch, ds, aux, native_d, fusion,
                         traceback.format_exc())

    # ADM + LinearAttention smoke matrix.
    lin_cases = [
        # (label, lin_levels)
        ("light",     [0, 1]),                # shallow only
        ("full",      [0, 1, 2, 3]),          # lucidrains-style every-level
        ("overlap",   [3, 4]),                # overlap with softmax-attn levels
    ]
    for label, lin_levels in lin_cases:
        try:
            _run_arch("adm", ds=False, aux=True, keep_native_view_depth=True,
                      stem_fusion_mode="multi_stem_proj",
                      adm_linear_attention_levels=lin_levels)
            logger.info("[adm] linear-attn (%s, levels=%s) smoke OK",
                        label, lin_levels)
        except Exception:
            failures += 1
            logger.error("FAIL: adm linear-attn label=%s levels=%s\n%s",
                         label, lin_levels, traceback.format_exc())

    # Single-FOV (n_views=1) smoke — aux disabled automatically.
    for arch in ("adm", "edm2"):
        try:
            cfg = _make_cfg(arch, n_views=1, aux=False, keep_native_view_depth=False,
                            stem_fusion_mode="multi_stem_proj")
            from taskcore.models.factory import build_model
            model = build_model(cfg)
            x, _, _, _ = _make_input(cfg)
            model.eval()
            with torch.no_grad():
                out = model(x)
            assert not isinstance(out, dict)
            num_fg = int(cfg.data.num_classes - 1)
            D = int(cfg.data.patch_size[0])
            H = int(cfg.data.patch_size[1])
            W = int(cfg.data.patch_size[2])
            _check_main_shape(out, num_fg, D, H, W)
            logger.info("[%s] single-FOV smoke OK", arch)
        except Exception:
            failures += 1
            logger.error("FAIL: %s single-FOV\n%s", arch, traceback.format_exc())

    if failures:
        logger.error("==== %d cases failed ====", failures)
        return 1
    logger.info("==== all smoke cases passed ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
