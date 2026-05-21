"""Smoke test for configs/seg2_5d_planA.yaml.

Validates Plan A lift (2.5D → 3D R(2+1)D) end-to-end without touching the
real lung dataset:

  1. ``load_config`` → ``sync()`` → ``validate()`` on the yaml as-is
     (catches divisibility / mutex / enum violations early).
  2. ``build_model(cfg)`` and parameter count.
  3. Single dummy forward + backward on the model's expected input shape
     ``(B, n_views, D, H, W)`` → ``(B, num_fg, D, H, W)``.
  4. ``build_loss(cfg.loss)`` wrapped in ``MultiResolutionLoss(num_res=1)``
     (the wrapper the trainer applies in lift mode) on raw integer labels.

Run:
    D:/miniconda/envs/torch27_env/python.exe test_planA_lift_config_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from segtask_v1.config import load_config
from segtask_v1.models.factory import build_model
from segtask_v1.losses.losses import MultiResolutionLoss, build_loss


def main() -> None:
    cfg_path = REPO_ROOT / "configs" / "seg2_5d_planA.yaml"
    print(f"[1/4] load_config({cfg_path.name}) + sync() + validate() …")
    cfg = load_config(str(cfg_path))
    # Sanity: print the fields that sync() derives — they MUST match the
    # values the new config relies on for lift mode.
    print(f"      data.patch_mode          = {cfg.data.patch_mode}")
    print(f"      data.patch_size          = {cfg.data.patch_size}")
    print(f"      data.multi_res_scales    = {cfg.data.multi_res_scales}")
    print(f"      data.aux_keep_native_d   = {cfg.data.aux_keep_native_d}")
    print(f"      data.z_boundary_mode     = {cfg.data.z_boundary_mode}")
    print(f"      model.backbone           = {cfg.model.backbone}")
    print(f"      model.block_type         = {cfg.model.block_type}")
    print(f"      model.lift_2_5d_to_3d    = {cfg.model.lift_2_5d_to_3d}")
    print(f"      model.spatial_dims       = {cfg.model.spatial_dims}")
    print(f"      model.in_channels        = {cfg.model.in_channels}")
    print(f"      model.aux_seg_supervision= {cfg.model.aux_seg_supervision}")
    print(f"      num_fg_classes (derived) = {cfg.num_fg_classes}")

    assert cfg.model.spatial_dims == 3, "lift requires spatial_dims=3"
    n_views = len(cfg.data.multi_res_scales)
    assert cfg.model.in_channels == n_views, (
        f"lift requires in_channels=n_views={n_views}, got "
        f"{cfg.model.in_channels}")
    assert cfg.model.block_type == "r2plus1d"
    D = int(cfg.data.patch_size[0])
    n_levels = len(cfg.model.encoder_channels)
    req = 1 << (n_levels - 1)
    assert D % req == 0, (
        f"patch_size[0]={D} must be divisible by 2**(n_levels-1)={req}")

    print(f"[2/4] build_model(cfg) …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      device = {device}")
    model = build_model(cfg).to(device).train()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      params: total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M")

    print(f"[3/4] dummy forward + backward …")
    # Use a tiny patch in-plane so this smoke fits on CPU / small GPU.
    # The forward path is identical to the production shape; only H, W
    # are scaled down. ``D`` stays at the configured value (16) so the
    # 4 downsamples actually halve it 4 times (16→8→4→2→1) — the
    # geometric path is fully exercised.
    B = 1
    H = W = 64
    image = torch.randn(B, n_views, D, H, W, device=device)
    # Raw integer labels in label_values = [0, 1, 2].
    label = torch.randint(0, 3, (B, 1, D, H, W), device=device,
                          dtype=torch.float32)

    pred = model(image)
    if isinstance(pred, list):
        # Single-resolution lift main path returns a tensor; if a list
        # comes back it'd indicate deep_supervision was on — fail loud.
        raise RuntimeError(
            "Unexpected list output from lift main path; expected a "
            "single tensor when deep_supervision=False.")
    expected = (B, cfg.num_fg_classes, D, H, W)
    assert tuple(pred.shape) == expected, (
        f"forward output shape {tuple(pred.shape)} != expected {expected}")
    print(f"      forward OK: input  {tuple(image.shape)}")
    print(f"                  output {tuple(pred.shape)}")

    print(f"[4/4] build_loss + MultiResolutionLoss(num_res=1) + backward …")
    # In lift mode the trainer wraps the base loss in
    # ``MultiResolutionLoss(num_res=1)`` and bypasses ``SliceChannelLoss``
    # entirely. Mirror that here so the smoke exercises the same loss
    # contract as the production trainer.
    base = build_loss(cfg.loss)
    criterion = MultiResolutionLoss(
        base,
        num_fg_classes=cfg.num_fg_classes,
        num_res=1,
        label_values=cfg.data.label_values)
    loss = criterion(pred.float(), label, weight_map=None)
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1e6)
    print(f"      loss = {loss.item():.6f}   grad_norm = {grad_norm:.4f}")

    print("\nALL OK — configs/seg2_5d_planA.yaml passes validate + build + "
          "forward + backward smoke.")


if __name__ == "__main__":
    main()
