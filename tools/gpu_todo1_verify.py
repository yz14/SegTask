"""TODO 1 待 GPU 验证项 — 本地一次性冒烟。

用法:
    D:\\miniconda\\envs\\torch27_env\\python.exe tools/gpu_todo1_verify.py
"""

from __future__ import annotations

import gc
import logging
import sys
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("gpu_todo1_verify")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAILURES: list[str] = []


def _mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _reset_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _record(name: str, fn) -> None:
    log.info("=== %s ===", name)
    try:
        fn()
        log.info("[PASS] %s", name)
    except Exception:
        FAILURES.append(name)
        log.error("[FAIL] %s\n%s", name, traceback.format_exc())


class _WorkerDummy(Dataset):
    """spawn worker 可 pickle 的最小 dataset。"""

    def __len__(self) -> int:
        return 16

    def __getitem__(self, idx: int):
        return {"x": torch.tensor(float(idx)), "idx": idx}


def test_source_tagged_dataloader_workers() -> None:
    from taskcore.data.mixed_sampler import SOURCE_PRIMARY, SourceTaggedDataset

    ds = SourceTaggedDataset(_WorkerDummy(), SOURCE_PRIMARY)
    dl = DataLoader(ds, batch_size=4, num_workers=2, persistent_workers=False)
    batch = next(iter(dl))
    assert batch["source"].tolist() == [0, 0, 0, 0]
    assert len(batch["x"]) == 4


def _make_edm2_cfg():
    from taskcore.config.core import Config

    cfg = Config()
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.patch_size = [8, 64, 64]
    cfg.data.patch_mode = "2_5d"
    cfg.data.multi_res_scales = [1.0]
    cfg.data.batch_size = 2
    cfg.model.arch = "edm2"
    cfg.model.encoder_channels = [16, 32, 64, 64, 64]
    cfg.model.encoder_blocks_per_stage = [1] * 5
    cfg.model.decoder_blocks_per_stage = [1] * 4
    cfg.model.edm2_attention_levels = [4]
    cfg.model.edm2_channels_per_head = 32
    cfg.model.dropout = 0.0
    cfg.train.compile_mode = "default"
    cfg.train.use_amp = True
    cfg.train.amp_dtype = "bf16"
    cfg.sync()
    cfg.validate()
    return cfg


def _edm2_forward_backward(compiled: bool, steps: int = 3) -> float:
    from taskcore.models.factory import build_model

    cfg = _make_edm2_cfg()
    if not compiled:
        cfg.train.compile_mode = "none"
    model = build_model(cfg).to(DEVICE)
    if compiled and hasattr(torch, "compile"):
        model = torch.compile(model, mode="default")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16: no scaler
    in_ch = int(cfg.model.in_channels)
    H, W = int(cfg.data.patch_size[1]), int(cfg.data.patch_size[2])
    _reset_mem()
    for step in range(steps):
        x = torch.randn(2, in_ch, H, W, device=DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        main = out["main"] if isinstance(out, dict) else out
        if isinstance(main, list):
            main = main[0]
        loss = main.float().mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        log.info("  step %d loss=%.4f", step, float(loss))
    peak = _mem_mb()
    del model, opt
    _reset_mem()
    return peak


def test_edm2_bf16_eager() -> None:
    peak = _edm2_forward_backward(compiled=False)
    log.info("  peak_mem=%.0f MB (eager bf16)", peak)
    assert peak < 15000, f"OOM risk: peak {peak:.0f} MB on 16G card"


def _triton_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("triton") is not None


def test_edm2_bf16_compile() -> None:
    if not hasattr(torch, "compile"):
        log.warning("  torch.compile unavailable, skip")
        return
    if not _triton_available():
        log.warning(
            "  Triton not installed (Windows 常见) — 与 BaseTrainer 一致回退 eager，跳过 compile 实测")
        return
    peak = _edm2_forward_backward(compiled=True)
    log.info("  peak_mem=%.0f MB (compile bf16)", peak)
    assert peak < 15000, f"OOM risk: peak {peak:.0f} MB on 16G card"


def test_adm_compile_smoke() -> None:
    from taskcore.config.core import Config
    from taskcore.models.factory import build_model

    cfg = Config()
    cfg.data.label_values = [0, 1]
    cfg.data.num_classes = 2
    cfg.data.patch_size = [8, 64, 64]
    cfg.data.patch_mode = "2_5d"
    cfg.data.multi_res_scales = [1.0]
    cfg.model.arch = "adm"
    cfg.model.encoder_channels = [16, 32, 64, 64]
    cfg.model.encoder_blocks_per_stage = [1] * 4
    cfg.model.decoder_blocks_per_stage = [1] * 3
    cfg.model.adm_attention_levels = [3]
    cfg.model.dropout = 0.0
    cfg.sync()
    cfg.validate()
    model = build_model(cfg).to(DEVICE)
    if hasattr(torch, "compile") and _triton_available():
        model = torch.compile(model, mode="default")
    elif hasattr(torch, "compile"):
        log.warning("  Triton missing — ADM compile skipped (BaseTrainer 同样回退 eager)")
        return
    model.train()
    in_ch = int(cfg.model.in_channels)
    x = torch.randn(1, in_ch, 64, 64, device=DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
    main = out["main"] if isinstance(out, dict) else out
    if isinstance(main, list):
        main = main[0]
    loss = main.float().mean()
    loss.backward()
    log.info("  adm compile loss=%.4f peak_mem=%.0f MB", float(loss), _mem_mb())


def test_diffusion_backbone_smoke() -> None:
    from taskcore.models.adm_unet import build_adm_diffusion_unet
    from taskcore.models.edm2_unet import build_edm2_diffusion_unet

    for arch, builder in (("adm", build_adm_diffusion_unet),
                          ("edm2", build_edm2_diffusion_unet)):
        from taskcore.config.core import Config
        cfg = Config()
        cfg.data.patch_size = [4, 32, 32]
        cfg.data.patch_mode = "2_5d"
        cfg.data.multi_res_scales = [1.0]
        cfg.data.label_values = [0, 1]
        cfg.data.num_classes = 2
        cfg.model.arch = arch
        cfg.model.encoder_channels = [16, 32, 64]
        cfg.model.encoder_blocks_per_stage = [1] * 3
        cfg.model.decoder_blocks_per_stage = [1] * 2
        cfg.model.dropout = 0.0
        if arch == "adm":
            cfg.model.adm_attention_levels = [2]
        else:
            cfg.model.edm2_attention_levels = [2]
            cfg.model.edm2_channels_per_head = 32
        cfg.sync()
        cfg.validate()
        d = 4
        net = builder(cfg, in_channels=2 * d, out_channels=d).to(DEVICE)
        net.train()
        xc = torch.randn(2, 2 * d, 32, 32, device=DEVICE)
        cn = torch.rand(2, device=DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = net(xc, cn)
        assert tuple(out.shape) == (2, d, 32, 32), out.shape
        out.float().sum().backward()
        log.info("  diffusion backbone %s OK, peak_mem=%.0f MB", arch, _mem_mb())


def test_edm2_realistic_memory() -> None:
    """接近 seg2_5d_edm2 通道宽度，缩 patch 到 16G 可承受。"""
    from taskcore.config.core import Config
    from taskcore.models.factory import build_model

    cfg = Config()
    cfg.data.label_values = [0, 1, 2]
    cfg.data.num_classes = 3
    cfg.data.patch_size = [12, 128, 128]
    cfg.data.patch_mode = "2_5d"
    cfg.data.multi_res_scales = [1.0, 1.5, 2.0]
    cfg.data.keep_native_view_depth = True
    cfg.data.batch_size = 1
    cfg.model.aux_seg_supervision = True
    cfg.model.arch = "edm2"
    cfg.model.encoder_channels = [32, 64, 128, 256, 320]
    cfg.model.encoder_blocks_per_stage = [2, 2, 2, 2, 2]
    cfg.model.decoder_blocks_per_stage = [2, 2, 2, 2]
    cfg.model.edm2_attention_levels = [4]
    cfg.model.dropout = 0.0
    cfg.sync()
    cfg.validate()
    model = build_model(cfg).to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    in_ch = int(cfg.model.in_channels)
    H, W = 128, 128
    _reset_mem()
    x = torch.randn(1, in_ch, H, W, device=DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x)
    main = out["main"] if isinstance(out, dict) else out
    if isinstance(main, list):
        main = main[0]
    loss = main.float().mean()
    loss.backward()
    opt.step()
    peak = _mem_mb()
    log.info("  realistic edm2 3-view 128² peak_mem=%.0f MB", peak)
    assert peak < 15500, f"realistic config OOM risk: {peak:.0f} MB"


def main() -> int:
    if not torch.cuda.is_available():
        log.error("CUDA not available — aborting GPU verify")
        return 1
    props = torch.cuda.get_device_properties(0)
    log.info("GPU: %s (%.1f GB)", props.name, props.total_memory / 1e9)

    _record("SourceTaggedDataset + DataLoader workers", test_source_tagged_dataloader_workers)
    _record("EDM2 bf16 eager (MPConv in-place)", test_edm2_bf16_eager)
    _record("EDM2 bf16 + torch.compile", test_edm2_bf16_compile)
    _record("ADM + torch.compile", test_adm_compile_smoke)
    _record("Diffusion backbone ADM/EDM2", test_diffusion_backbone_smoke)
    _record("EDM2 realistic memory (3-view 128²)", test_edm2_realistic_memory)

    if FAILURES:
        log.error("FAILED: %s", ", ".join(FAILURES))
        return 1
    log.info("All GPU TODO1 checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
