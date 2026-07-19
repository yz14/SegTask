# -*- coding: utf-8 -*-
"""O8 观察项基准脚本：测 dataset.resize_3d 里 scipy.ndimage.zoom 的单次耗时，
并对比候选后端（torch.nn.functional.interpolate CPU / CUDA）。

用法（按你的真实 patch 尺寸改参数即可）：
    D:\\miniconda\\envs\\torch27_env\\python.exe tools\\bench_zoom_o8.py ^
        --src 24 96 96 --dst 16 64 64 --label-channels 2 --repeat 20

输出各后端的单次耗时（中位数）与相对 scipy 的加速比。
判读：若 scipy zoom 单次耗时 × 每步样本数 相比训练 step 时间占比很小（<5%），
O8 不值得做；若占比高且 torch CPU/CUDA 明显更快，则值得换后端。
"""
from __future__ import annotations

import argparse
import time
import numpy as np
from scipy.ndimage import zoom


def _bench(fn, repeat: int) -> float:
    fn()  # 预热
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0  # ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, nargs=3, default=[24, 96, 96],
                    help="裁剪出的原始 patch (D H W)，即 zoom 的输入尺寸")
    ap.add_argument("--dst", type=int, nargs=3, default=[16, 64, 64],
                    help="网络输入 patch (D H W)，即 zoom 的目标尺寸")
    ap.add_argument("--label-channels", type=int, default=2,
                    help="前景类数（label 二值堆叠的通道数）")
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()

    sd, sh, sw = args.src
    dd, dh, dw = args.dst
    rng = np.random.default_rng(0)
    img = rng.standard_normal((sd, sh, sw)).astype(np.float32)
    lab = (rng.random((args.label_channels, sd, sh, sw)) > 0.9).astype(np.float32)
    f_img = [dd / sd, dh / sh, dw / sw]
    f_lab = [1.0] + f_img

    results = {}
    results["scipy zoom image(order=1)"] = _bench(
        lambda: zoom(img, f_img, order=1, mode="nearest"), args.repeat)
    results["scipy zoom label(order=0)"] = _bench(
        lambda: zoom(lab, f_lab, order=0, mode="nearest"), args.repeat)

    try:
        import torch
        import torch.nn.functional as F
        t_img = torch.from_numpy(img)[None, None]
        t_lab = torch.from_numpy(lab)[None]
        results["torch CPU trilinear image"] = _bench(
            lambda: F.interpolate(t_img, size=(dd, dh, dw), mode="trilinear",
                                  align_corners=False).numpy(), args.repeat)
        results["torch CPU nearest label"] = _bench(
            lambda: F.interpolate(t_lab, size=(dd, dh, dw),
                                  mode="nearest-exact").numpy(), args.repeat)
        if torch.cuda.is_available():
            g_img = t_img.cuda()
            g_lab = t_lab.cuda()

            def _gpu_img():
                F.interpolate(g_img, size=(dd, dh, dw), mode="trilinear",
                              align_corners=False)
                torch.cuda.synchronize()

            def _gpu_lab():
                F.interpolate(g_lab, size=(dd, dh, dw), mode="nearest-exact")
                torch.cuda.synchronize()

            results["torch CUDA trilinear image"] = _bench(_gpu_img, args.repeat)
            results["torch CUDA nearest label"] = _bench(_gpu_lab, args.repeat)
    except ImportError:
        print("torch 不可用，仅测 scipy。")

    base = results["scipy zoom image(order=1)"]
    print(f"\nsrc={tuple(args.src)} -> dst={tuple(args.dst)}, "
          f"label_channels={args.label_channels}, repeat={args.repeat} (中位数)")
    for k, v in results.items():
        print(f"  {k:32s} {v:8.2f} ms  (x{base / v:5.1f} vs scipy image)")
    total = results["scipy zoom image(order=1)"] + results["scipy zoom label(order=0)"]
    print(f"\n每个样本 scipy 缩放总耗时约 {total:.2f} ms。"
          f"对照你的训练 step 耗时估算占比：占比 <5% 则 O8 不值得做。")


if __name__ == "__main__":
    main()
