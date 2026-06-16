#!/usr/bin/env python
"""Checkpoint 完整性检查与诊断脚本。

用法：
    python scripts/inspect_checkpoint.py outputs/body_resnet_bnorm/best_model.pth
    python scripts/inspect_checkpoint.py outputs/body_resnet_bnorm/  # 扫描目录
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import torch


def _format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def inspect_zip_structure(path: str) -> None:
    """用 zipfile 检查 PyTorch ZIP 存档结构。"""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            namelist = zf.namelist()
            print(f"  ZIP entries: {len(namelist)}")
            # 列出前 20 个
            for name in namelist[:20]:
                info = zf.getinfo(name)
                print(f"    - {name}: {_format_size(info.file_size)} "
                      f"(compressed {_format_size(info.compress_size)})")
            if len(namelist) > 20:
                print(f"    ... 还有 {len(namelist) - 20} 个条目")

            # 重点检查 data/0
            data0 = [n for n in namelist if n == "data/0"]
            if not data0:
                print("  [WARN] 未找到 data/0 — 这不是标准的 PyTorch 1.6+ ZIP 格式")
            else:
                info = zf.getinfo("data/0")
                print(f"  data/0: file_size={_format_size(info.file_size)}, "
                      f"compress_size={_format_size(info.compress_size)}, "
                      f"compress_type={info.compress_type}")
                # 尝试读取头几个字节
                try:
                    with zf.open("data/0") as f:
                        header = f.read(8)
                        print(f"  data/0 header (8 bytes): {header.hex()}")
                except zipfile.BadZipFile as e:
                    print(f"  [ERROR] 无法读取 data/0: {e}")
    except zipfile.BadZipFile as e:
        print(f"  [ERROR] 不是有效的 ZIP 文件: {e}")
    except Exception as e:
        print(f"  [ERROR] ZIP 检查异常: {e}")


def inspect_checkpoint(path: str) -> bool:
    """检查单个 checkpoint 文件，返回是否可正常 torch.load。"""
    print(f"\n{'=' * 60}")
    print(f"检查: {path}")
    print(f"{'=' * 60}")

    # 1. 基本文件信息
    if not os.path.exists(path):
        print("[ERROR] 文件不存在")
        return False

    if not os.path.isfile(path):
        print("[ERROR] 不是文件")
        return False

    size = os.path.getsize(path)
    print(f"文件大小: {_format_size(size)} ({size} bytes)")
    print(f"绝对路径: {os.path.abspath(path)}")

    if size == 0:
        print("[ERROR] 文件大小为 0，空文件")
        return False

    if size < 1024 * 1024:
        print("[WARN] 文件小于 1MB，很可能是损坏或不完整的 checkpoint")

    # 2. ZIP 结构检查
    print("\n--- ZIP 结构检查 ---")
    inspect_zip_structure(path)

    # 3. torch.load 检查
    print("\n--- torch.load 检查 ---")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        print("[OK] torch.load 成功")

        print(f"  checkpoint keys: {list(ckpt.keys())}")

        # model_state_dict
        sd = ckpt.get("model_state_dict")
        if sd is None:
            print("  [WARN] 缺少 'model_state_dict'")
        else:
            n_params = len(sd)
            total_elems = sum(v.numel() for v in sd.values())
            total_bytes = sum(v.numel() * v.element_size() for v in sd.values())
            print(f"  model_state_dict: {n_params} 个参数, "
                  f"{total_elems} 个元素, {_format_size(total_bytes)}")
            print(f"  示例 key: {list(sd.keys())[:5]}")

        # ema
        ema = ckpt.get("ema_state_dict")
        if ema is None:
            print("  [INFO] 无 'ema_state_dict'")
        else:
            print(f"  ema_state_dict: 存在 (类型 {type(ema).__name__})")

        # meta
        print(f"  epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"  best_metric: {ckpt.get('best_metric', 'N/A')}")
        print(f"  optimizer_state_dict: {'存在' if 'optimizer_state_dict' in ckpt else '缺失'}")

        return True

    except RuntimeError as e:
        print(f"[ERROR] torch.load 失败 (RuntimeError): {e}")
        return False
    except Exception as e:
        print(f"[ERROR] torch.load 失败 ({type(e).__name__}): {e}")
        return False


def scan_directory(dir_path: str) -> None:
    """扫描目录下所有 .pth 文件。"""
    p = Path(dir_path)
    if not p.is_dir():
        print(f"[ERROR] 不是目录: {dir_path}")
        return

    pth_files = sorted(p.rglob("*.pth"))
    print(f"\n在 {dir_path} 下找到 {len(pth_files)} 个 .pth 文件:")
    for f in pth_files:
        size = f.stat().st_size
        ok = "OK" if size > 1024 * 1024 else "SMALL"
        print(f"  {f.relative_to(p)}: {_format_size(size)} [{ok}]")

    print("\n开始逐个检查...")
    ok_count = 0
    for f in pth_files:
        if inspect_checkpoint(str(f)):
            ok_count += 1
    print(f"\n{'=' * 60}")
    print(f"检查结果: {ok_count}/{len(pth_files)} 个 checkpoint 可正常加载")


def main():
    if len(sys.argv) < 2:
        target = "outputs/body_resnet_bnorm/best_model.pth"
        print(f"未提供参数，默认检查: {target}")
    else:
        target = sys.argv[1]

    if os.path.isdir(target):
        scan_directory(target)
    else:
        inspect_checkpoint(target)


if __name__ == "__main__":
    main()
