"""``python -m segtask_v1.launcher`` 入口：起本地服务并打开浏览器。

示例：

    python -m segtask_v1.launcher                 # 127.0.0.1:8771，自动开浏览器
    python -m segtask_v1.launcher --port 9000
    python -m segtask_v1.launcher --no-browser    # 不自动开浏览器（如远程转发）
"""

from __future__ import annotations

import argparse

from .server import run


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m segtask_v1.launcher",
        description="Segtask 训练/推理可视化启动器（本地 http.server + 单页 HTML）。")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址（默认仅本机 127.0.0.1）。")
    ap.add_argument("--port", type=int, default=8771, help="监听端口。")
    ap.add_argument("--no-browser", action="store_true",
                    help="不自动打开浏览器。")
    args = ap.parse_args()
    run(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
