"""仅绑定 127.0.0.1 的极简本地服务（Python 标准库 ``http.server``，零新增依赖）。

路由：

* ``GET  /``                 → 重定向到 ``/2_5d``；
* ``GET  /2_5d`` / ``/3d``    → 对应模式的单页 HTML；
* ``GET  /api/payload``       → per-mode 表单渲染载荷；
* ``GET  /api/base_configs``  → per-mode 可载入的分割模板列表；
* ``POST /api/load_base``     → 读模板 YAML → 全量值字典；
* ``POST /api/validate``      → 复用 ``Config.validate()``；
* ``POST /api/preview``       → 表单值 → 运行 YAML 文本；
* ``POST /api/launch``        → 落盘运行 YAML 并启动 train/predict 子进程；
* ``GET  /api/status``        → 当前运行状态；
* ``GET  /api/logs``          → 自游标起的增量日志；
* ``POST /api/stop``          → 停止运行中的进程。

安全：默认仅监听回环地址；仅本机单用户使用。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from . import build, page
from .process import MANAGER


class Handler(BaseHTTPRequestHandler):
    server_version = "SegtaskLauncher/1.0"

    # ---- helpers ----
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _html(self, html: str, code: int = 200) -> None:
        self._send(code, html.encode("utf-8"), "text/html; charset=utf-8")

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def log_message(self, *args: Any) -> None:  # 静默默认访问日志。
        pass

    # ---- GET ----
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path == "/":
                self.send_response(302)
                self.send_header("Location", "/2_5d")
                self.end_headers()
                return
            if path in ("/2_5d", "/3d"):
                mode = path.lstrip("/")
                self._html(page.render_page(mode))
                return
            if path == "/api/payload":
                mode = (qs.get("mode") or ["2_5d"])[0]
                self._json(build.build_payload(mode))
                return
            if path == "/api/base_configs":
                mode = (qs.get("mode") or [None])[0]
                self._json({"configs": build.list_base_configs(mode)})
                return
            if path == "/api/status":
                self._json(MANAGER.status())
                return
            if path == "/api/logs":
                since = int((qs.get("since") or ["0"])[0])
                self._json(MANAGER.get_logs(since))
                return
            self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    # ---- POST ----
    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = self._read_json()
        try:
            if path == "/api/load_base":
                vals = build.load_base_values(body.get("path", ""))
                self._json({"values": vals})
                return
            if path == "/api/validate":
                ok, msg = build.validate_values(body.get("values", {}))
                self._json({"ok": ok, "message": msg})
                return
            if path == "/api/preview":
                self._json({"yaml": build.values_to_yaml(body.get("values", {}))})
                return
            if path == "/api/launch":
                self._handle_launch(body)
                return
            if path == "/api/stop":
                self._json(MANAGER.stop())
                return
            self._json({"error": "not found"}, 404)
        except build._core_config.ConfigError as e:  # 校验类错误友好回传。
            self._json({"ok": False, "error": str(e)}, 200)
        except Exception as e:  # noqa: BLE001
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 200)

    def _handle_launch(self, body: Dict[str, Any]) -> None:
        task = body.get("task", "train")
        values = body.get("values", {})
        run_values = body.get("run_values", {})
        if task not in ("train", "predict"):
            self._json({"ok": False, "error": f"未知任务: {task}"})
            return
        # 落盘运行 YAML（内部已 validate）。
        yaml_path = build.write_run_yaml(values, task)
        rel = str(yaml_path.relative_to(build.REPO_ROOT)).replace("\\", "/")
        args = ["--config", rel]
        if task == "predict":
            args += build.predict_cli_args(run_values)
        result = MANAGER.start(task, args)
        if not result.get("ok"):
            self._json({"ok": False, "error": result.get("error", "启动失败")})
            return
        self._json({"ok": True, "cmd": result["cmd"], "yaml_path": rel})


def serve(host: str = "127.0.0.1", port: int = 8771) -> ThreadingHTTPServer:
    """创建并返回已绑定的服务器（调用方负责 serve_forever）。"""
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd


def run(host: str = "127.0.0.1", port: int = 8771,
        open_browser: bool = True) -> None:
    """启动服务并（可选）自动打开浏览器，阻塞直至 Ctrl-C。"""
    httpd = serve(host, port)
    url = f"http://{host}:{port}/2_5d"
    print(f"[segtask launcher] serving at {url}  (3D: http://{host}:{port}/3d)")
    print("[segtask launcher] press Ctrl-C to stop.")
    if open_browser:
        def _open() -> None:
            import webbrowser
            webbrowser.open(url)
        threading.Timer(0.6, _open).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[segtask launcher] shutting down…")
    finally:
        MANAGER.stop()
        httpd.shutdown()
        httpd.server_close()
