"""跨平台子进程管理：启动 train/predict、实时收集日志、停止。

单实例模型（``RunManager`` 单例）：本启动器只服务本机单用户，同一时刻只允许一个
运行中的训练/推理进程，避免 GPU 抢占与日志串流。前端轮询 ``/api/logs`` 增量拉取。

跨平台要点：
* 用 ``sys.executable -m segtask_v1.train/predict`` 调用，避免依赖 PATH 里的
  ``python``；
* ``cwd=REPO_ROOT`` 保证 ``-m`` 包可见、相对路径（configs/…）一致；
* 子进程输出行缓冲 + 守护线程读取，写入有界 ``deque``（防内存膨胀）；
* 停止：POSIX 用进程组（``start_new_session`` + ``killpg``）确保 DDP 子进程一并
  退出；Windows 用 ``CREATE_NEW_PROCESS_GROUP`` + ``taskkill /T`` 杀整棵树。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
_MAX_LINES = 5000  # 日志环形缓冲上限。


class RunManager:
    """单运行实例的生命周期与日志缓冲管理（线程安全）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._lines: deque = deque(maxlen=_MAX_LINES)
        self._seq = 0                 # 已产出的总行数（含被挤出的），用于增量游标。
        self._task = ""               # "train" / "predict"
        self._cmd: List[str] = []
        self._started_at = 0.0
        self._reader: Optional[threading.Thread] = None

    # ---------------------------------------------------------------- status
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            running = self._proc is not None and self._proc.poll() is None
            returncode = None if self._proc is None else self._proc.poll()
            return {
                "running": running,
                "task": self._task,
                "cmd": self._cmd,
                "returncode": returncode,
                "started_at": self._started_at,
                "total_lines": self._seq,
            }

    def get_logs(self, since: int = 0) -> Dict[str, Any]:
        """返回自 ``since``（全局行号）以来的新增日志行与新游标。"""
        with self._lock:
            start_global = self._seq - len(self._lines)
            if since < start_global:
                since = start_global  # 被挤出的行无法回放。
            offset = since - start_global
            new_lines = list(self._lines)[offset:]
            return {
                "lines": new_lines,
                "next": self._seq,
                "running": self._proc is not None and self._proc.poll() is None,
                "returncode": None if self._proc is None else self._proc.poll(),
            }

    # ----------------------------------------------------------------- start
    def start(self, task: str, args: List[str]) -> Dict[str, Any]:
        """启动 ``python -m segtask_v1.<task> <args>``。已有运行中实例则拒绝。"""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return {"ok": False, "error": "已有运行中的任务，请先停止。"}
            module = {"train": "segtask_v1.train",
                      "predict": "segtask_v1.predict"}.get(task)
            if module is None:
                return {"ok": False, "error": f"未知任务: {task}"}
            cmd = [sys.executable, "-u", "-m", module, *args]
            popen_kwargs: Dict[str, Any] = dict(
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                encoding="utf-8",
                errors="replace",
            )
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            popen_kwargs["env"] = env
            if os.name == "nt":
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP)  # type: ignore[attr-defined]
            else:
                popen_kwargs["start_new_session"] = True
            try:
                self._proc = subprocess.Popen(cmd, **popen_kwargs)
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "error": f"启动失败: {e}"}
            self._lines.clear()
            self._seq = 0
            self._task = task
            self._cmd = cmd
            self._started_at = time.time()
            self._append(f"$ {' '.join(cmd)}")
            self._reader = threading.Thread(
                target=self._pump, args=(self._proc,), daemon=True)
            self._reader.start()
            return {"ok": True, "cmd": cmd}

    def _append(self, line: str) -> None:
        # 调用方需持锁，或在 _pump 内持锁。
        self._lines.append(line.rstrip("\n"))
        self._seq += 1

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            with self._lock:
                self._append(raw)
        proc.wait()
        with self._lock:
            self._append(f"[进程结束] returncode={proc.returncode}")

    # ------------------------------------------------------------------ stop
    def stop(self) -> Dict[str, Any]:
        """终止运行中的进程（含其子进程/进程组）。"""
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return {"ok": True, "info": "无运行中的任务。"}
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True)
            else:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                # 宽限后强杀。
                for _ in range(20):
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"停止失败: {e}"}
        with self._lock:
            self._append("[已请求停止]")
        return {"ok": True}


# 进程内单例。
MANAGER = RunManager()
