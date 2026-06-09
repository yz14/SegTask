"""集中式日志配置：控制台按"模块类别 + 日志级别"双重上色，文件保持纯文本。

设计要点
--------
- 颜色仅作用于控制台 handler；文件 handler 始终输出纯文本，避免 ``train.log``
  里混入 ANSI 转义符导致编辑器乱码。
- 模块类别由 logger 名（``logging.getLogger(__name__)`` 产生，形如
  ``segtask_v1.trainer.trainer``）的子包前缀推断，无需各模块改动。
- 颜色启用条件遵循业界惯例：输出为 TTY 且未设置 ``NO_COLOR`` 环境变量；
  非 TTY（重定向/管道）自动退化为纯文本。Windows 下经 ``colorama`` 处理 ANSI。
- ``train.py`` / ``predict.py`` / ``make_data.py`` 统一调用本模块的
  ``setup_logging``，消除原先重复的日志配置代码。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from colorama import Back, Fore, Style
from colorama import init as _colorama_init

# colorama 在 Windows 上把 ANSI 转义翻译为 Win32 调用；其它平台无副作用。
# strip=False：不主动剥离 ANSI（我们自己根据 _supports_color 决定是否上色）。
_colorama_init(strip=False)

# ---------------------------------------------------------------------------
# 颜色映射
# ---------------------------------------------------------------------------
# 模块类别 -> 颜色。键为 logger 名去掉顶层包后的首段子包（见 _module_category）。
_MODULE_COLORS = {
    "data":      Fore.GREEN,
    "models":    Fore.MAGENTA,
    "trainer":   Fore.BLUE,
    "predictor": Fore.CYAN,
    "losses":    Fore.YELLOW,
    "config":    Fore.LIGHTBLACK_EX,
    "train":     Fore.LIGHTBLUE_EX,
    "predict":   Fore.LIGHTCYAN_EX,
    "utils":     Fore.LIGHTBLACK_EX,
}
_DEFAULT_MODULE_COLOR = Fore.WHITE

# 日志级别 -> 颜色。
_LEVEL_COLORS = {
    logging.DEBUG:    Fore.LIGHTBLACK_EX,
    logging.INFO:     Fore.GREEN,
    logging.WARNING:  Fore.YELLOW,
    logging.ERROR:    Fore.RED,
    logging.CRITICAL: Back.RED + Fore.WHITE + Style.BRIGHT,
}

_TOP_PACKAGE = "segtask_v1"


def _module_category(logger_name: str) -> str:
    """从 logger 名推断模块类别（首段子包名）。

    示例：``segtask_v1.trainer.pipelines.slab25d`` -> ``trainer``；
    ``segtask_v1.config`` -> ``config``；``segtask_v1`` -> ``segtask_v1``。
    """
    parts = logger_name.split(".")
    if parts and parts[0] == _TOP_PACKAGE:
        return parts[1] if len(parts) > 1 else _TOP_PACKAGE
    return parts[0] if parts else logger_name


def _supports_color(stream) -> bool:
    """判断给定流是否应启用彩色输出。

    规则（与 no-color.org / 业界惯例一致）：
    - 显式设置 ``NO_COLOR`` 环境变量 -> 关闭；
    - 显式设置 ``FORCE_COLOR`` 环境变量 -> 开启；
    - 否则仅当流是 TTY 时开启。
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------
class ColorFormatter(logging.Formatter):
    """按模块类别给 logger 名上色、按级别给 levelname 上色的 Formatter。

    ``use_color=False`` 时行为与普通 ``logging.Formatter`` 完全一致，
    便于在非 TTY / 文件 handler 复用同一格式而不输出 ANSI。
    """

    def __init__(self, fmt: str, datefmt: Optional[str] = None,
                 use_color: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_color:
            return super().format(record)

        # 在副本字段上着色，避免污染原 record（其它 handler 仍需纯文本）。
        orig_levelname = record.levelname
        orig_name = record.name
        level_color = _LEVEL_COLORS.get(record.levelno, "")
        module_color = _MODULE_COLORS.get(
            _module_category(record.name), _DEFAULT_MODULE_COLOR)
        try:
            record.levelname = f"{level_color}{orig_levelname}{Style.RESET_ALL}"
            record.name = f"{module_color}{orig_name}{Style.RESET_ALL}"
            s = super().format(record)
        finally:
            record.levelname = orig_levelname
            record.name = orig_name

        # 支持通过 extra={"msg_color": ...} 给日志消息内容上色
        msg_color = getattr(record, "msg_color", None)
        if msg_color:
            msg = getattr(record, "message", None)
            if msg:
                # 只替换最后一次出现，避免误伤头部相同文本
                parts = s.rsplit(msg, 1)
                if len(parts) == 2:
                    colored_msg = f"{msg_color}{msg}{Style.RESET_ALL}"
                    s = colored_msg.join(parts)
        return s


# ---------------------------------------------------------------------------
# Public setup
# ---------------------------------------------------------------------------
_FMT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    output_dir: Optional[str] = None,
    level: str = "INFO",
    log_filename: str = "train.log",
    color: Optional[bool] = None,
) -> None:
    """配置根 logger：控制台彩色 + 可选文件纯文本。

    参数
    ----
    output_dir:
        日志文件目录；为 ``None`` 时只配置控制台（供 make_data 等使用）。
    level:
        根 logger 级别名（如 ``"INFO"``、``"DEBUG"``）。
    log_filename:
        日志文件名，仅在 ``output_dir`` 非空时生效。
    color:
        是否启用控制台彩色；``None`` 表示自动探测（TTY / NO_COLOR / FORCE_COLOR）。
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    stream = sys.stdout
    use_color = _supports_color(stream) if color is None else color

    handlers = []

    console = logging.StreamHandler(stream)
    console.setFormatter(ColorFormatter(_FMT, _DATEFMT, use_color=use_color))
    handlers.append(console)

    if output_dir:
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        # 文件始终纯文本：复用同一格式但关闭颜色。
        fh.setFormatter(ColorFormatter(_FMT, _DATEFMT, use_color=False))
        handlers.append(fh)

    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )
