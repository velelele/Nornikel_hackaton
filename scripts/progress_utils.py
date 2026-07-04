from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass


@dataclass
class TerminalProgress:
    """Small dependency-free progress bar for long CLI ingestion jobs.

    It works both in an interactive terminal and when stdout is piped through
    nightly_two_stage.py into log files. In pipe/log mode it prints at most one
    line per percent step, plus forced state changes.
    """

    label: str
    total: int
    enabled: bool = True
    width: int = 34
    min_interval_sec: float = 0.4

    def __post_init__(self) -> None:
        self.total = max(0, int(self.total or 0))
        self.current = 0
        self.started_at = time.time()
        self._last_render_at = 0.0
        self._last_percent: int | None = None
        self._last_line_len = 0
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        if self.width <= 0:
            term_width = shutil.get_terminal_size((100, 20)).columns
            self.width = max(18, min(40, term_width // 3))

    def _percent(self, current: int | None = None) -> int | None:
        if self.total <= 0:
            return None
        value = self.current if current is None else max(0, min(int(current), self.total))
        return int(round((value / self.total) * 100))

    def _bar(self, percent: int | None) -> str:
        if percent is None:
            return "[" + ("." * self.width) + "]"
        filled = int(round(self.width * percent / 100))
        return "[" + ("#" * filled) + ("." * (self.width - filled)) + "]"

    def _line(self, status: str = "") -> str:
        percent = self._percent()
        elapsed = max(0.0, time.time() - self.started_at)
        if percent is None:
            base = f"{self.label} {self._bar(percent)} {self.current}/? elapsed={elapsed:,.0f}s"
        else:
            base = f"{self.label} {self._bar(percent)} {percent:3d}% {self.current}/{self.total} elapsed={elapsed:,.0f}s"
        if status:
            status = " ".join(str(status).split())
            if len(status) > 100:
                status = status[:97] + "..."
            base += f" | {status}"
        return base

    def update(self, current: int | None = None, status: str = "", *, force: bool = False) -> None:
        if not self.enabled:
            return
        if current is not None:
            self.current = max(0, min(int(current), self.total if self.total > 0 else int(current)))
        now = time.time()
        percent = self._percent()
        should_render = force or self.current == self.total
        if percent != self._last_percent:
            should_render = True
        if now - self._last_render_at >= self.min_interval_sec and status:
            should_render = True
        if not should_render:
            return
        line = self._line(status)
        if self._tty:
            pad = " " * max(0, self._last_line_len - len(line))
            print("\r" + line + pad, end="", flush=True)
            self._last_line_len = len(line)
        else:
            print(line, flush=True)
        self._last_render_at = now
        self._last_percent = percent

    def finish(self, status: str = "done") -> None:
        if self.total > 0:
            self.current = self.total
        self.update(self.current, status, force=True)
        if self.enabled and self._tty:
            print(flush=True)

    def log(self, message: str) -> None:
        if self.enabled and self._tty and self._last_line_len:
            print("\r" + (" " * self._last_line_len) + "\r", end="", flush=True)
        print(message, flush=True)
        if self.enabled and self._tty and self.current < self.total:
            self.update(self.current, force=True)
