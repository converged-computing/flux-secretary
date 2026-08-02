"""Structured, line-oriented output."""

from __future__ import annotations

import json
import shlex
import sys
import time

PREFIX = "FLUXSEC"

OUTPUT_LIMIT = 64 * 1024


def _fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "-"
    s = str(value)
    return shlex.quote(s) if (" " in s or not s) else s


def emit(kind: str, **fields) -> None:
    """One record. Field order is preserved so lines diff cleanly."""
    parts = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
    print(f"{PREFIX} {kind} {parts}".rstrip(), flush=True)


def section(title: str) -> None:
    print(f"{PREFIX} === {title} ===", flush=True)


def note(text: str) -> None:
    """Free-form human line, still prefixed so it can be filtered out."""
    for line in str(text).splitlines():
        print(f"{PREFIX} # {line}", flush=True)


def final(payload: dict) -> None:
    """The machine-readable transcript. Always the LAST line."""
    print(f"{PREFIX} json " + json.dumps(payload, separators=(",", ":")), flush=True)


class Transcript:
    """Accumulates attempts so the final record is complete and ordered."""

    def __init__(self, command):
        self.started = time.time()
        self.command = list(command)
        self.attempts = []
        self.resources = {}
        self.mode = "unknown"

    def add(self, **attempt):
        attempt["n"] = len(self.attempts) + 1
        self.attempts.append(attempt)
        emit("attempt", **attempt)
        return attempt

    def finish(self, status, rc, reason="", jobid=None, job=None):
        for a in self.attempts:
            for k in ("stdout", "stderr"):
                text = a.get(k) or ""
                if len(text) > OUTPUT_LIMIT:
                    a[k] = text[-OUTPUT_LIMIT:]
                    a[k + "_truncated"] = True
        payload = {
            "status": status,
            "rc": rc,
            "reason": reason,
            "mode": self.mode,
            "attempts": self.attempts,
            "attempt_count": len(self.attempts),
            "resources": self.resources,
            "command": self.command,
            "jobid": jobid,
            "job": job or {},
            "wall_s": round(time.time() - self.started, 2),
        }
        emit(
            "result",
            status=status,
            rc=rc,
            attempts=len(self.attempts),
            jobid=jobid,
            mode=self.mode,
            wall_s=payload["wall_s"],
            reason=reason or None,
        )
        final(payload)
        sys.stdout.flush()
        return payload


def attempt_banner(n: int, command: str) -> None:
    """Announce an attempt with the flux command it is equivalent to."""
    print(f"{PREFIX} === attempt {n}: {command} ===", flush=True)


def application_output(stdout: str, stderr: str, n: int = 0) -> None:
    """Print the application's own output, unprefixed, between markers."""
    label = f"attempt {n} " if n else "application "
    for name, text in (("stdout", stdout), ("stderr", stderr)):
        if not (text or "").strip():
            continue
        print(f"{PREFIX} === {label}{name} ===", flush=True)
        print(text.rstrip("\n"), flush=True)
        print(f"{PREFIX} === end {label}{name} ===", flush=True)
