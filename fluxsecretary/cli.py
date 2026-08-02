"""flux-secretary: run a command inside a Flux allocation, correctly.

    flux-secretary run -- osu_allreduce -m 8:1048576

Uses the agent by default and falls back to a deterministic submit.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from . import flux as fluxio
from .launch import ladder
from .report import Transcript, emit, note, section

DEFAULT_TOKEN_FILE = "/etc/flux-secretary/token"


def read_token(path):
    """Token from a mounted secret file, else the environment. The MiniCluster
    CRD's `environment` is plain key/value with no secretKeyRef, so a mounted
    secret is the only way to get a token in without putting it in the CRD."""
    for p in (path, DEFAULT_TOKEN_FILE):
        if p and os.path.isfile(p):
            tok = open(p).read().strip()
            if tok:
                return tok, f"file:{p}"
    for var in ("ANTHROPIC_API_KEY", "FLUX_SECRETARY_TOKEN"):
        if os.environ.get(var):
            return os.environ[var], f"env:{var}"
    return None, "none"


def run_deterministic(command, want_nodes, tr, timeout, max_attempts):
    """Ask Flux what it has, then walk the ladder until something runs."""
    h = fluxio.handle()
    res = fluxio.resources(h)
    tr.resources = res
    emit(
        "resources",
        nodes=res.get("nodes"),
        cores=res.get("cores"),
        gpus=res.get("gpus"),
        free_nodes=(res.get("free") or {}).get("nodes"),
        source=res.get("source"),
    )
    last = None
    for plan in ladder(res, want_nodes)[:max_attempts]:
        out = fluxio.submit_and_wait(
            command,
            nodes=plan.nodes,
            tasks=plan.tasks,
            cores_per_task=plan.cores_per_task,
            duration=timeout,
            h=h,
        )
        last = out
        exc = (out.get("exceptions") or [{}])[0]
        tr.add(
            status="ok" if out["rc"] == 0 else "failed",
            rc=out["rc"],
            jobid=out.get("jobid"),
            exception=exc.get("type"),
            runtime_s=out.get("runtime"),
            **plan.as_fields(),
        )
        if out["rc"] == 0:
            return 0, out.get("jobid"), ""
        # No attempt is made to interpret why it failed. A launch that does not
        # fit gets the next rung; a job that fails for reasons outside launching,
        # such as being placed on the wrong hardware, fails every rung and exits
        # non zero, which is the right answer.
        if exc.get("note"):
            note(str(exc["note"])[:500])
    return (
        (last or {}).get("rc", 1),
        (last or {}).get("jobid"),
        "no launch configuration worked",
    )


def run_agent(command, want_nodes, tr, timeout, max_attempts, backend, model):
    from behalf import make_runner  # imported lazily: optional dependency

    from .task import LaunchTask

    task = LaunchTask(
        command, want_nodes=want_nodes, max_attempts=max_attempts, timeout=timeout
    )
    task.transcript = tr
    runner = make_runner(backend=backend, model=model)
    outcome = asyncio.run(task.execute(runner, {"goal": "launch"}, lambda n, a: True))
    if outcome and outcome.get("rc") == 0:
        return 0, outcome.get("jobid"), ""
    if outcome and outcome.get("refused"):
        return outcome.get("rc", 1), None, outcome["refused"]
    return 1, None, (outcome or {}).get("reason", "agent did not achieve a launch")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="flux-secretary", description="Run a command inside a Flux allocation."
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="launch a command")
    r.add_argument(
        "--nodes",
        type=int,
        default=None,
        help="requested node count (default: all the allocation has)",
    )
    r.add_argument("--attempts", type=int, default=4)
    r.add_argument("--timeout", type=int, default=None, help="per-attempt seconds")
    r.add_argument("--token-file", default=DEFAULT_TOKEN_FILE)
    r.add_argument(
        "--backend", default=os.environ.get("FLUX_SECRETARY_BACKEND", "auto")
    )
    r.add_argument("--model", default=os.environ.get("FLUX_SECRETARY_MODEL"))
    r.add_argument(
        "--deterministic", action="store_true", help="skip the agent entirely"
    )
    r.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)

    command = [c for c in args.command if c != "--"]
    if not command:
        print("nothing to run: give a command after --", file=sys.stderr)
        return 2

    tr = Transcript(command)
    section("flux-secretary")
    emit("command", argv=" ".join(command), nodes=args.nodes)

    token, source = read_token(args.token_file)
    use_agent = bool(token) and not args.deterministic
    tr.mode = "agent" if use_agent else "deterministic"
    emit("mode", mode=tr.mode, token=source, attempts_max=args.attempts)

    rc, jobid, reason = 1, None, ""
    if use_agent:
        try:
            rc, jobid, reason = run_agent(
                command,
                args.nodes,
                tr,
                args.timeout,
                args.attempts,
                args.backend,
                args.model,
            )
        except Exception as e:  # noqa: BLE001 - never let the agent break the run
            note(f"agent unavailable ({e}); falling back to deterministic")
            tr.mode = "deterministic-fallback"
            emit("mode", mode=tr.mode, reason=str(e)[:200])
            rc, jobid, reason = run_deterministic(
                command, args.nodes, tr, args.timeout, args.attempts
            )
    else:
        rc, jobid, reason = run_deterministic(
            command, args.nodes, tr, args.timeout, args.attempts
        )

    tr.finish(
        "ok" if rc == 0 else "failed",
        rc,
        reason=reason,
        jobid=jobid,
        job=tr.attempts[-1] if tr.attempts else {},
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
