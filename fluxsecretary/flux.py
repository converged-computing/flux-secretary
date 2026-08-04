"""Flux Python Handles"""

from __future__ import annotations

import os

import flux
import flux.job
from flux.job import JobspecV1
from flux.resource import resource_list


class FluxError(RuntimeError):
    pass


def handle():
    """A handle to the broker we are running under."""
    return flux.Flux()


def resources(h=None) -> dict:
    """What this allocation has from resource list"""
    h = h or handle()
    rl = resource_list(h).get()
    out = {}
    for label in ("all", "free", "up"):
        rset = getattr(rl, label, None)
        if rset is None:
            continue
        out[label] = {
            "nodes": rset.nnodes,
            "cores": rset.ncores,
            "gpus": getattr(rset, "ngpus", 0),
        }
    allr = out.get("all", {})
    out.update(
        {
            "nodes": allr.get("nodes", 0),
            "cores": allr.get("cores", 0),
            "gpus": allr.get("gpus", 0),
            "source": "flux.resource.resource_list",
        }
    )
    return out


def exit_code(status) -> int:
    """Turn a wait status from the finish event into an exit code."""
    try:
        return os.waitstatus_to_exitcode(int(status))
    except (AttributeError, ValueError):
        status = int(status)
        return status >> 8 if status >= 256 else status


def watch(h, jobid) -> dict:
    """Follow the job eventlog until it is clean."""
    events, exceptions = [], []
    times, status = {}, None

    for event in flux.job.event_watch(h, jobid):
        events.append(
            {
                "name": event.name,
                "timestamp": event.timestamp,
                "context": dict(event.context or {}),
            }
        )
        times[event.name] = event.timestamp

        if event.name == "exception":
            exceptions.append(dict(event.context or {}))
        elif event.name == "finish":
            status = (event.context or {}).get("status")
        elif event.name == "clean":
            break

    return {
        "events": events,
        "exceptions": exceptions,
        "times": times,
        "status": status,
    }


def job_output(h, jobid) -> dict:
    """The application's stdout and stderr, once the job is done."""
    try:
        out = flux.job.job_output(h, jobid)
    except Exception as e:
        return {"stdout": "", "stderr": "", "error": str(e)}
    return {
        "stdout": getattr(out, "stdout", "") or "",
        "stderr": getattr(out, "stderr", "") or "",
    }


def submit_and_wait(
    command,
    nodes=None,
    tasks=None,
    cores_per_task=None,
    gpus_per_task=None,
    duration=None,
    environment=None,
    cpu_affinity=None,
    gpu_affinity=None,
    exclusive=False,
    cwd=None,
    h=None,
) -> dict:
    """Submit via the SDK and wait for the job to finish."""
    h = h or handle()
    # flux rejects a jobspec with more nodes than tasks, and "let flux size the
    # job" arrives here as tasks=None: defaulting that to 1 asked for N nodes and
    # one task, and from_command raised before anything ran.
    if not tasks:
        tasks = nodes or 1
    spec = JobspecV1.from_command(
        command=list(command),
        num_tasks=tasks,
        num_nodes=nodes,
        cores_per_task=cores_per_task or 1,
        gpus_per_task=gpus_per_task,
        exclusive=bool(exclusive),
    )
    # The job's environment starts from ours, so the view and PATH survive, and
    # anything the caller adds layers on top.
    env = dict(os.environ)
    env.update(environment or {})
    spec.environment = env
    if duration:
        spec.duration = duration
    if cwd:
        spec.cwd = cwd
    if cpu_affinity or gpu_affinity:
        shell = spec.attributes["system"].get("shell", {})
        options = shell.get("options", {})
        if cpu_affinity:
            options["cpu-affinity"] = cpu_affinity
        if gpu_affinity:
            options["gpu-affinity"] = gpu_affinity
        shell["options"] = options
        spec.attributes["system"]["shell"] = shell

    jobid = flux.job.submit(h, spec, waitable=True)
    log = watch(h, jobid)

    fatal = [e for e in log["exceptions"] if e.get("severity") == 0]

    if log["status"] is not None:
        rc = exit_code(log["status"])
    else:
        rc = 1

    out = job_output(h, jobid)
    times = log["times"]
    info = {
        "jobid": str(jobid),
        "rc": rc,
        "success": rc == 0 and not fatal,
        "exceptions": log["exceptions"],
        "fatal": fatal,
        "events": [e["name"] for e in log["events"]],
        "eventlog": log["events"],
        "stdout": out["stdout"],
        "stderr": out["stderr"],
    }
    for name, key in (
        ("submit", "t_submit"),
        ("depend", "t_depend"),
        ("alloc", "t_alloc"),
        ("start", "t_run"),
        ("finish", "t_finish"),
        ("clean", "t_cleanup"),
    ):
        if name in times:
            info[key] = float(times[name])
    if "t_run" in info and "t_finish" in info:
        info["runtime"] = round(info["t_finish"] - info["t_run"], 3)
    return info
