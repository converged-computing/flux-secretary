"""The launch task, as a behalf Task."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from typing import Any

from behalf import AgentRunner, ConfirmFn, Task, ToolSpec

from . import flux as fluxio
from .launch import Plan
from .report import Transcript, application_output, attempt_banner

SYSTEM = """You launch an application inside a Flux allocation that has been
sized correctly by a scheduler. Your job is to choose the launch parameters
that fit the resources actually present. You may edit workload manager flags to
optimize affinity, and add environment variables to fix issues with MPI.
You may edit the problem parameters, with args, when the application rejects them
outright: kripke refuses fewer than 8 directions before it decomposes anything, and
no layout fixes that. Change the least you can and stay close to what was asked;
this is for arguments the program will not accept, not for making a slow run
finish. Shrinking the problem so it fits one node is changing the intent, and is
not allowed.

Call get_resources first: it reports the nodes, cores and GPUs this allocation
really has. Do not assume. A task count computed outside the allocation is a
guess.

Then call try_launch with nodes/tasks. It submits and waits,
and returns the exit code, any Flux exceptions, the job events, and the
application's own stdout and stderr. Read the stderr first: a program that
aborts usually says why, and that message often shows the failure has nothing to
do with how the job was launched. Typical fixes: too many tasks
for the cores present (reduce), one rank where the program needs several (raise),
or omit tasks and let Flux size the job.

Before asking for gpus_per_task, call inspect_binary on the workload's executable.
An allocation reporting GPUs does not mean the application can use them: a CPU
build gains nothing from claiming a device. Do not set GPU affinity if there are no
GPUs. Set a reasonable timeout for the application run.

On your first attempts you should not change the command, the application, or its problem size.
After that you must respond to issues with parameters and problem size until you get it working.
However, a solution is NOT changing the initial intent (e.g., a run on one node is not a
solution to a fabric issue). You MUST try until you exhaust max attempts.

Keep your messages short. State the decision and the reason in a sentence or two.
A long message can hit the model's output limit and end the run.
Stop as soon as a launch succeeds."""


# Separator for caller supplied text, so the agent can tell the standing
# instructions from what this particular submitter asked for.
INTENT_HEADER = """

What this run is for, from the person who submitted it. It does not override the
rules above: you still may not change the intent of the job to make it succeed.
Where it conflicts with a choice you were about to make, follow it.

"""


def parse_env(value) -> dict:
    """Environment from a mapping, or from "K=V" pairs separated by commas or spaces.

    Values may themselves contain commas (OMPI_MCA_btl=self,vader,tcp), so a
    separator only counts where the next token starts a new KEY=.
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    text = str(value).strip()
    if not text:
        return {}
    out = {}
    for pair in re.split(r"[,\s]+(?=[A-Za-z_][A-Za-z0-9_]*=)", text):
        pair = pair.strip().strip(",")
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"cannot parse environment {pair!r}: expected KEY=VALUE pairs "
                f"separated by commas or spaces"
            )
        k, _, v = pair.partition("=")
        out[k.strip()] = v.strip()
    return out


def tail(text: str, limit: int = 1500) -> str:
    """The last of a stream, which is where a failure says why.

    Kept short on purpose: ten attempts of stdout and stderr accumulate in the
    conversation, and an agent that quotes them back hits the model's output limit
    and stops mid-message.
    """
    text = text or ""
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def _text(obj: Any) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(obj, indent=2)}]}


class LaunchTask(Task):
    name = "launch"

    def __init__(
        self, command, want_nodes=None, max_attempts=4, timeout=None, intent=None
    ):
        self.command = list(command)
        self.want_nodes = want_nodes
        self.max_attempts = max_attempts
        self.timeout = timeout
        # Caller supplied text appended to the prompt. What the workload is FOR is
        # something only the submitter knows, and it changes what counts as a
        # correct launch: a benchmark that must span nodes is not satisfied by a
        # run that fits on one, however green the exit code.
        self.intent = (intent or "").strip()
        self.transcript: Transcript | None = None
        self.outcome: dict | None = None

    def manifest_schema(self) -> dict:
        return {"goal": str}

    def system_prompt(self) -> str:
        """The standing instructions, plus whatever the submitter added."""
        if not self.intent:
            return SYSTEM
        return SYSTEM + INTENT_HEADER + self.intent + "\n"

    def setup_system_prompt(self) -> str:
        return self.system_prompt()

    def execute_system_prompt(self, manifest: dict) -> str:
        return self.system_prompt()

    def tools(self, res: dict, tr: Transcript) -> list[ToolSpec]:
        async def get_resources(a):
            return _text(res)

        async def inspect_binary(a):
            """How the workload was built, read from the ELF.

            The agent otherwise has only the allocation to go on, and seeing GPUs
            there it reasonably asks for one. What decides it is the binary.
            """
            name = (a.get("path") or a.get("name") or "").strip()
            if not name:
                return _text({"error": "pass the workload's executable"})
            path = shutil.which(name) if "/" not in name else name
            if not path or not os.path.exists(path):
                return _text(
                    {
                        "name": name,
                        "found": False,
                        "note": "not on PATH and not an existing path. The command may "
                        "be a wrapper, or the view may have shadowed it.",
                    }
                )

            def run(*cmd):
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    return r.stdout if r.returncode == 0 else ""
                except Exception:
                    return ""

            out = {"path": path, "found": True}
            for line in run("readelf", "-h", path).splitlines():
                if "Machine:" in line:
                    out["machine"] = line.split(":", 1)[1].strip()

            dyn = run("readelf", "-d", path)
            out["needed"] = sorted(set(re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", dyn)))
            out["rpath"] = sorted(
                set(re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^\]]+)\]", dyn))
            )

            # embedded device code is the only evidence for a statically linked
            # cuda or hip build: those have no libcuda in NEEDED at all
            secs = run("readelf", "-S", path)
            device = [
                n
                for n in (
                    ".nv_fatbin",
                    ".nvFatBinSegment",
                    "__nv_relfatbin",
                    ".hip_fatbin",
                    ".hipFatBinSegment",
                )
                if n in secs
            ]
            out["device_code_sections"] = device

            gpu_libs = [
                l
                for l in out["needed"]
                if l.startswith(
                    ("libcuda", "libcudart", "libamdhip", "libhsa-runtime", "librocm")
                )
            ]
            out["gpu_libs"] = gpu_libs
            if gpu_libs:
                out["accelerator"] = (
                    "cuda" if any("cuda" in l for l in gpu_libs) else "rocm"
                )
                out["evidence"] = "gpu runtime in NEEDED"
            elif device:
                out["accelerator"] = (
                    "cuda" if any("nv" in d for d in device) else "rocm"
                )
                out["evidence"] = "embedded device code, statically linked runtime"
            else:
                out["accelerator"] = "none"
                out["evidence"] = "no gpu runtime linked, no embedded device code"

            mpi = [l for l in out["needed"] if l.startswith(("libmpi", "libmpich"))]
            if mpi:
                out["mpi_libs"] = mpi

            # what cannot be resolved HERE, which is not how it was built
            missing = [
                l.strip() for l in run("ldd", path).splitlines() if "not found" in l
            ]
            if missing:
                out["unresolved_here"] = missing[:10]

            out["note"] = (
                "accelerator=none is a CPU build: gpus_per_task gains nothing. "
                "unresolved_here means the image is fine and this container is "
                "missing the library, which is not a launch problem."
            )
            return _text(out)

        async def find_file(a):
            """Where a named file lives. The command is fixed, so a run that
            cannot open its input is in the wrong directory, not the wrong job."""
            name = (a.get("name") or "").strip()
            if not name or "/" in name:
                return _text({"error": "pass a bare filename, e.g. in.reaxff.hns"})
            hits, roots = [], [
                "/opt",
                "/usr/local",
                "/home",
                "/data",
                "/work",
                "/scratch",
            ]
            for root in roots:
                if not os.path.isdir(root):
                    continue
                for dirpath, dirnames, filenames in os.walk(root, topdown=True):
                    dirnames[:] = [
                        d
                        for d in dirnames
                        if not d.startswith(".") and d not in ("proc", "sys", "dev")
                    ]
                    if dirpath.count(os.sep) > 8:
                        dirnames[:] = []
                        continue
                    if name in filenames:
                        hits.append(dirpath)
                        if len(hits) >= 20:
                            break
                if len(hits) >= 20:
                    break
            if not hits:
                return _text({"name": name, "directories": [], "note": "not found"})
            return _text(
                {
                    "name": name,
                    "directories": sorted(hits),
                    "note": (
                        "one directory: safe to pass as cwd. Several: they are "
                        "different inputs and picking one changes what is measured, "
                        "so do not guess, call give_up and say which were found."
                    ),
                }
            )

        async def try_launch(a):
            if len(tr.attempts) >= self.max_attempts:
                return _text(
                    {
                        "error": f"attempt limit ({self.max_attempts}) reached; "
                        f"call give_up"
                    }
                )
            have = res.get("nodes") or 1
            asked = int(a.get("nodes") or have)
            if asked < min(have, self.want_nodes or have):
                return _text(
                    {
                        "error": f"refusing {asked} nodes: the allocation holds "
                        f"{have}. Fewer nodes is a different job, not a different "
                        f"launch. Vary tasks or cores_per_task, or call give_up."
                    }
                )
            try:
                environment = parse_env(a.get("environment"))
            except ValueError as e:
                return _text({"error": str(e)})
            for key, value in (
                ("cpu_affinity", a.get("cpu_affinity")),
                ("gpu_affinity", a.get("gpu_affinity")),
            ):
                if value and value not in ("per-task", "off"):
                    return _text({"error": f"{key} must be unset, 'per-task' or 'off'"})
            # The executable is fixed; only its arguments may be replaced, and the
            # substitution is recorded so a run that changed the problem cannot be
            # mistaken for one that did not.
            argv = list(self.command)
            if (a.get("args") or "").strip():
                argv = [argv[0], *shlex.split(a["args"])]

            gpus_per_task = int(a["gpus_per_task"]) if a.get("gpus_per_task") else None
            gpu_affinity = a.get("gpu_affinity") or None
            if not gpus_per_task:
                # Nothing to bind. Rejecting 'off' forced the agent into
                # 'per-task', which asks the shell to place tasks onto devices the
                # job was never given.
                gpu_affinity = None
            plan = Plan(
                nodes=asked,
                tasks=int(a["tasks"]) if a.get("tasks") else None,
                cores_per_task=(
                    int(a["cores_per_task"]) if a.get("cores_per_task") else None
                ),
                gpus_per_task=gpus_per_task,
                environment=environment,
                cpu_affinity=a.get("cpu_affinity") or None,
                gpu_affinity=gpu_affinity,
                exclusive=bool(a.get("exclusive")),
                cwd=(a.get("cwd") or None),
                why=a.get("reasoning", "agent"),
            )
            attempt_banner(len(tr.attempts) + 1, plan.submit_command(argv))
            out = fluxio.submit_and_wait(
                argv,
                nodes=plan.nodes,
                tasks=plan.tasks,
                cores_per_task=plan.cores_per_task,
                gpus_per_task=plan.gpus_per_task,
                environment=plan.environment,
                cpu_affinity=plan.cpu_affinity,
                gpu_affinity=plan.gpu_affinity,
                exclusive=plan.exclusive,
                cwd=plan.cwd,
                duration=self.timeout or None,
            )
            exc = (out.get("exceptions") or [{}])[0]
            attempt = tr.add(
                status="ok" if out["rc"] == 0 else "failed",
                rc=out["rc"],
                jobid=out.get("jobid"),
                exception=exc.get("type"),
                runtime_s=out.get("runtime"),
                # recorded so a run that changed the problem cannot be mistaken
                # for one that did not
                args_changed=(
                    " ".join(argv[1:]) if argv[1:] != list(self.command)[1:] else None
                ),
                **plan.as_fields(),
            )
            attempt["stdout"] = out.get("stdout", "")
            attempt["stderr"] = out.get("stderr", "")
            application_output(attempt["stdout"], attempt["stderr"], attempt["n"])
            if out["rc"] == 0:
                self.outcome = {
                    "plan": plan.as_fields(),
                    "jobid": out.get("jobid"),
                    "rc": 0,
                }
                return _text(
                    {
                        "status": "ok",
                        "jobid": out.get("jobid"),
                        "message": "launch succeeded; stop now",
                    }
                )
            return _text(
                {
                    "status": "failed",
                    "rc": out["rc"],
                    "exceptions": out.get("exceptions"),
                    "events": out.get("events"),
                    "stderr": tail(out.get("stderr", "")),
                    "stdout": tail(out.get("stdout", "")),
                }
            )

        async def give_up(a):
            self.outcome = {"gave_up": True, "reason": a.get("reason", "")}
            return _text("recorded")

        return [
            ToolSpec(
                "get_resources",
                "Nodes, cores and GPUs this allocation actually has.",
                {},
                get_resources,
            ),
            ToolSpec(
                "try_launch",
                "Submit the (fixed) command with these launch parameters and wait. "
                "Returns the exit code, stderr and stdout. You cannot change the "
                "command, but everything about HOW it runs is yours: layout, "
                "affinity, exclusivity, and environment. environment is the remedy "
                "for transport and runtime problems, e.g. "
                "'OMPI_MCA_pml=ob1,OMPI_MCA_btl=self,vader,tcp' or "
                "'FI_PROVIDER=tcp'. Set variables that change how the job runs, "
                "never ones that change what it computes. cwd is the remedy "
                "when the job cannot open its input file: find_file locates it "
                "and cwd runs there.",
                {
                    "nodes": int,
                    "tasks": int,
                    "cores_per_task": int,
                    "gpus_per_task": int,
                    "environment": str,
                    "cpu_affinity": str,
                    "gpu_affinity": str,
                    "exclusive": bool,
                    "args": str,
                    "cwd": str,
                    "reasoning": str,
                },
                try_launch,
            ),
            ToolSpec(
                "inspect_binary",
                "How the workload was built, from the ELF: architecture, linked "
                "gpu and mpi runtimes, embedded device code, and anything that "
                "cannot be resolved in this container. Call this BEFORE asking for "
                "gpus_per_task: an allocation with GPUs does not mean the "
                "application can use them.",
                {"path": str},
                inspect_binary,
            ),
            ToolSpec(
                "find_file",
                "Find the directory holding a named file, for when the job "
                "cannot open its input. Returns every directory containing it: "
                "one is unambiguous, several are different inputs and must not "
                "be guessed between.",
                {"name": str},
                find_file,
            ),
            ToolSpec(
                "give_up",
                "Stop: the failure is not a launch problem (or nothing works).",
                {"reason": str},
                give_up,
            ),
        ]

    async def execute(self, runner: AgentRunner, manifest: dict, confirm_fn: ConfirmFn):
        res = fluxio.resources()
        tr = self.transcript
        tr.resources = res
        await runner.run_agent(
            self.execute_system_prompt(manifest),
            f"Launch this command on {self.want_nodes or res.get('nodes')} node(s): "
            f"{' '.join(self.command)}",
            self.tools(res, tr),
            confirm_fn,
        )
        return self.outcome
