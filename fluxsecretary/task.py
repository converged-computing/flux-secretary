"""The launch task, as a behalf Task."""

from __future__ import annotations

import json
from typing import Any

from behalf import AgentRunner, ConfirmFn, Task, ToolSpec

from . import flux as fluxio
from .launch import Plan
from .report import Transcript, application_output, attempt_banner

SYSTEM = """You launch an application inside a Flux allocation that has ALREADY been
sized correctly by a scheduler. Your only job is to choose the launch parameters
that fit the resources actually present. You may edit workload manager flags to
optimize affinity, and add environment variables to fix issues with MPI.

Call get_resources first: it reports the nodes, cores and GPUs this allocation
really has. Do not assume. A task count computed outside the allocation is a
guess and is usually what went wrong.

Then call try_launch with nodes/tasks (and optionally cores_per_task). It submits and waits,
and returns the exit code, any Flux exceptions, the job events, and the
application's own stdout and stderr. Read the stderr first: a program that
aborts usually says why, and that message often shows the failure has nothing to
do with how the job was launched. Typical fixes: too many tasks
for the cores present (reduce), one rank where the program needs several (raise),
or omit tasks and let Flux size the job.

On your first attempts you should not change the command, the application, or its problem size. 
After that please do your best to get it working. A solution is NOT changing the initial intent 
(e.g., a run on one node is not a solution to a fabric issue).

Stop as soon as a launch succeeds."""


def parse_env(value) -> dict:
    """Environment from a mapping or a "K=V,K=V" string."""
    if not value:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    out = {}
    for pair in str(value).split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"cannot parse environment {pair!r}: expected KEY=VALUE, "
                f"comma separated"
            )
        k, _, v = pair.partition("=")
        out[k.strip()] = v.strip()
    return out


def tail(text: str, limit: int = 4000) -> str:
    """The last of a stream, which is where a failure says why."""
    text = text or ""
    return text if len(text) <= limit else "...(truncated)...\n" + text[-limit:]


def _text(obj: Any) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(obj, indent=2)}]}


class LaunchTask(Task):
    name = "launch"

    def __init__(self, command, want_nodes=None, max_attempts=4, timeout=None):
        self.command = list(command)
        self.want_nodes = want_nodes
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.transcript: Transcript | None = None
        self.outcome: dict | None = None

    def manifest_schema(self) -> dict:
        return {"goal": str}

    def setup_system_prompt(self) -> str:
        return SYSTEM

    def execute_system_prompt(self, manifest: dict) -> str:
        return SYSTEM

    def tools(self, res: dict, tr: Transcript) -> list[ToolSpec]:
        async def get_resources(a):
            return _text(res)

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
                if value and value != "per-task":
                    return _text({"error": f"{key} must be unset or 'per-task'"})
            plan = Plan(
                nodes=asked,
                tasks=int(a["tasks"]) if a.get("tasks") else None,
                cores_per_task=(
                    int(a["cores_per_task"]) if a.get("cores_per_task") else None
                ),
                gpus_per_task=(
                    int(a["gpus_per_task"]) if a.get("gpus_per_task") else None
                ),
                environment=environment,
                cpu_affinity=a.get("cpu_affinity") or None,
                gpu_affinity=a.get("gpu_affinity") or None,
                exclusive=bool(a.get("exclusive")),
                why=a.get("reasoning", "agent"),
            )
            attempt_banner(len(tr.attempts) + 1, plan.submit_command(self.command))
            out = fluxio.submit_and_wait(
                self.command,
                nodes=plan.nodes,
                tasks=plan.tasks,
                cores_per_task=plan.cores_per_task,
                gpus_per_task=plan.gpus_per_task,
                environment=plan.environment,
                cpu_affinity=plan.cpu_affinity,
                gpu_affinity=plan.gpu_affinity,
                exclusive=plan.exclusive,
                duration=self.timeout,
            )
            exc = (out.get("exceptions") or [{}])[0]
            attempt = tr.add(
                status="ok" if out["rc"] == 0 else "failed",
                rc=out["rc"],
                jobid=out.get("jobid"),
                exception=exc.get("type"),
                runtime_s=out.get("runtime"),
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
                "never ones that change what it computes.",
                {
                    "nodes": int,
                    "tasks": int,
                    "cores_per_task": int,
                    "gpus_per_task": int,
                    "environment": str,
                    "cpu_affinity": str,
                    "gpu_affinity": str,
                    "exclusive": bool,
                    "reasoning": str,
                },
                try_launch,
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
