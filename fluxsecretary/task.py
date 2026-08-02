"""The launch task, as a behalf Task."""

from __future__ import annotations

import json
from typing import Any

from behalf import AgentRunner, ConfirmFn, Task, ToolSpec

from . import flux as fluxio
from .launch import Plan
from .report import Transcript

SYSTEM = """You launch an application inside a Flux allocation that has ALREADY been
sized correctly by a scheduler. Your only job is to choose the launch parameters
that fit the resources actually present.

Call get_resources first: it reports the nodes, cores and GPUs this allocation
really has. Do not assume. A task count computed outside the allocation is a
guess and is usually what went wrong.

Then call try_launch with nodes/tasks (and optionally cores_per_task). It submits and waits,
and returns the exit code together with any Flux exceptions and the sequence of
job events. Read those and decide what to change. Typical fixes: too many tasks
for the cores present (reduce), one rank where the program needs several (raise),
or omit tasks and let Flux size the job.

You may change ONLY how the job is launched. On your first attempts you should not
change the command, the application, or its problem size. If the job failed for a reason
that has nothing to do with how it was launched, for example it was placed on
hardware it cannot run on, do not try to work around it. Call give_up with the
reason so the scheduler sees the real outcome. If the job is failing due to a subtle
configuration issue (e.g., problem size done incorrectly) you MAY correct that after
your first few attempts.

Stop as soon as a launch succeeds."""


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
            plan = Plan(
                nodes=int(a.get("nodes") or res.get("nodes") or 1),
                tasks=int(a["tasks"]) if a.get("tasks") else None,
                cores_per_task=(
                    int(a["cores_per_task"]) if a.get("cores_per_task") else None
                ),
                why=a.get("reasoning", "agent"),
            )
            out = fluxio.submit_and_wait(
                self.command,
                nodes=plan.nodes,
                tasks=plan.tasks,
                cores_per_task=plan.cores_per_task,
                duration=self.timeout,
            )
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
                "Returns the exit code and stderr. You cannot change the command.",
                {"nodes": int, "tasks": int, "cores_per_task": int, "reasoning": str},
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
