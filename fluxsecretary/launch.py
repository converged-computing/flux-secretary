"""Choosing HOW to launch — never WHAT to run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Plan:
    """One candidate launch. `why` is recorded in the transcript."""

    nodes: int
    tasks: int | None = None
    cores_per_task: int | None = None
    gpus_per_task: int | None = None
    environment: dict = field(default_factory=dict)
    cpu_affinity: str | None = None
    gpu_affinity: str | None = None
    exclusive: bool = False
    cwd: str | None = None
    extra: list = field(default_factory=list)
    why: str = ""

    def submit_command(self, command) -> str:
        """The flux submit this plan is equivalent to."""
        parts = ["flux", "submit"]
        if self.nodes:
            parts += ["-N", str(self.nodes)]
        if self.tasks:
            parts += ["-n", str(self.tasks)]
        if self.cores_per_task:
            parts += ["-c", str(self.cores_per_task)]
        if self.gpus_per_task:
            parts += ["-g", str(self.gpus_per_task)]
        if self.exclusive:
            parts += ["--exclusive"]
        if self.cwd:
            parts += ["--cwd", self.cwd]
        if self.cpu_affinity:
            parts += ["-o", f"cpu-affinity={self.cpu_affinity}"]
        if self.gpu_affinity:
            parts += ["-o", f"gpu-affinity={self.gpu_affinity}"]
        env = " ".join(f"{k}={v}" for k, v in sorted(self.environment.items()))
        line = " ".join(parts + list(command))
        return f"{env} {line}" if env else line

    def as_fields(self):
        return {
            "nodes": self.nodes,
            "tasks": self.tasks,
            "cores_per_task": self.cores_per_task,
            "gpus_per_task": self.gpus_per_task,
            "environment": self.environment or None,
            "cpu_affinity": self.cpu_affinity,
            "gpu_affinity": self.gpu_affinity,
            "exclusive": self.exclusive or None,
            "cwd": self.cwd,
            "why": self.why,
        }


def ladder(res: dict, want_nodes: int | None = None) -> list[Plan]:
    """Launch layouts to try, from fully packed to letting flux decide.

    The node count never drops below what the allocation holds: fewer nodes is a
    different job, not a different launch.
    """
    nodes = want_nodes or res.get("nodes") or 1
    nodes = min(nodes, res.get("nodes") or nodes)
    cores = res.get("cores") or 0
    per_node = max(1, cores // nodes) if nodes else 1

    plans = []
    if cores:
        plans.append(Plan(nodes, nodes * per_node, why="one rank per core"))
    if per_node > 1:
        plans.append(Plan(nodes, nodes, why="one rank per node"))
    plans.append(Plan(nodes, None, why="let flux size the job"))

    seen, out = set(), []
    for p in plans:
        key = (p.nodes, p.tasks, p.cores_per_task)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
