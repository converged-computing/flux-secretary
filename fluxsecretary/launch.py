"""Choosing HOW to launch — never WHAT to run.

The command is fixed by the caller and is never rewritten here. Only launch
parameters (nodes, tasks, cores-per-task, flags) are chosen, which is the whole
scope of this process: the allocation is already correct, so the only open
question is how to fill it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Plan:
    """One candidate launch. `why` is recorded in the transcript."""

    nodes: int
    tasks: int | None = None
    cores_per_task: int | None = None
    extra: list = field(default_factory=list)
    why: str = ""

    def as_fields(self):
        return {
            "nodes": self.nodes,
            "tasks": self.tasks,
            "cores_per_task": self.cores_per_task,
            "why": self.why,
        }


def ladder(res: dict, want_nodes: int | None = None) -> list[Plan]:
    """Deterministic fallback: progressively weaker claims on the allocation.

    Used when no token is available, and as the backstop when the agent gives up.
    Ordered most-to-least specific, because the first plan that works is the one
    that uses the allocation best.
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
    if nodes > 1:
        plans.append(Plan(1, None, why="single node, last resort"))

    # de-duplicate while preserving order
    seen, out = set(), []
    for p in plans:
        key = (p.nodes, p.tasks, p.cores_per_task)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
