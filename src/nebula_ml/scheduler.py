from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from nebula_ml.models import ClusterNode, JobSpec


@dataclass(slots=True)
class Placement:
    nodes: List[str]
    reason: str


class SchedulingError(RuntimeError):
    """Raised when no valid placement can be found."""


class ResourceScheduler:
    """A simple best-fit scheduler for distributed ML jobs."""

    def place(self, job: JobSpec, nodes: Iterable[ClusterNode]) -> Placement:
        if job.distributed_replicas < 1:
            raise SchedulingError("distributed_replicas must be at least 1")

        eligible = [
            node
            for node in sorted(nodes, key=lambda item: (item.gpus, item.cpu, item.memory_gb), reverse=True)
            if node.cpu >= job.cpu and node.memory_gb >= job.memory_gb and node.gpus >= job.gpus
        ]

        if len(eligible) < job.distributed_replicas:
            raise SchedulingError(
                f"insufficient capacity for {job.name}: need {job.distributed_replicas} replicas"
            )

        selected = eligible[: job.distributed_replicas]
        return Placement(
            nodes=[node.name for node in selected],
            reason="selected highest-capacity eligible nodes",
        )
