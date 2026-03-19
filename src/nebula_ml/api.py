from __future__ import annotations

from nebula_ml.models import JobRun


def serialize_job(run: JobRun) -> dict[str, object]:
    return {
        "id": run.id,
        "name": run.spec.name,
        "status": run.status.value,
        "assigned_nodes": run.assigned_nodes,
        "image": run.spec.image,
        "command": run.spec.command,
    }
