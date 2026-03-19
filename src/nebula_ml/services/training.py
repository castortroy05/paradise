from __future__ import annotations

from nebula_ml.models import JobRun, JobSpec, JobStatus
from nebula_ml.scheduler import ResourceScheduler
from nebula_ml.services.cluster import ClusterService


class TrainingService:
    def __init__(self, cluster_service: ClusterService) -> None:
        self.cluster_service = cluster_service
        self.scheduler = ResourceScheduler()

    def submit(self, spec: JobSpec) -> JobRun:
        placement = self.scheduler.place(spec, self.cluster_service.list_nodes())
        return JobRun(spec=spec, status=JobStatus.SCHEDULED, assigned_nodes=placement.nodes)
