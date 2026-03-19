from __future__ import annotations

from nebula_ml.api import serialize_job
from nebula_ml.models import ClusterNode, JobSpec
from nebula_ml.services.cluster import ClusterService
from nebula_ml.services.training import TrainingService


def main() -> None:
    cluster = ClusterService(
        [
            ClusterNode(name="gpu-a", cpu=32, memory_gb=128, gpus=4),
            ClusterNode(name="gpu-b", cpu=16, memory_gb=64, gpus=1),
            ClusterNode(name="cpu-a", cpu=32, memory_gb=128, gpus=0),
        ]
    )
    training = TrainingService(cluster)
    job = JobSpec(
        name="demo-distributed-training",
        image="ghcr.io/example/train:latest",
        command=["python", "train.py", "--epochs", "5"],
        cpu=8,
        memory_gb=32,
        gpus=1,
        distributed_replicas=2,
    )
    run = training.submit(job)
    print(serialize_job(run))


if __name__ == "__main__":
    main()
