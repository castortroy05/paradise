# NebulaML

NebulaML is a starter repository for a distributed computing ML platform. It provides a clean Python codebase for orchestrating training jobs across a cluster, tracking datasets and artifacts, and exposing a simple control-plane API.

## What this repo includes

- **Control plane domain models** for jobs, clusters, and datasets
- **Scheduling primitives** for assigning distributed jobs to worker nodes
- **Training service abstractions** for launching and monitoring workloads
- **Fast local iteration** with a lightweight CLI entrypoint
- **Test suite** for scheduler and orchestration logic
- **Project docs** describing the architecture and roadmap

## Platform goals

NebulaML is designed as a foundation for a platform that can grow into:

- Distributed training orchestration for PyTorch, Ray, or Spark workloads
- Multi-node experiment tracking and artifact management
- Dataset registration and lineage tracking
- GPU-aware scheduling and autoscaling
- Team workspaces, quotas, and RBAC
- Model packaging and deployment pipelines

## Repository layout

```text
src/nebula_ml/
├── api.py               # API response helpers and control-plane surface
├── cli.py               # Command line entrypoint
├── models.py            # Core dataclasses for jobs, datasets, clusters, nodes
├── scheduler.py         # Resource-aware job placement logic
└── services/
    ├── cluster.py       # Cluster management service
    └── training.py      # Job submission and lifecycle management

docs/
└── architecture.md      # System architecture and future roadmap

tests/
├── test_scheduler.py    # Scheduler behavior tests
└── test_training.py     # Training service tests
```

## Quickstart

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install the package

```bash
pip install -e .
```

### 3. Run the CLI demo

```bash
python -m nebula_ml.cli
```

### 4. Run the tests

```bash
python -m pytest
```

## Example use

```python
from nebula_ml.models import ClusterNode, JobSpec
from nebula_ml.services.cluster import ClusterService
from nebula_ml.services.training import TrainingService

cluster = ClusterService([
    ClusterNode(name="worker-a", cpu=32, memory_gb=128, gpus=4),
    ClusterNode(name="worker-b", cpu=16, memory_gb=64, gpus=1),
])
training = TrainingService(cluster)

job = JobSpec(
    name="resnet50-training",
    image="ghcr.io/acme/resnet:latest",
    command=["python", "train.py"],
    cpu=8,
    memory_gb=32,
    gpus=1,
)

run = training.submit(job)
print(run.status)
```

## Development roadmap

1. Add a real HTTP API using FastAPI.
2. Introduce a persistent metadata store.
3. Add pluggable execution backends for Kubernetes and Ray.
4. Implement queue priorities, preemption, and quotas.
5. Add experiment tracking and artifact storage integrations.

## License

MIT
