# NebulaML Architecture

## Vision

NebulaML separates the platform into a **control plane**, **execution plane**, and **data plane**.

## Core components

### 1. Control plane

Responsible for:
- authenticating users and services
- accepting job submissions
- storing metadata for jobs, datasets, models, and artifacts
- exposing APIs for cluster status and experiment history

### 2. Execution plane

Responsible for:
- scheduling distributed jobs to available nodes
- launching training workers and parameter servers
- monitoring heartbeats, logs, and retries
- integrating with backends such as Kubernetes, Ray, Slurm, or Spark

### 3. Data plane

Responsible for:
- dataset registration and versioning
- feature and training data access policies
- artifact and checkpoint storage
- lineage between datasets, code revisions, models, and deployments

## Initial implementation in this repo

This starter repo currently includes:
- in-memory cluster registration
- a best-fit job scheduler
- a training orchestration service
- simple API serialization helpers
- tests for placement and job submission behavior

## Suggested next steps

1. Add FastAPI routers for jobs, nodes, datasets, and artifacts.
2. Persist metadata in PostgreSQL.
3. Add background workers for log collection and retries.
4. Add Kubernetes and Ray execution backends.
5. Support experiment tracking integrations such as MLflow.
