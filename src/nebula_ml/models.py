from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List
from uuid import uuid4


class JobStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(slots=True)
class ClusterNode:
    name: str
    cpu: int
    memory_gb: int
    gpus: int = 0
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class JobSpec:
    name: str
    image: str
    command: List[str]
    cpu: int
    memory_gb: int
    gpus: int = 0
    distributed_replicas: int = 1
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class JobRun:
    spec: JobSpec
    status: JobStatus = JobStatus.PENDING
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_nodes: List[str] = field(default_factory=list)


@dataclass(slots=True)
class Dataset:
    name: str
    uri: str
    version: str
    format: str
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
