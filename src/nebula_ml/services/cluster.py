from __future__ import annotations

from typing import Iterable, List

from nebula_ml.models import ClusterNode


class ClusterService:
    def __init__(self, nodes: Iterable[ClusterNode] | None = None) -> None:
        self._nodes: List[ClusterNode] = list(nodes or [])

    def register_node(self, node: ClusterNode) -> None:
        self._nodes.append(node)

    def list_nodes(self) -> list[ClusterNode]:
        return list(self._nodes)
