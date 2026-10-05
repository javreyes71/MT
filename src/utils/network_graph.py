"""Grafo de adyacencia de semáforos con BFS a través de nodos sin TLS.

Corrige H7: el grafo anterior solo consideraba aristas directas TLS→TLS,
perdiendo vecinos separados por nodos sin semáforo. Además, `getNode(tls_id)`
fallaba con IDs `cluster_*`/`joinedS_*`.

Ahora usa BFS sobre la red vial, atravesando nodos intermedios sin TLS,
para encontrar los vecinos reales a 1+ saltos.
"""

import sumolib
from typing import List, Dict, Set, Tuple
from collections import deque


class NetworkGraph:
    """Grafo de adyacencia de semáforos basado en BFS vial."""

    def __init__(self, net_file: str) -> None:
        self.net = sumolib.net.readNet(net_file, withPrograms=True)

        # Mapeo: node_id → tls_id (si el nodo tiene semáforo)
        self._node_to_tls: Dict[str, str] = {}
        # Mapeo: tls_id → node_id
        self._tls_to_node: Dict[str, str] = {}

        for tls in self.net.getTrafficLights():
            tls_id = tls.getID()
            # Los TLS cluster_*/joinedS_* no corresponden 1:1 a un nodo.
            # Buscamos los nodos que contienen este TLS en sus conexiones.
            conns = tls.getConnections()
            if conns:
                # Usar el nodo del primer carril entrante
                first_in_lane = conns[0][0]
                edge = first_in_lane.getEdge()
                node = edge.getToNode()
                node_id = node.getID()
                self._node_to_tls[node_id] = tls_id
                self._tls_to_node[tls_id] = node_id

        self.tls_ids: List[str] = list(self._tls_to_node.keys())
        self.adjacency_dict: Dict[str, List[str]] = self._build_adjacency()

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """BFS desde cada nodo con TLS, atravesando nodos sin TLS."""
        adj: Dict[str, List[str]] = {tls: [] for tls in self.tls_ids}
        tls_node_ids = set(self._tls_to_node.values())

        for tls_id in self.tls_ids:
            start_node_id = self._tls_to_node.get(tls_id)
            if start_node_id is None:
                continue

            # BFS: buscar nodos TLS alcanzables con exactamente 1 "salto TLS"
            visited_nodes: Set[str] = {start_node_id}
            queue: deque = deque()

            # Expandir desde el nodo inicial
            try:
                start_node = self.net.getNode(start_node_id)
            except KeyError:
                continue

            for edge in list(start_node.getOutgoing()) + list(start_node.getIncoming()):
                other = edge.getToNode() if edge.getFromNode().getID() == start_node_id else edge.getFromNode()
                other_id = other.getID()
                if other_id not in visited_nodes:
                    queue.append(other_id)
                    visited_nodes.add(other_id)

            while queue:
                current_id = queue.popleft()

                if current_id in tls_node_ids:
                    # Encontramos un nodo con TLS → es vecino
                    neighbor_tls = self._node_to_tls[current_id]
                    if neighbor_tls != tls_id and neighbor_tls not in adj[tls_id]:
                        adj[tls_id].append(neighbor_tls)
                    continue  # No seguir más allá de un TLS

                # Nodo sin TLS → seguir explorando
                try:
                    node = self.net.getNode(current_id)
                except KeyError:
                    continue

                for edge in list(node.getOutgoing()) + list(node.getIncoming()):
                    other = edge.getToNode() if edge.getFromNode().getID() == current_id else edge.getFromNode()
                    other_id = other.getID()
                    if other_id not in visited_nodes:
                        queue.append(other_id)
                        visited_nodes.add(other_id)

        return adj

    def get_neighbors(self, tls_id: str, radius: int = 1) -> List[str]:
        """Retorna vecinos TLS hasta `radius` saltos de distancia.

        Un "salto" = un TLS vecino (puede haber muchos nodos intermedios
        sin semáforo en la red vial).
        """
        if tls_id not in self.adjacency_dict:
            return []

        visited: Set[str] = {tls_id}
        current_level: Set[str] = set(self.adjacency_dict.get(tls_id, []))
        all_neighbors: Set[str] = set(current_level)

        for _ in range(1, radius):
            next_level: Set[str] = set()
            for node in current_level:
                visited.add(node)
                neighbors = set(self.adjacency_dict.get(node, []))
                new = neighbors - visited
                next_level.update(new)
                all_neighbors.update(new)
            current_level = next_level

        return sorted(all_neighbors - {tls_id})  # Determinista
