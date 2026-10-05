"""Dynamic Multi-Strategy Grouping Learning (DMSGL) basado en HAPS-PPO (Lu et al., 2025).

Agrupa agentes semafóricos por similitud en su espacio de acción y asigna
cabezales de política especializados para cada grupo.
"""
from typing import Dict, List, Tuple
from collections import defaultdict


class IntersectionGrouper:
    """Agrupa intersecciones por número de fases verdes disponibles.
    
    Intersecciones con el mismo número de acciones posibles comparten
    un cabezal de actor, mientras que el critic permanece compartido.
    
    Esto permite que:
    - Intersecciones de 2 fases (simples) tengan políticas especializadas
    - Intersecciones de 4 fases (complejas) tengan políticas diferentes
    - El conocimiento del valor de estado (critic) siga siendo global
    """
    
    def __init__(self, tls_ids: List[str], action_dims: List[int]) -> None:
        self.tls_ids = tls_ids
        self.action_dims = action_dims
        self.groups: Dict[int, List[str]] = defaultdict(list)
        self.tls_to_group: Dict[str, int] = {}
        
        self._build_groups()
    
    def _build_groups(self) -> None:
        """Construye los grupos basándose en el número de acciones."""
        for tls_id, n_actions in zip(self.tls_ids, self.action_dims):
            self.groups[n_actions].append(tls_id)
            self.tls_to_group[tls_id] = n_actions
        
        print(f"📊 DMSGL: {len(self.groups)} grupo(s) de intersecciones detectados:")
        for n_actions, members in sorted(self.groups.items()):
            print(f"   Grupo {n_actions}-fases: {len(members)} intersecciones")
    
    def get_group(self, tls_id: str) -> int:
        """Retorna el ID de grupo (número de acciones) para un semáforo."""
        return self.tls_to_group.get(tls_id, -1)
    
    def get_group_members(self, group_id: int) -> List[str]:
        """Retorna la lista de semáforos en un grupo."""
        return self.groups.get(group_id, [])
    
    def get_all_groups(self) -> Dict[int, List[str]]:
        """Retorna todos los grupos."""
        return dict(self.groups)
    
    def get_group_stats(self) -> Dict[str, any]:
        """Retorna estadísticas de los grupos para logging."""
        return {
            'num_groups': len(self.groups),
            'group_sizes': {k: len(v) for k, v in self.groups.items()},
            'total_agents': len(self.tls_ids),
            'largest_group': max(len(v) for v in self.groups.values()) if self.groups else 0,
            'smallest_group': min(len(v) for v in self.groups.values()) if self.groups else 0,
        }
