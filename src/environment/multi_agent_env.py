"""Entorno multi-agente v2 para control de tráfico descentralizado.

Usa OPW v2 con build_observation para crear observaciones con:
  - can_act flag, type one-hot, lane/phase masks
  - vecinos reales via NetworkGraph BFS
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, List

from src.environment.traffic_env import TrafficSumoEnv
from src.utils.network_graph import NetworkGraph
from src.utils.config import load_config
from src.environment.observation_padding import ObservationPaddingWrapper
from src.utils.dmsgl import IntersectionGrouper


class MultiAgentTrafficEnv:
    """Entorno multi-agente con observación MDP v2 (§3.1)."""

    def __init__(self, config: dict = None, gui: bool = False):
        if config is None:
            config = load_config("config/default.yaml")

        self.config = config
        self.base_env = TrafficSumoEnv(config=config, gui=gui)

        net_file = config.get("simulation", {}).get("net_file", "sumo/network.net.xml")
        self.network_graph = NetworkGraph(net_file)

        self.agent_ids = self.base_env.tls_ids
        self.num_agents = len(self.agent_ids)
        self.comm_radius = config.get("marl", {}).get("communication_radius", 1)

        # Max phases across all controlled TLS
        self.max_phases = max(
            ts.num_green_phases for ts in self.base_env.signals.values()
        )

        # Grouper for intersection types
        self.grouper = IntersectionGrouper(self.agent_ids, self.base_env.action_dims)
        all_groups = sorted(self.grouper.get_all_groups().keys())
        self._group_to_idx = {g: i for i, g in enumerate(all_groups)}
        self.n_groups = len(all_groups)

        # OPW v2
        self.opw = ObservationPaddingWrapper(
            max_lanes=self.base_env.max_lanes,
            max_neighbors=0,  # not used in v2 build_observation
            max_phases=self.max_phases,
            n_groups=self.n_groups,
        )
        self.local_obs_size = self.opw.padded_obs_dim

        # Spaces
        self._observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.local_obs_size,), dtype=np.float32
        )
        self._action_space = spaces.Discrete(self.max_phases)

    def reset(self) -> Dict[str, np.ndarray]:
        self.base_env.reset()
        return {aid: self._get_local_obs(aid) for aid in self.agent_ids}

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        action_list = []
        for i, aid in enumerate(self.agent_ids):
            a = actions.get(aid, 0)
            max_a = self.base_env.action_dims[i] - 1
            action_list.append(min(a, max_a))

        _, global_reward, terminated, truncated, base_info = self.base_env.step(
            action_list
        )

        obs = {aid: self._get_local_obs(aid) for aid in self.agent_ids}
        rewards = {aid: global_reward for aid in self.agent_ids}
        terminateds = {aid: terminated for aid in self.agent_ids}
        truncateds = {aid: truncated for aid in self.agent_ids}
        infos = {aid: base_info for aid in self.agent_ids}

        return obs, rewards, terminateds, truncateds, infos

    def _get_local_obs(self, agent_id: str) -> np.ndarray:
        """Construye la observación MDP v2 usando OPW.build_observation."""
        import traci

        ts = self.base_env.signals[agent_id]
        max_lanes = self.base_env.max_lanes

        # 1. Halt counts normalizados por capacidad [0,1]
        halts: List[float] = []
        for lane in ts.in_lanes:
            try:
                h = traci.lane.getLastStepHaltingNumber(lane)
                cap = ts.lane_capacity.get(lane, 10.0)
                halts.append(min(h / cap, 1.0))
            except Exception:
                halts.append(0.0)

        # 2. Ocupación entrante [0,1]
        in_occ: List[float] = []
        for lane in ts.in_lanes:
            try:
                in_occ.append(traci.lane.getLastStepOccupancy(lane) / 100.0)
            except Exception:
                in_occ.append(0.0)

        # 3. Ocupación saliente media [0,1]
        try:
            if ts.out_lanes:
                out_occs = [
                    traci.lane.getLastStepOccupancy(l) / 100.0 for l in ts.out_lanes
                ]
                avg_out_occ = sum(out_occs) / len(out_occs)
            else:
                avg_out_occ = 0.0
        except Exception:
            avg_out_occ = 0.0

        # 4. Phase norm
        phase_norm = ts.current_green_idx / max(ts.num_green_phases - 1, 1)

        # 5. Can act flag
        can_act = ts.can_act()

        # 6. Time norm
        time_norm = min(ts.time_since_switch / self.base_env.g_max, 1.0)

        # 7. Local pressure [-1,1]
        try:
            in_veh = sum(traci.lane.getLastStepVehicleNumber(l) for l in ts.in_lanes)
            in_cap = sum(ts.lane_capacity.get(l, 10.0) for l in ts.in_lanes)
            out_veh = (
                sum(traci.lane.getLastStepVehicleNumber(l) for l in ts.out_lanes)
                if ts.out_lanes else 0
            )
            out_cap = (
                sum(ts.lane_capacity.get(l, 10.0) for l in ts.out_lanes)
                if ts.out_lanes else 1.0
            )
            local_pressure = (in_veh / max(in_cap, 1.0)) - (out_veh / max(out_cap, 1.0))
        except Exception:
            local_pressure = 0.0

        # 8-9. Neighbor summaries
        neighbors = self.network_graph.get_neighbors(agent_id, self.comm_radius)
        n_pressure_sum = 0.0
        n_queue_sum = 0.0
        n_count = 0
        for nid in neighbors:
            if nid in self.base_env.signals:
                n_ts = self.base_env.signals[nid]
                try:
                    n_in = sum(traci.lane.getLastStepVehicleNumber(l) for l in n_ts.in_lanes)
                    n_in_cap = sum(n_ts.lane_capacity.get(l, 10.0) for l in n_ts.in_lanes)
                    n_out = (
                        sum(traci.lane.getLastStepVehicleNumber(l) for l in n_ts.out_lanes)
                        if n_ts.out_lanes else 0
                    )
                    n_out_cap = (
                        sum(n_ts.lane_capacity.get(l, 10.0) for l in n_ts.out_lanes)
                        if n_ts.out_lanes else 1.0
                    )
                    n_pressure_sum += abs(n_in / max(n_in_cap, 1.0) - n_out / max(n_out_cap, 1.0))

                    n_halts = sum(traci.lane.getLastStepHaltingNumber(l) for l in n_ts.in_lanes)
                    n_queue_sum += n_halts / max(n_in_cap, 1.0)
                    n_count += 1
                except Exception:
                    pass

        neighbor_pressure_avg = n_pressure_sum / max(n_count, 1)
        neighbor_queue_avg = n_queue_sum / max(n_count, 1)

        # 10. Group index
        group_id = self.grouper.get_group(agent_id)
        group_idx = self._group_to_idx.get(group_id, 0)

        return self.opw.build_observation(
            halts=halts,
            in_occ=in_occ,
            avg_out_occ=avg_out_occ,
            phase_norm=phase_norm,
            can_act=can_act,
            time_norm=time_norm,
            local_pressure=local_pressure,
            neighbor_pressure_avg=neighbor_pressure_avg,
            neighbor_queue_avg=neighbor_queue_avg,
            group_idx=group_idx,
            actual_lanes=len(ts.in_lanes),
            actual_phases=ts.num_green_phases,
        )

    def get_action_mask(self, agent_id: str) -> np.ndarray:
        """Retorna la máscara de acciones válidas para MaskablePPO."""
        ts = self.base_env.signals[agent_id]
        mask = np.zeros(self.max_phases, dtype=np.float32)
        mask[:ts.num_green_phases] = 1.0
        return mask

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Discrete:
        return self._action_space

    def close(self) -> None:
        self.base_env.close()
