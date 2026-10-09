"""Entorno Gymnasium para control de semÃ¡foros en SUMO.

VersiÃ³n v2: usa TrafficSignal por TLS para que SUMO nunca controle
las fases autÃ³nomamente (corrige B1).  Cada TLS tiene su propia
mÃ¡quina de estados verdeâ†”amarillo con g_min/g_max reales.
"""

import gymnasium as gym
from gymnasium import spaces
import os
import sumolib
import numpy as np

if os.environ.get("USE_TRACI", "0") == "1":
    import traci
else:
    import libsumo as traci
import sys
import logging
from typing import Tuple, Dict, Any, Optional, List

from src.utils.config import load_config
from src.rewards.base import RewardManager
from src.environment.traffic_signal import TrafficSignal

logger = logging.getLogger(__name__)


class TrafficSumoEnv(gym.Env):
    """Entorno multi-semÃ¡foro con control de fases propio (patrÃ³n SUMO-RL)."""

    def __init__(self, config: dict = None, gui: bool = False):
        super().__init__()
        self.gui = gui

        if config is None:
            config = load_config("config/default.yaml")
        self.config = config

        sim = config.get("simulation", {})
        self.delta_time: float = sim.get("delta_time", 5.0)
        self.g_min: float = sim.get("min_phase_time", 15.0)
        self.g_max: float = sim.get("max_phase_time", 35.0)
        self.yellow_time: float = sim.get("yellow_time", 3.0)
        self.sim_duration: int = sim.get("duration", 3600)

        # Recompensa
        reward_config = config.get("reward", {})
        self.reward_manager = RewardManager.from_config(reward_config)

        # SUMO paths
        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            if tools not in sys.path:
                sys.path.append(tools)
        else:
            sys.exit("Error: Declara la variable de entorno 'SUMO_HOME'")

        self._sumo_binary = (
            sumolib.checkBinary("sumo-gui") if self.gui else sumolib.checkBinary("sumo")
        )
        self.config_file = sim.get("config_file", "sumo/simulation.sumocfg")
        self.net_file = sim.get("net_file", "sumo/network.net.xml")

        # ------------------------------------------------------------------
        # Leer la red y construir los TrafficSignal
        # ------------------------------------------------------------------
        net = sumolib.net.readNet(self.net_file, withPrograms=True)
        controlled_tls = sim.get("controlled_tls", None)

        self.signals: Dict[str, TrafficSignal] = {}
        for tls_obj in net.getTrafficLights():
            tls_id = tls_obj.getID()

            # Filtrar por lista de controlados
            if controlled_tls is not None and tls_id not in controlled_tls:
                continue

            progs = tls_obj.getPrograms()
            if not progs:
                continue

            # Solo incluir TLS con â‰¥2 fases verdes
            prog = progs.get("0", next(iter(progs.values())))
            n_greens = sum(
                1
                for p in prog.getPhases()
                if ("G" in p.state or "g" in p.state) and "y" not in p.state
            )
            if n_greens < 2:
                continue

            try:
                ts = TrafficSignal(
                    tls_id, tls_obj, net,
                    g_min=self.g_min,
                    g_max=self.g_max,
                    yellow_time=self.yellow_time,
                )
                self.signals[tls_id] = ts
            except Exception as e:
                logger.warning(f"No se pudo crear TrafficSignal para {tls_id}: {e}")

        # Listas ordenadas (determinismo)
        self.tls_ids: List[str] = sorted(self.signals.keys())
        self.num_agents: int = len(self.tls_ids)

        if self.num_agents == 0:
            raise ValueError(
                "No se encontraron semÃ¡foros controlables con >=2 fases verdes."
            )

        logger.info(f"Semaforos controlados: {self.num_agents}")

        # Dimensiones de acciÃ³n por TLS
        self.action_dims: List[int] = [
            self.signals[tid].num_green_phases for tid in self.tls_ids
        ]

        # Propiedades de conveniencia para multi_agent_env
        self.max_lanes: int = max(
            len(ts.in_lanes) for ts in self.signals.values()
        )
        self.lanes_per_tls: Dict[str, List[str]] = {
            tid: self.signals[tid].in_lanes for tid in self.tls_ids
        }
        self.outgoing_lanes_per_tls: Dict[str, List[str]] = {
            tid: self.signals[tid].out_lanes for tid in self.tls_ids
        }
        self.lane_capacity: Dict[str, float] = {}
        for ts in self.signals.values():
            self.lane_capacity.update(ts.lane_capacity)

        # ------------------------------------------------------------------
        # Espacios de Gymnasium (centralizado)
        # ------------------------------------------------------------------
        self.action_space = spaces.MultiDiscrete(self.action_dims)

        obs_per_tls = self.max_lanes + 4  # halts + in_occ + out_occ + phase + time
        total_obs = self.num_agents * obs_per_tls
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_obs,), dtype=np.float32
        )

        # Runtime state
        self.step_count: int = 0
        self.connection = None

    # ==================================================================
    # Setup / Reset
    # ==================================================================
    def setup(self, seed: Optional[int] = None) -> None:
        try:
            traci.close()
        except Exception:
            pass

        if seed is None:
            seed = self.config.get("training", {}).get("seed", 42)
            
        cmd = [
            self._sumo_binary,
            "-c", self.config_file,
            "-n", self.net_file,
            "--seed", str(seed),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--start",
        ]
        
        sim_config = self.config.get("simulation", {})
        if "route_files" in sim_config:
            cmd.extend(["--route-files", sim_config["route_files"]])
        if "tripinfo_output" in sim_config:
            cmd.extend(["--tripinfo-output", sim_config["tripinfo_output"]])

        traci.start(cmd)
        self.connection = traci

        # Inicializar cada TrafficSignal: desactivar programa estÃ¡tico
        for tid in self.tls_ids:
            self.signals[tid].init_at_reset(traci)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.setup(seed=seed)
        self.step_count = 0
        return self._get_observation(), {}

    # ==================================================================
    # Step
    # ==================================================================
    def step(
        self, actions=None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        # 1. Aplicar acciones a cada TLS
        if actions is not None:
            for i, tid in enumerate(self.tls_ids):
                ts = self.signals[tid]
                action = int(actions[i])
                ts.apply_action(action, traci)

        # 2. Avanzar la simulaciÃ³n delta_time segundos
        steps = int(self.delta_time)
        for _ in range(steps):
            traci.simulationStep()
            # Tick de las mÃ¡quinas de estado (1 segundo por simulationStep)
            for tid in self.tls_ids:
                self.signals[tid].tick(1.0, traci)

        # 3. ObservaciÃ³n y recompensa
        obs = self._get_observation()
        env_info = self.get_env_info()
        rewards_dict = self.reward_manager.calculate_per_agent(env_info)
        reward = sum(rewards_dict.values()) / max(len(rewards_dict), 1)

        # 4. TerminaciÃ³n
        total_vehicles = traci.vehicle.getIDCount()
        sim_time = traci.simulation.getTime()
        terminated = total_vehicles == 0 and self.step_count > 50
        truncated = sim_time >= self.sim_duration

        return obs, reward, terminated, truncated, {
            "vehicles": total_vehicles,
            "sim_time": sim_time,
            "env_info": env_info,
            "rewards_dict": rewards_dict
        }

    # ==================================================================
    # ObservaciÃ³n
    # ==================================================================
    def _get_observation(self) -> np.ndarray:
        """ObservaciÃ³n normalizada en [-1, 1] o [0, 1].

        Por TLS: [halt_counts(max_lanes), avg_in_occ, avg_out_occ,
                  phase_norm, time_in_phase_norm]
        """
        full_obs: List[float] = []

        for tid in self.tls_ids:
            ts = self.signals[tid]

            # 1. Halt counts por carril entrante (normalizado por capacidad)
            halts: List[float] = []
            for lane in ts.in_lanes:
                try:
                    h = traci.lane.getLastStepHaltingNumber(lane)
                    cap = ts.lane_capacity.get(lane, 10.0)
                    halts.append(min(h / cap, 1.0))
                except Exception:
                    halts.append(0.0)
            # Pad a max_lanes
            while len(halts) < self.max_lanes:
                halts.append(0.0)
            full_obs.extend(halts[: self.max_lanes])

            # 2. OcupaciÃ³n promedio entrante [0, 1]
            try:
                in_occs = [
                    traci.lane.getLastStepOccupancy(l) / 100.0 for l in ts.in_lanes
                ]
                avg_in = sum(in_occs) / max(len(in_occs), 1)
            except Exception:
                avg_in = 0.0
            full_obs.append(avg_in)

            # 3. OcupaciÃ³n promedio saliente [0, 1]
            try:
                if ts.out_lanes:
                    out_occs = [
                        traci.lane.getLastStepOccupancy(l) / 100.0 for l in ts.out_lanes
                    ]
                    avg_out = sum(out_occs) / max(len(out_occs), 1)
                else:
                    avg_out = 0.0
            except Exception:
                avg_out = 0.0
            full_obs.append(avg_out)

            # 4. Fase actual (normalizada)
            phase_norm = ts.current_green_idx / max(ts.num_green_phases - 1, 1)
            full_obs.append(phase_norm)

            # 5. Tiempo en fase (normalizado por g_max)
            t_norm = min(ts.time_since_switch / self.g_max, 1.0)
            full_obs.append(t_norm)

        return np.array(full_obs, dtype=np.float32)

    # ==================================================================
    # Info para recompensa
    # ==================================================================
    def get_env_info(self) -> Dict[str, Any]:
        """InformaciÃ³n del entorno para el cÃ¡lculo de recompensas."""
        agents_info = {}
        halt_counts: List[int] = []
        wait_times: List[float] = []
        total_co2 = 0.0
        total_pressure = 0.0
        tls_pressures: Dict[str, float] = {}
        changes = 0

        for tid in self.tls_ids:
            ts = self.signals[tid]
            try:
                h = 0
                w = 0.0
                c = 0.0
                for lane in ts.in_lanes:
                    h_l = traci.lane.getLastStepHaltingNumber(lane)
                    w_l = traci.lane.getWaitingTime(lane)
                    c_l = traci.lane.getCO2Emission(lane)
                    halt_counts.append(h_l)
                    wait_times.append(w_l)
                    total_co2 += c_l
                    h += h_l
                    w += w_l
                    c += c_l

                in_veh = sum(
                    traci.lane.getLastStepVehicleNumber(l) for l in ts.in_lanes
                )
                in_cap = sum(ts.lane_capacity.get(l, 10.0) for l in ts.in_lanes)
                out_veh = (
                    sum(traci.lane.getLastStepVehicleNumber(l) for l in ts.out_lanes)
                    if ts.out_lanes
                    else 0
                )
                out_cap = (
                    sum(ts.lane_capacity.get(l, 10.0) for l in ts.out_lanes)
                    if ts.out_lanes
                    else 1.0
                )

                # Fix H5: Eliminado el abs() para mantener la propiedad vectorial de Max-Pressure.
                pressure = (in_veh / max(in_cap, 1.0)) - (out_veh / max(out_cap, 1.0))
                tls_pressures[tid] = pressure
                total_pressure += abs(pressure)

                if ts.is_yellow:
                    changes += 1

                agents_info[tid] = {
                    "halts": h,
                    "wait": w,
                    "co2": c,
                    "capacity": max(in_cap, 1.0),
                    "pressure": pressure,
                    "is_yellow": ts.is_yellow
                }
            except Exception as e:
                logger.warning(f"Error obteniendo info de {tid}: {e}")
                agents_info[tid] = {"halts": 0, "wait": 0.0, "co2": 0.0, "capacity": 1.0, "pressure": 0.0, "is_yellow": False}

        try:
            throughput = traci.simulation.getArrivedNumber()
        except Exception:
            throughput = 0

        num_tls = len(self.tls_ids)
        return {
            "agents_info": agents_info,
            "halt_counts": halt_counts,
            "wait_times": wait_times,
            "total_co2": total_co2,
            "throughput": throughput,
            "avg_pressure": total_pressure / max(num_tls, 1),
            "phase_changes": changes,
            "num_tls": num_tls,
            "neighbor_pressures": list(tls_pressures.values()),
            "tls_phases": {
                tid: self.signals[tid].current_green_idx for tid in self.tls_ids
            },
        }

    # ==================================================================
    # Close
    # ==================================================================
    def close(self) -> None:
        try:
            traci.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        if "connection" in state:
            state["connection"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

