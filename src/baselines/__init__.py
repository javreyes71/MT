"""Políticas baseline no aprendidas para comparación justa.

Todas usan la misma interfaz: reciben un dict de observaciones por agente
y devuelven un dict de acciones por agente.

Políticas:
  - FixedTime: deja el programa estático (no hace nada, rotación natural)
  - MaxPressure: elige la fase con más presión entrante (Varaiya 2013)
  - Random: acción aleatoria uniforme (sanity check)
"""

import numpy as np
import libsumo as traci
from typing import Dict, List
from abc import ABC, abstractmethod


class BaselinePolicy(ABC):
    """Interfaz común para todas las políticas baseline."""

    @abstractmethod
    def get_actions(
        self, agent_ids: List[str], signals: dict
    ) -> Dict[str, int]:
        """Retorna un dict {tls_id: action_index}."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class FixedTimePolicy(BaselinePolicy):
    """Deja la fase actual sin cambios (rotación cíclica del entorno).

    Como TrafficSignal tiene anti-starvation (must_switch en g_max),
    las fases rotan automáticamente sin intervención del agente.
    Equivale al programa estático de SUMO.
    """

    @property
    def name(self) -> str:
        return "FixedTime"

    def get_actions(self, agent_ids, signals):
        # Siempre pedir mantener la fase actual → rotación forzada por g_max
        return {tid: signals[tid].current_green_idx for tid in agent_ids}


class MaxPressurePolicy(BaselinePolicy):
    """Elige la fase verde que tiene mayor presión entrante (Varaiya 2013).

    Para cada fase verde, calcula la presión como la diferencia entre
    vehículos entrantes y salientes en los movimientos de esa fase.
    Selecciona la fase con mayor presión.
    """

    @property
    def name(self) -> str:
        return "MaxPressure"

    def get_actions(self, agent_ids, signals):
        actions = {}
        for tid in agent_ids:
            ts = signals[tid]
            
            # Cachear los carriles que corresponden a cada fase verde
            if not hasattr(ts, "_phase_in_lanes"):
                ts._phase_in_lanes = []
                ts._phase_out_lanes = []
                links = traci.trafficlight.getControlledLinks(tid)
                for state in ts.green_states:
                    in_l = set()
                    out_l = set()
                    for idx, char in enumerate(state):
                        if char in ("G", "g"):
                            if idx < len(links) and links[idx]:
                                for link in links[idx]:
                                    in_l.add(link[0])
                                    out_l.add(link[1])
                    ts._phase_in_lanes.append(list(in_l))
                    ts._phase_out_lanes.append(list(out_l))

            best_phase = 0
            best_pressure = -float("inf")

            for phase_idx, state in enumerate(ts.green_states):
                pressure = 0.0
                
                # Presión entrante de esta fase
                for lane in ts._phase_in_lanes[phase_idx]:
                    try:
                        veh = traci.lane.getLastStepVehicleNumber(lane)
                        cap = ts.lane_capacity.get(lane, 10.0)
                        pressure += veh / max(cap, 1.0)
                    except Exception:
                        pass
                
                # Presión saliente de esta fase
                for lane in ts._phase_out_lanes[phase_idx]:
                    try:
                        veh = traci.lane.getLastStepVehicleNumber(lane)
                        cap = ts.lane_capacity.get(lane, 10.0)
                        pressure -= veh / max(cap, 1.0)
                    except Exception:
                        pass

                if pressure > best_pressure:
                    best_pressure = pressure
                    best_phase = phase_idx

            actions[tid] = best_phase
        return actions


class RandomPolicy(BaselinePolicy):
    """Acción aleatoria uniforme (sanity check)."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    @property
    def name(self) -> str:
        return "Random"

    def get_actions(self, agent_ids, signals):
        return {
            tid: self.rng.randint(0, signals[tid].num_green_phases)
            for tid in agent_ids
        }


class SumoNativePolicy(BaselinePolicy):
    """Base class for SUMO native algorithms (Actuated, DelayBased)."""
    def __init__(self, type_id: int, name: str):
        self.type_id = type_id
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_actions(self, agent_ids, signals):
        return None

class AdaptiveActuatedPolicy(SumoNativePolicy):
    """Proxy para SCATS (Time-Gap basado en densidad)."""
    def __init__(self):
        super().__init__(2, "Adaptive_Actuated_SCATS")

class AdaptiveDelayBasedPolicy(SumoNativePolicy):
    """Proxy para SCOOT (Minimización de retraso proyectado)."""
    def __init__(self):
        super().__init__(3, "Adaptive_DelayBased_SCOOT")
