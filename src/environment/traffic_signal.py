"""Controlador de fases por semáforo (patrón SUMO-RL).

Cada instancia de TrafficSignal gestiona un único TLS con una máquina
de estados propia (verde ↔ amarillo), generando los strings de estado
con `setRedYellowGreenState` para que SUMO **nunca** avance el programa
estático por su cuenta.

Corrige B1 (SUMO pelea con el agente) y M7 (amarillo global).
"""

import logging
from typing import List, Dict, Optional, Tuple
import sumolib

logger = logging.getLogger(__name__)


class TrafficSignal:
    """Máquina de estados de un semáforo individual.

    Estados internos
    ----------------
    - ``green``: mostrando una fase verde; el agente puede actuar si
      ``time_since_switch >= g_min``.
    - ``yellow``: transición obligatoria; al terminar vuelve a ``green``
      con la fase objetivo.

    Atributos públicos
    ------------------
    green_states : list[str]
        Strings de señalización verde extraídos del programa original.
    yellow_states : dict[int, dict[int, str]]
        ``yellow_states[from_idx][to_idx]`` → string donde las luces que
        cambian pasan a 'y' y el resto se mantiene en rojo.
    num_green_phases : int
        Número de fases verdes (= tamaño del espacio de acciones).
    in_lanes : list[str]
        Carriles entrantes (orden determinista por linkIndex).
    out_lanes : list[str]
        Carriles salientes (orden determinista).
    """

    def __init__(
        self,
        tls_id: str,
        sumolib_tls,
        sumolib_net,
        g_min: float = 15.0,
        g_max: float = 35.0,
        yellow_time: float = 3.0,
    ):
        self.id = tls_id
        self.g_min = g_min
        self.g_max = g_max
        self.yellow_time = yellow_time

        # --- Extraer fases verdes del programa estático ---
        progs = sumolib_tls.getPrograms()
        if not progs:
            raise ValueError(f"TLS {tls_id} no tiene programas definidos")
        prog = progs.get("0", next(iter(progs.values())))
        all_phases = prog.getPhases()

        self.green_states: List[str] = []
        self._green_indices: List[int] = []       # índices en el programa original

        for i, phase in enumerate(all_phases):
            state = phase.state
            has_green = ("G" in state or "g" in state)
            has_yellow = "y" in state
            if has_green and not has_yellow:
                self.green_states.append(state)
                self._green_indices.append(i)

        if not self.green_states:
            # Fallback: usar la primera fase como verde
            self.green_states = [all_phases[0].state]
            self._green_indices = [0]

        self.num_green_phases: int = len(self.green_states)

        # --- Generar amarillos automáticos (G/g → y, resto → r) ---
        self.yellow_states: Dict[int, Dict[int, str]] = {}
        for from_idx, from_state in enumerate(self.green_states):
            self.yellow_states[from_idx] = {}
            for to_idx, to_state in enumerate(self.green_states):
                if from_idx == to_idx:
                    continue
                yellow = self._make_yellow(from_state, to_state)
                self.yellow_states[from_idx][to_idx] = yellow

        # --- Carriles entrantes y salientes (deterministas) ---
        connections = sumolib_tls.getConnections()
        in_lanes_set: Dict[str, int] = {}   # lane_id → min linkIndex
        out_lanes_set: Dict[str, int] = {}
        for conn in connections:
            in_lane = conn[0]
            out_lane = conn[1]
            link_idx = conn[2]

            lid = in_lane.getID()
            if lid not in in_lanes_set or link_idx < in_lanes_set[lid]:
                in_lanes_set[lid] = link_idx

            oid = out_lane.getID()
            if not oid.startswith(":"):
                if oid not in out_lanes_set or link_idx < out_lanes_set[oid]:
                    out_lanes_set[oid] = link_idx

        # Orden determinista por linkIndex (corrige B6)
        self.in_lanes: List[str] = sorted(in_lanes_set, key=lambda x: in_lanes_set[x])
        self.out_lanes: List[str] = sorted(out_lanes_set, key=lambda x: out_lanes_set[x])

        # Capacidad de carriles (vehículos que caben) precalculada
        self.lane_capacity: Dict[str, float] = {}
        for lid in self.in_lanes + self.out_lanes:
            try:
                edge_id = lid.rsplit("_", 1)[0]
                edge = sumolib_net.getEdge(edge_id)
                lane = next(l for l in edge.getLanes() if l.getID() == lid)
                self.lane_capacity[lid] = max(1.0, lane.getLength() / 7.5)
            except Exception:
                self.lane_capacity[lid] = 10.0

        # --- Estado de la máquina ---
        self._current_green_idx: int = 0
        self._state: str = "green"       # "green" | "yellow"
        self._target_green_idx: int = 0  # usado solo durante "yellow"
        self._time_since_switch: float = 0.0
        self._yellow_countdown: float = 0.0

    # ------------------------------------------------------------------
    # Generación de amarillo
    # ------------------------------------------------------------------
    @staticmethod
    def _make_yellow(from_state: str, to_state: str) -> str:
        """Genera el string amarillo de transición.

        Regla: si una luz estaba en G/g y en el destino NO estará en G/g,
        pasa a 'y'; el resto pasa a 'r' (todo-rojo excepto los amarillos).
        """
        yellow = []
        for f, t in zip(from_state, to_state):
            if f in ("G", "g") and t not in ("G", "g"):
                yellow.append("y")
            elif f in ("G", "g") and t in ("G", "g"):
                # Se mantiene verde (no cambia)
                yellow.append(f)
            else:
                yellow.append("r")
        return "".join(yellow)

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------
    @property
    def current_green_idx(self) -> int:
        return self._current_green_idx

    @property
    def time_since_switch(self) -> float:
        return self._time_since_switch

    @property
    def is_yellow(self) -> bool:
        return self._state == "yellow"

    def can_act(self) -> bool:
        """El agente puede cambiar de fase si estamos en verde y
        se cumplió el tiempo mínimo de verde."""
        return self._state == "green" and self._time_since_switch >= self.g_min

    def must_switch(self) -> bool:
        """Forzar cambio si se excedió g_max (anti-starvation)."""
        return (
            self._state == "green"
            and self._time_since_switch >= self.g_max
            and self.num_green_phases > 1
        )

    def apply_action(self, action: int, traci_conn) -> None:
        """Procesa la acción del agente.

        Si ``can_act()`` es True y la acción pide una fase distinta, inicia
        la transición amarilla.  Si la acción pide la fase actual, no hace nada.
        Si ``must_switch()``, fuerza la rotación a la siguiente fase.
        """
        # Clamp
        action = min(action, self.num_green_phases - 1)

        if self._state == "yellow":
            # Ignorar acciones durante amarillo
            return

        target_idx = action

        # Anti-starvation: si excedimos g_max, forzar rotación
        if self.must_switch() and target_idx == self._current_green_idx:
            target_idx = (self._current_green_idx + 1) % self.num_green_phases

        if target_idx != self._current_green_idx and self.can_act():
            self._start_yellow(target_idx, traci_conn)
        elif self.must_switch() and target_idx != self._current_green_idx:
            self._start_yellow(target_idx, traci_conn)

    def tick(self, dt: float, traci_conn) -> None:
        """Avanza la máquina de estados por ``dt`` segundos de simulación."""
        self._time_since_switch += dt

        if self._state == "yellow":
            self._yellow_countdown -= dt
            if self._yellow_countdown <= 0:
                # Fin del amarillo → aplicar verde objetivo
                self._state = "green"
                self._current_green_idx = self._target_green_idx
                self._time_since_switch = 0.0
                try:
                    traci_conn.trafficlight.setRedYellowGreenState(
                        self.id, self.green_states[self._current_green_idx]
                    )
                except Exception as e:
                    logger.warning(f"Error aplicando verde en {self.id}: {e}")

    def init_at_reset(self, traci_conn) -> None:
        """Inicializa el TLS al hacer reset del entorno.

        Fuerza la primera fase verde y desactiva el programa estático
        de SUMO para que no interfiera.
        """
        self._current_green_idx = 0
        self._state = "green"
        self._time_since_switch = 0.0
        self._yellow_countdown = 0.0
        self._target_green_idx = 0

        try:
            # Desactivar el programa estático poniéndolo en modo "off"
            # y luego forzando nuestro estado manualmente
            traci_conn.trafficlight.setProgram(self.id, "off")
            traci_conn.trafficlight.setRedYellowGreenState(
                self.id, self.green_states[0]
            )
        except Exception as e:
            logger.warning(f"Error inicializando TLS {self.id}: {e}")

    def _start_yellow(self, target_idx: int, traci_conn) -> None:
        """Inicia la transición amarilla hacia ``target_idx``."""
        yellow_state = self.yellow_states.get(self._current_green_idx, {}).get(
            target_idx
        )
        if yellow_state is None:
            # No hay transición amarilla definida → cambio directo
            self._current_green_idx = target_idx
            self._time_since_switch = 0.0
            try:
                traci_conn.trafficlight.setRedYellowGreenState(
                    self.id, self.green_states[target_idx]
                )
            except Exception as e:
                logger.warning(f"Error cambio directo en {self.id}: {e}")
            return

        self._state = "yellow"
        self._target_green_idx = target_idx
        self._yellow_countdown = self.yellow_time
        try:
            traci_conn.trafficlight.setRedYellowGreenState(self.id, yellow_state)
        except Exception as e:
            logger.warning(f"Error aplicando amarillo en {self.id}: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"TrafficSignal(id={self.id}, phases={self.num_green_phases}, "
            f"state={self._state}, green_idx={self._current_green_idx}, "
            f"t={self._time_since_switch:.1f}s)"
        )
