"""Tests de invariantes de control de fases para TrafficSignal.

Verifica:
  - Ningún verde < g_min ni > g_max
  - Todo cambio de fase precedido por yellow_time de amarillo
  - Cero cambios no comandados (SUMO no avanza por su cuenta)
  - Orden determinista de carriles (misma semilla → mismo orden)
"""

import pytest
from unittest.mock import MagicMock, patch
from src.environment.traffic_signal import TrafficSignal


class FakeSumolibPhase:
    def __init__(self, state: str):
        self.state = state


class FakeSumolibProgram:
    def __init__(self, phases):
        self._phases = [FakeSumolibPhase(s) for s in phases]

    def getPhases(self):
        return self._phases


class FakeSumolibLane:
    def __init__(self, lane_id: str, length: float = 100.0):
        self._id = lane_id
        self._length = length
        self._edge = FakeSumolibEdge(lane_id.rsplit("_", 1)[0], [self])

    def getID(self):
        return self._id

    def getLength(self):
        return self._length

    def getEdge(self):
        return self._edge


class FakeSumolibEdge:
    def __init__(self, edge_id: str, lanes):
        self._id = edge_id
        self._lanes = lanes

    def getID(self):
        return self._id

    def getLanes(self):
        return self._lanes

    def getName(self):
        return ""


class FakeSumolibTLS:
    def __init__(self, tls_id: str, phases: list, connections: list):
        self._id = tls_id
        self._prog = FakeSumolibProgram(phases)
        self._connections = connections

    def getID(self):
        return self._id

    def getPrograms(self):
        return {"0": self._prog}

    def getConnections(self):
        return self._connections


class FakeSumolibNet:
    def __init__(self, edges):
        self._edges = {e.getID(): e for e in edges}

    def getEdge(self, eid):
        return self._edges[eid]


def make_simple_tls(tls_id="test_tls"):
    """Crea un TLS simple con 2 fases verdes y 2 carriles."""
    phases = ["GGrr", "rrrr", "rrGG", "rrrr"]  # green0, allred, green1, allred

    lane_a = FakeSumolibLane("edge_a_0", 75.0)
    lane_b = FakeSumolibLane("edge_b_0", 75.0)
    out_a = FakeSumolibLane("edge_out_a_0", 75.0)
    out_b = FakeSumolibLane("edge_out_b_0", 75.0)

    connections = [
        (lane_a, out_a, 0),
        (lane_b, out_b, 1),
    ]

    tls = FakeSumolibTLS(tls_id, phases, connections)

    edges = [
        lane_a.getEdge(),
        lane_b.getEdge(),
        out_a.getEdge(),
        out_b.getEdge(),
    ]
    net = FakeSumolibNet(edges)

    return TrafficSignal(tls_id, tls, net, g_min=5.0, g_max=15.0, yellow_time=3.0)


class TestTrafficSignalCreation:
    def test_extracts_green_phases(self):
        ts = make_simple_tls()
        assert ts.num_green_phases == 2
        assert ts.green_states[0] == "GGrr"
        assert ts.green_states[1] == "rrGG"

    def test_generates_yellow_transitions(self):
        ts = make_simple_tls()
        # De fase 0 a fase 1: GG→rr debería dar yy, rr→GG debería dar rr
        yellow_0_to_1 = ts.yellow_states[0][1]
        assert "y" in yellow_0_to_1
        # De fase 1 a fase 0
        yellow_1_to_0 = ts.yellow_states[1][0]
        assert "y" in yellow_1_to_0

    def test_deterministic_lane_order(self):
        """Dos instancias con la misma red deben tener el mismo orden de carriles."""
        ts1 = make_simple_tls("tls_a")
        ts2 = make_simple_tls("tls_a")
        assert ts1.in_lanes == ts2.in_lanes
        assert ts1.out_lanes == ts2.out_lanes

    def test_lanes_sorted_by_link_index(self):
        ts = make_simple_tls()
        # lane_a tiene linkIndex=0, lane_b tiene linkIndex=1
        assert ts.in_lanes == ["edge_a_0", "edge_b_0"]


class TestStateMachine:
    def setup_method(self):
        self.ts = make_simple_tls()
        self.traci = MagicMock()

    def test_initial_state_is_green(self):
        assert not self.ts.is_yellow
        assert self.ts.current_green_idx == 0

    def test_cannot_act_before_g_min(self):
        """No debe permitir cambio antes del tiempo mínimo de verde."""
        assert not self.ts.can_act()  # time_since_switch = 0

        # Avanzar 3 segundos (< g_min=5)
        for _ in range(3):
            self.ts.tick(1.0, self.traci)
        assert not self.ts.can_act()

    def test_can_act_after_g_min(self):
        """Debe permitir cambio después del tiempo mínimo de verde."""
        for _ in range(5):
            self.ts.tick(1.0, self.traci)
        assert self.ts.can_act()

    def test_yellow_before_phase_change(self):
        """Todo cambio de fase debe pasar por un período de amarillo."""
        # Llegar a g_min
        for _ in range(5):
            self.ts.tick(1.0, self.traci)

        assert self.ts.can_act()

        # Pedir cambio a fase 1
        self.ts.apply_action(1, self.traci)

        # Debe estar en amarillo
        assert self.ts.is_yellow
        assert self.ts.current_green_idx == 0  # aún en la fase vieja

        # Verificar que se aplicó el estado amarillo via setRedYellowGreenState
        self.traci.trafficlight.setRedYellowGreenState.assert_called()

    def test_yellow_duration(self):
        """El amarillo debe durar exactamente yellow_time segundos."""
        for _ in range(5):
            self.ts.tick(1.0, self.traci)

        self.ts.apply_action(1, self.traci)
        assert self.ts.is_yellow

        # Avanzar 2 seg (< 3 seg de amarillo)
        for _ in range(2):
            self.ts.tick(1.0, self.traci)
        assert self.ts.is_yellow

        # Avanzar 1 seg más → fin del amarillo
        self.ts.tick(1.0, self.traci)
        assert not self.ts.is_yellow
        assert self.ts.current_green_idx == 1  # ahora en fase 1

    def test_no_action_during_yellow(self):
        """Las acciones deben ignorarse durante el período amarillo."""
        for _ in range(5):
            self.ts.tick(1.0, self.traci)
        self.ts.apply_action(1, self.traci)
        assert self.ts.is_yellow

        # Intentar otra acción durante amarillo → debe ignorarse
        call_count_before = self.traci.trafficlight.setRedYellowGreenState.call_count
        self.ts.apply_action(0, self.traci)
        call_count_after = self.traci.trafficlight.setRedYellowGreenState.call_count
        assert call_count_before == call_count_after  # sin llamadas extra

    def test_anti_starvation_forces_switch(self):
        """Debe forzar cambio si se excede g_max."""
        # Avanzar hasta g_max (15s)
        for _ in range(15):
            self.ts.tick(1.0, self.traci)

        assert self.ts.must_switch()

        # Pedir mantener la misma fase → debe rotar automáticamente
        self.ts.apply_action(0, self.traci)
        assert self.ts.is_yellow  # inició transición forzada

    def test_green_never_shorter_than_g_min(self):
        """Simula múltiples ciclos y verifica que ningún verde < g_min."""
        green_durations = []
        current_phase = self.ts.current_green_idx

        for step in range(100):
            self.ts.tick(1.0, self.traci)

            if self.ts.can_act() and step % 7 == 0:
                # Intentar cambiar cada 7 pasos
                target = (self.ts.current_green_idx + 1) % self.ts.num_green_phases
                old_phase = self.ts.current_green_idx
                self.ts.apply_action(target, self.traci)

                if self.ts.is_yellow and old_phase == current_phase:
                    green_durations.append(self.ts.time_since_switch)
                    current_phase = target

        # Todas las duraciones deben ser >= g_min
        for d in green_durations:
            assert d >= self.ts.g_min, f"Verde de {d}s < g_min de {self.ts.g_min}s"


class TestInitAtReset:
    def test_disables_static_program(self):
        ts = make_simple_tls()
        traci_mock = MagicMock()

        ts.init_at_reset(traci_mock)

        traci_mock.trafficlight.setProgram.assert_called_once_with(ts.id, "off")
        traci_mock.trafficlight.setRedYellowGreenState.assert_called_once_with(
            ts.id, ts.green_states[0]
        )

    def test_resets_state(self):
        ts = make_simple_tls()
        traci_mock = MagicMock()

        # Simular que estaba en medio de algo
        ts._current_green_idx = 1
        ts._time_since_switch = 20.0
        ts._state = "yellow"

        ts.init_at_reset(traci_mock)

        assert ts.current_green_idx == 0
        assert ts.time_since_switch == 0.0
        assert not ts.is_yellow
