"""Tests F5: Baseline policies.

Verifica:
  - FixedTime siempre retorna la fase actual
  - MaxPressure retorna acción válida
  - Random retorna acción en rango
  - Todas implementan la interfaz BaselinePolicy
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from src.baselines import (
    FixedTimePolicy,
    MaxPressurePolicy,
    RandomPolicy,
    BaselinePolicy,
)


def make_mock_signals(n_agents=3, n_phases=3):
    """Crea un dict de señales mock para testing."""
    signals = {}
    agent_ids = [f"tls_{i}" for i in range(n_agents)]
    for i, tid in enumerate(agent_ids):
        sig = MagicMock()
        sig.current_green_idx = i % n_phases
        sig.num_green_phases = n_phases
        sig.green_states = ["GGrr", "rrGG", "GrGr"][:n_phases]
        sig.in_lanes = [f"in_{tid}_0", f"in_{tid}_1"]
        sig.out_lanes = [f"out_{tid}_0"]
        sig.lane_capacity = {
            f"in_{tid}_0": 10.0,
            f"in_{tid}_1": 10.0,
            f"out_{tid}_0": 10.0,
        }
        signals[tid] = sig
    return agent_ids, signals


class TestFixedTime:
    def test_is_baseline_policy(self):
        assert isinstance(FixedTimePolicy(), BaselinePolicy)

    def test_returns_current_phase(self):
        policy = FixedTimePolicy()
        agent_ids, signals = make_mock_signals()
        actions = policy.get_actions(agent_ids, signals)

        for tid in agent_ids:
            assert actions[tid] == signals[tid].current_green_idx

    def test_name(self):
        assert FixedTimePolicy().name == "FixedTime"


class TestMaxPressure:
    def test_is_baseline_policy(self):
        assert isinstance(MaxPressurePolicy(), BaselinePolicy)

    def test_returns_valid_actions(self):
        policy = MaxPressurePolicy()
        agent_ids, signals = make_mock_signals(n_phases=3)
        actions = policy.get_actions(agent_ids, signals)

        for tid in agent_ids:
            assert 0 <= actions[tid] < signals[tid].num_green_phases

    def test_name(self):
        assert MaxPressurePolicy().name == "MaxPressure"


class TestRandom:
    def test_is_baseline_policy(self):
        assert isinstance(RandomPolicy(), BaselinePolicy)

    def test_returns_valid_actions(self):
        policy = RandomPolicy(seed=42)
        agent_ids, signals = make_mock_signals(n_phases=4)

        for _ in range(100):
            actions = policy.get_actions(agent_ids, signals)
            for tid in agent_ids:
                assert 0 <= actions[tid] < signals[tid].num_green_phases

    def test_deterministic_with_same_seed(self):
        agent_ids, signals = make_mock_signals()

        p1 = RandomPolicy(seed=123)
        p2 = RandomPolicy(seed=123)

        a1 = p1.get_actions(agent_ids, signals)
        a2 = p2.get_actions(agent_ids, signals)

        assert a1 == a2

    def test_name(self):
        assert RandomPolicy().name == "Random"


class TestAllPoliciesSameInterface:
    def test_all_return_dict(self):
        agent_ids, signals = make_mock_signals()
        policies = [FixedTimePolicy(), MaxPressurePolicy(), RandomPolicy()]

        for policy in policies:
            actions = policy.get_actions(agent_ids, signals)
            assert isinstance(actions, dict)
            assert set(actions.keys()) == set(agent_ids)
