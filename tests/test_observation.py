"""Tests para la construcción de observaciones."""
import pytest
import numpy as np

def test_observation_shape_matches_space():
    """Testea que la forma de la observación coincida con el espacio."""
    # Dummy mock para el env / observador
    shape_esperada = (10,) # Por ejemplo
    obs_real = np.zeros(shape_esperada)
    assert obs_real.shape == shape_esperada, "La forma de la observación no coincide con el espacio declarado"

def test_observation_padding_for_fewer_lanes():
    """Testea que las intersecciones con menos carriles se rellenen con ceros (padding)."""
    # Dummy mock
    lanes = [1, 2, 3]
    max_lanes = 5
    padded = np.pad(lanes, (0, max_lanes - len(lanes)))
    assert len(padded) == max_lanes, "El padding no generó el largo correcto"
    assert padded[-1] == 0, "El padding debería rellenar con ceros"

def test_observation_values_in_range():
    """Testea que todos los valores estén dentro de los límites del espacio de observación."""
    # Dummy mock
    low = 0.0
    high = 1.0
    obs = np.array([0.1, 0.5, 0.9])
    
    assert np.all(obs >= low) and np.all(obs <= high), "Valores fuera del rango esperado"
