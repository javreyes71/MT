import libsumo as traci
import numpy as np
import pandas as pd
import os
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MetricsCollector:
    """
    Módulo para la recolección de métricas de la simulación SUMO.
    Extrae información del estado actual a través de TraCI.
    """
    def __init__(self) -> None:
        """Inicializa el recolector de métricas."""
        self.reset()
        
    def reset(self) -> None:
        """Limpia los datos recolectados para un nuevo episodio."""
        self.metrics_history: List[Dict[str, float]] = []
        self.entry_times: Dict[str, float] = {}

    def collect_step(self, step_number: int) -> None:
        """
        Recolecta métricas en el paso actual de la simulación.
        
        Args:
            step_number (int): El número de paso (timestep) actual.
        """
        try:
            current_time = traci.simulation.getTime()
            vehicle_ids = traci.vehicle.getIDList()
            num_vehicles = len(vehicle_ids)
            
            # Registrar tiempos de entrada para calcular tiempo de viaje
            for v in vehicle_ids:
                if v not in self.entry_times:
                    self.entry_times[v] = current_time

            if num_vehicles > 0:
                wait_times = [traci.vehicle.getWaitingTime(v) for v in vehicle_ids]
                avg_waiting_time = sum(wait_times) / num_vehicles
                
                speeds = [traci.vehicle.getSpeed(v) for v in vehicle_ids]
                avg_speed = sum(speeds) / num_vehicles
                
                total_halted_vehicles = sum(1 for s in speeds if s < 0.1)
                
                co2 = [traci.vehicle.getCO2Emission(v) for v in vehicle_ids]
                total_co2_emissions = sum(co2)
            else:
                avg_waiting_time = 0.0
                avg_speed = 0.0
                total_halted_vehicles = 0
                total_co2_emissions = 0.0
            
            # Throughput y tiempos de viaje (vehículos que llegaron a destino en este paso)
            throughput = traci.simulation.getArrivedNumber()
            arrived_ids = traci.simulation.getArrivedIDList()
            
            travel_times = []
            for v in arrived_ids:
                if v in self.entry_times:
                    travel_times.append(current_time - self.entry_times[v])
                    del self.entry_times[v]
            
            avg_travel_time = sum(travel_times) / len(travel_times) if travel_times else 0.0
            
            # Cola en carriles (lanes monitorizados)
            lane_ids = traci.lane.getIDList()
            if lane_ids:
                queue_lengths = [traci.lane.getLastStepHaltingNumber(lane) for lane in lane_ids]
                avg_queue_length = sum(queue_lengths) / len(lane_ids)
            else:
                avg_queue_length = 0.0

            step_metrics = {
                'step': float(step_number),
                'avg_waiting_time': avg_waiting_time,
                'total_halted_vehicles': float(total_halted_vehicles),
                'avg_speed': avg_speed,
                'throughput': float(throughput),
                'avg_queue_length': avg_queue_length,
                'total_co2_emissions': total_co2_emissions,
                'avg_travel_time': avg_travel_time
            }
            
            self.metrics_history.append(step_metrics)
            
        except Exception as e:
            logging.error(f"❌ Error al recolectar métricas en el paso {step_number}: {e}")

    def get_summary(self) -> Dict[str, float]:
        """
        Retorna un diccionario con las métricas agregadas de todo el episodio.
        """
        if not self.metrics_history:
            return {}
            
        df = pd.DataFrame(self.metrics_history)
        
        # Filtramos avg_travel_time para promediar solo pasos donde hubo llegadas
        travel_time_df = df[df['throughput'] > 0]
        mean_travel_time = float(travel_time_df['avg_travel_time'].mean()) if not travel_time_df.empty else 0.0
        
        summary = {
            'mean_avg_waiting_time': float(df['avg_waiting_time'].mean()),
            'mean_halted_vehicles': float(df['total_halted_vehicles'].mean()),
            'mean_speed': float(df['avg_speed'].mean()),
            'total_throughput': float(df['throughput'].sum()),
            'mean_queue_length': float(df['avg_queue_length'].mean()),
            'total_co2_emissions': float(df['total_co2_emissions'].sum()),
            'mean_avg_travel_time': mean_travel_time
        }
        
        return summary

    def export_csv(self, filepath: str) -> None:
        """Exporta las métricas por paso a un archivo CSV."""
        if not self.metrics_history:
            logging.warning("⚠️ No hay métricas para exportar.")
            return
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(filepath, index=False)
        logging.info(f"💾 Métricas exportadas a: {filepath}")

    def export_summary_csv(self, filepath: str) -> None:
        """Exporta el resumen del episodio a CSV, añadiendo al archivo si ya existe."""
        summary = self.get_summary()
        if not summary:
            return
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame([summary])
        
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, mode='w', header=True, index=False)
        logging.info(f"📊 Resumen añadido a: {filepath}")
