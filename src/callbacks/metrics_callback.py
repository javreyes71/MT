import os
from stable_baselines3.common.callbacks import BaseCallback
from src.utils.metrics import MetricsCollector


class MetricsCallback(BaseCallback):
    """Callback para recolectar métricas detalladas de la simulación SUMO.
    
    Guarda datos step-by-step y resúmenes por episodio para comparación
    con el baseline de tiempos fijos.
    """
    
    def __init__(self, results_dir: str = "results", mode: str = "centralized",
                 verbose: int = 0) -> None:
        super(MetricsCallback, self).__init__(verbose)
        self.collector = MetricsCollector()
        self.mode = mode
        self.mode_dir = os.path.join(results_dir, mode)
        os.makedirs(self.mode_dir, exist_ok=True)
        
        self.episode_count = 0
        self.step_in_episode = 0

    def _on_step(self) -> bool:
        """Recolecta métricas en cada paso del entorno."""
        try:
            import libsumo as traci
            step = int(traci.simulation.getTime())
            self.collector.collect_step(step)
            self.step_in_episode += 1
        except Exception:
            pass
            
        # Si el episodio termina (todos los agentes comparten episodio)
        dones = self.locals.get("dones")
        if dones is not None and dones.all():
            self.episode_count += 1
            summary = self.collector.get_summary()
            
            if summary:
                # 1. Loggear en TensorBoard
                for key, value in summary.items():
                    self.logger.record(f"sumo/{key}", value)
                
                # 2. Exportar datos step-by-step de este episodio
                steps_file = os.path.join(
                    self.mode_dir, f"{self.mode}_ep{self.episode_count}_steps.csv"
                )
                self.collector.export_csv(steps_file)
                
                # 3. Exportar resumen acumulado
                summary_file = os.path.join(
                    self.mode_dir, f"{self.mode}_summary.csv"
                )
                self.collector.export_summary_csv(summary_file)
                
                if self.verbose > 0 or self.episode_count % 5 == 0:
                    wt = summary.get('mean_avg_waiting_time', 0)
                    spd = summary.get('mean_speed', 0)
                    halt = summary.get('mean_halted_vehicles', 0)
                    print(f"📊 Ep {self.episode_count}: espera={wt:.1f}s, "
                          f"vel={spd:.2f}m/s, detenidos={halt:.0f}")
            
            # Reset para el siguiente episodio
            self.collector.reset()
            self.step_in_episode = 0
            
        return True
