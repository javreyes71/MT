from stable_baselines3 import PPO
from src.environment.traffic_env import TrafficSumoEnv
import os

class CentralizedAgent:
    """Agente PPO centralizado que controla todos los semáforos."""
    
    def __init__(self, env: TrafficSumoEnv, config: dict):
        training_config = config.get('training', {})
        paths_config = config.get('paths', {})
        
        self.model = PPO(
            training_config.get('policy', 'MlpPolicy'),
            env,
            learning_rate=training_config.get('learning_rate', 3e-4),
            n_steps=training_config.get('n_steps', 2048),
            batch_size=training_config.get('batch_size', 64),
            gamma=training_config.get('gamma', 0.99),
            verbose=1,
            tensorboard_log=paths_config.get('tensorboard_dir', './logs')
        )
    
    def train(self, total_timesteps: int, callbacks=None):
        """Entrena el modelo centralizado."""
        print(f"🚀 Iniciando entrenamiento centralizado por {total_timesteps} pasos...")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)
        print("✅ Entrenamiento centralizado finalizado.")
    
    def predict(self, obs, deterministic=True):
        """Predice las acciones para el entorno centralizado."""
        return self.model.predict(obs, deterministic=deterministic)
    
    def save(self, path):
        """Guarda el modelo."""
        self.model.save(path)
        print(f"💾 Modelo guardado en {path}")
    
    @classmethod
    def load(cls, path, env):
        """Carga un modelo preentrenado."""
        agent = cls.__new__(cls)
        agent.model = PPO.load(path, env=env)
        print(f"📂 Modelo cargado desde {path}")
        return agent
