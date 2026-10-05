import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class FRAPExtractor(BaseFeaturesExtractor):
    """Feature Extractor basado en competencia entre fases (FRAP, Zheng et al. 2019).
    
    Modela pares de fases en conflicto y aprende cuál debe tener prioridad
    basándose en la demanda actual. Invariante a rotaciones e inversiones.
    
    Arquitectura:
    1. Phase Demand Encoder: Codifica la demanda de tráfico asociada a cada fase
    2. Phase Competition Module: Compara pares de fases en competencia
    3. Aggregation: Combina las representaciones de competencia
    
    Referencia: "Learning Phase Competition for Traffic Signal Control" (CIKM 2019)
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64,
                 max_phases: int = 4, hidden_dim: int = 32):
        super(FRAPExtractor, self).__init__(observation_space, features_dim)
        
        self.max_phases = max_phases
        obs_dim = observation_space.shape[0]
        
        # Encoder compartido para la demanda por fase (invarianza estructural)
        self.demand_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Módulo de competencia entre pares de fases
        # Recibe la concatenación de dos embeddings de fase
        self.competition_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Número de pares de fases en competencia: C(max_phases, 2)
        num_pairs = max_phases * (max_phases - 1) // 2
        
        # Aggregación final
        self.aggregator = nn.Sequential(
            nn.Linear(num_pairs * (hidden_dim // 2), features_dim),
            nn.ReLU()
        )
        
        # Phase embeddings (invarianza por rotación)
        self.phase_embeddings = nn.Embedding(max_phases, hidden_dim)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # 1. Codificar la demanda global del estado
        demand_encoding = self.demand_encoder(observations)  # (B, hidden_dim)
        
        # 2. Generar representación por fase combinando demanda + embedding de fase
        phase_reprs = []
        for p in range(self.max_phases):
            phase_idx = torch.full((batch_size,), p, dtype=torch.long, 
                                   device=observations.device)
            phase_emb = self.phase_embeddings(phase_idx)  # (B, hidden_dim)
            phase_repr = demand_encoding + phase_emb  # Residual connection
            phase_reprs.append(phase_repr)
        
        # 3. Competencia entre todos los pares de fases
        competition_outputs = []
        for i in range(self.max_phases):
            for j in range(i + 1, self.max_phases):
                pair = torch.cat([phase_reprs[i], phase_reprs[j]], dim=-1)
                comp = self.competition_net(pair)  # (B, hidden_dim//2)
                competition_outputs.append(comp)
        
        # 4. Concatenar y agregar
        all_competitions = torch.cat(competition_outputs, dim=-1)
        features = self.aggregator(all_competitions)  # (B, features_dim)
        
        return features
