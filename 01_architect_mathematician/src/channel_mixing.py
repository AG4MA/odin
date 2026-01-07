"""
RWKV ChannelMixing Block
========================
FFN replacement con gating mechanism.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ChannelMixConfig:
    embedding_dim: int = 768
    ffn_dim: int = 2688
    layer_id: int = 0
    num_layers: int = 12


class RWKV_ChannelMixing(nn.Module):
    """
    RWKV Channel-Mixing: sostituisce Feed-Forward Network
    
    Usa squared ReLU e gating per maggiore espressività.
    """
    
    def __init__(self, config: ChannelMixConfig):
        super().__init__()
        self.config = config
        dim = config.embedding_dim
        
        # Time-mix coefficients
        self.time_mix_k = nn.Parameter(torch.zeros(dim))
        self.time_mix_r = nn.Parameter(torch.zeros(dim))
        
        # Proiezioni
        self.W_k = nn.Linear(dim, config.ffn_dim, bias=False)
        self.W_v = nn.Linear(config.ffn_dim, dim, bias=False)
        self.W_r = nn.Linear(dim, dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        layer_scale = self.config.layer_id / max(self.config.num_layers, 1)
        ratio = 0.5 * (1 - layer_scale)
        with torch.no_grad():
            self.time_mix_k.fill_(ratio)
            self.time_mix_r.fill_(ratio)
    
    def forward(self, x: torch.Tensor, prev_x: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, dim)
            prev_x: (batch, 1, dim) ultimo token del blocco precedente
        Returns:
            output: (batch, seq_len, dim)
            last_x: (batch, 1, dim) per prossimo forward
        """
        B, T, D = x.shape
        
        if prev_x is None:
            prev_x = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        
        # Shift temporale
        x_prev = torch.cat([prev_x, x[:, :-1, :]], dim=1)
        
        # Time-mix interpolation
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        
        # FFN con squared ReLU
        k = self.W_k(xk)
        k = torch.square(torch.relu(k))  # Squared ReLU - chiave per RWKV
        v = self.W_v(k)
        
        # Gating
        r = torch.sigmoid(self.W_r(xr))
        output = r * v
        
        return output, x[:, -1:]
