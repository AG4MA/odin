"""
RWKV TimeMixing Block
=====================
Core attention-replacement con complessità O(N).
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class TimeMixConfig:
    embedding_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    layer_id: int = 0
    num_layers: int = 12


class RWKV_TimeMixing(nn.Module):
    """
    RWKV Time-Mixing: sostituisce Self-Attention
    
    Complexity: O(N) vs O(N²) dei Transformer
    Memory: O(1) per token (stato fisso)
    """
    
    def __init__(self, config: TimeMixConfig):
        super().__init__()
        self.config = config
        dim = config.embedding_dim
        
        # Time-mix interpolation (con token precedente)
        self.time_mix_r = nn.Parameter(torch.zeros(dim))
        self.time_mix_k = nn.Parameter(torch.zeros(dim))
        self.time_mix_v = nn.Parameter(torch.zeros(dim))
        
        # Decay temporale per-head
        self.time_decay = nn.Parameter(torch.zeros(config.num_heads, config.head_dim))
        self.time_first = nn.Parameter(torch.zeros(config.num_heads, config.head_dim))
        
        # Proiezioni lineari
        self.W_r = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        
        # Normalizzazione per stabilità
        self.group_norm = nn.GroupNorm(config.num_heads, dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Inizializzazione RWKV-specific"""
        layer_scale = self.config.layer_id / max(self.config.num_layers, 1)
        
        with torch.no_grad():
            # Recency bias maggiore nei primi layer
            ratio = 0.5 * (1 - layer_scale)
            self.time_mix_r.fill_(ratio)
            self.time_mix_k.fill_(ratio)
            self.time_mix_v.fill_(ratio)
            
            # Decay più lento nei layer profondi
            decay = -5 + 8 * (layer_scale ** 0.7)
            self.time_decay.fill_(decay)
            self.time_first.fill_(0.3 - layer_scale * 0.3)
    
    def forward(self, x: torch.Tensor, state: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, dim)
            state: (batch, num_heads, head_dim, 3) per inference incrementale
        Returns:
            output: (batch, seq_len, dim)
            new_state: stato aggiornato
        """
        B, T, D = x.shape
        H = self.config.num_heads
        K = self.config.head_dim
        
        # Inizializza stato se necessario
        if state is None:
            state = torch.zeros(B, H, K, 3, device=x.device, dtype=x.dtype)
        
        # Shift temporale
        x_prev = torch.cat([state[:, 0, :, 2:3].transpose(1, 2), x[:, :-1, :]], dim=1)
        
        # Interpolazione time-mix
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        
        # Calcola r (gate), k, v
        r = torch.sigmoid(self.W_r(xr))
        k = self.W_k(xk)
        v = self.W_v(xv)
        
        # Reshape multi-head
        r = r.view(B, T, H, K)
        k = k.view(B, T, H, K)
        v = v.view(B, T, H, K)
        
        # WKV computation (ricorrenza lineare)
        w = torch.exp(-torch.exp(self.time_decay))
        u = self.time_first
        
        output = self._wkv_sequential(k, v, w, u, state)
        
        # Applica gate e proietta
        output = (r * output).view(B, T, D)
        output = self.group_norm(output.transpose(1, 2)).transpose(1, 2)
        output = self.W_o(output)
        
        # Aggiorna stato
        new_state = state.clone()
        new_state[:, 0, :, 2] = x[:, -1]
        
        return output, new_state
    
    def _wkv_sequential(self, k, v, w, u, state):
        """WKV con loop sequenziale (per chiarezza, ottimizzare dopo)"""
        B, T, H, K = k.shape
        
        num = state[:, :, :, 0].clone()
        den = state[:, :, :, 1].clone()
        
        outputs = []
        for t in range(T):
            ek = torch.exp(k[:, t])
            
            wkv = (num + torch.exp(u) * ek * v[:, t]) / (den + torch.exp(u) * ek + 1e-9)
            outputs.append(wkv)
            
            num = num * w + ek * v[:, t]
            den = den * w + ek
        
        return torch.stack(outputs, dim=1)
