"""
ODIN-100M: RWKV-v6 Implementation
=================================
Modello 100M parametri basato su architettura RWKV-v6 (Eagle)
Ottimizzato per export WASM e inference su browser.

Author: Architect Mathematician
License: Apache 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class OdinConfig:
    """Configurazione per ODIN-100M"""
    vocab_size: int = 32768
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64  # embedding_dim // num_heads
    ffn_dim: int = 2688  # embedding_dim * 3.5
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    max_seq_len: int = 4096
    
    def __post_init__(self):
        assert self.embedding_dim == self.num_heads * self.head_dim
        self.state_size = self.num_heads * self.head_dim  # = embedding_dim


class LayerNorm(nn.Module):
    """LayerNorm senza bias (più stabile per RWKV)"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.size(-1),), self.weight, None, self.eps)


class RWKV_TimeMixing(nn.Module):
    """
    RWKV Time-Mixing Block (WKV Attention Replacement)
    
    Complessità: O(N) invece di O(N²)
    
    Equazioni:
        r = σ(x @ Wr)
        k = x @ Wk  
        v = x @ Wv
        wkv = Σ w^(t-i) * exp(k_i) * v_i / Σ w^(t-i) * exp(k_i)
        out = r ⊙ (wkv @ Wo)
    """
    
    def __init__(self, config: OdinConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        dim = config.embedding_dim
        
        # Mixing coefficients (learnable interpolation with previous token)
        self.time_mix_r = nn.Parameter(torch.zeros(dim))
        self.time_mix_k = nn.Parameter(torch.zeros(dim))
        self.time_mix_v = nn.Parameter(torch.zeros(dim))
        
        # Time decay (w in equations) - per head
        self.time_decay = nn.Parameter(torch.zeros(config.num_heads, config.head_dim))
        self.time_first = nn.Parameter(torch.zeros(config.num_heads, config.head_dim))
        
        # Projections
        self.W_r = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        
        # Group normalization for stability
        self.group_norm = nn.GroupNorm(config.num_heads, dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Inizializzazione specifica per RWKV"""
        layer_scale = self.layer_id / self.config.num_layers
        
        # Time mixing: più recency bias nei layer iniziali
        with torch.no_grad():
            ratio = 0.5 * (1 - layer_scale)
            self.time_mix_r.fill_(ratio)
            self.time_mix_k.fill_(ratio)
            self.time_mix_v.fill_(ratio)
            
            # Time decay: layer più profondi = decay più lento
            decay = -5 + 8 * (layer_scale ** 0.7)
            self.time_decay.fill_(decay)
            self.time_first.fill_(0.3 - layer_scale * 0.3)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, dim)
            state: (batch, num_heads, head_dim, 3) - [num, den, prev_x]
            
        Returns:
            output: (batch, seq_len, dim)
            new_state: updated state
        """
        B, T, D = x.shape
        H = self.config.num_heads
        K = self.config.head_dim
        
        # Initialize state if needed
        if state is None:
            state = torch.zeros(B, H, K, 3, device=x.device, dtype=x.dtype)
        
        # Shift for time mixing (token mixing with previous)
        x_prev = torch.cat([state[:, 0, :, 2:3].transpose(1, 2), x[:, :-1, :]], dim=1)
        
        # Time-mix interpolation
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        
        # Compute r, k, v
        r = torch.sigmoid(self.W_r(xr))  # Gate
        k = self.W_k(xk)
        v = self.W_v(xv)
        
        # Reshape for multi-head
        r = r.view(B, T, H, K)
        k = k.view(B, T, H, K)
        v = v.view(B, T, H, K)
        
        # WKV computation (linear recurrence)
        w = torch.exp(-torch.exp(self.time_decay))  # (H, K)
        u = self.time_first  # (H, K)
        
        # Efficient parallel scan for training
        output = self._parallel_wkv(k, v, w, u, state)
        
        # Apply gate and project
        output = (r * output).view(B, T, D)
        output = self.group_norm(output.transpose(1, 2)).transpose(1, 2)
        output = self.W_o(output)
        
        # Update state
        new_state = state.clone()
        new_state[:, :, :, 0] = state[:, :, :, 0] * w + torch.exp(k[:, -1])  # num
        new_state[:, :, :, 1] = state[:, :, :, 1] * w + torch.exp(k[:, -1]) * v[:, -1]  # den
        new_state[:, 0, :, 2] = x[:, -1]  # prev_x
        
        return output, new_state
    
    def _parallel_wkv(
        self, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        w: torch.Tensor, 
        u: torch.Tensor,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel WKV computation for training efficiency.
        
        Per inference token-by-token, usare _sequential_wkv.
        """
        B, T, H, K = k.shape
        
        # Expand w per time steps
        w_powers = w.unsqueeze(0).unsqueeze(0)  # (1, 1, H, K)
        
        # Build decay matrix
        positions = torch.arange(T, device=k.device).float()
        decay_matrix = w_powers ** positions.view(T, 1, 1, 1)  # (T, 1, H, K)
        
        # Compute attention-like scores with linear complexity trick
        exp_k = torch.exp(k - k.max(dim=1, keepdim=True).values)  # Numerical stability
        
        # Weighted sum using cumsum (O(N) complexity)
        weighted_v = exp_k.unsqueeze(-1) * v.unsqueeze(-2)
        
        # Cumulative sum with decay (simplified parallel scan)
        # In produzione usare CUDA kernel per efficienza
        num = torch.zeros(B, H, K, device=k.device, dtype=k.dtype)
        den = torch.zeros(B, H, K, device=k.device, dtype=k.dtype)
        
        outputs = []
        for t in range(T):
            ek = torch.exp(k[:, t])  # (B, H, K)
            
            # WKV formula
            if t == 0:
                wkv = (state[:, :, :, 0] + torch.exp(u) * ek * v[:, t]) / \
                      (state[:, :, :, 1] + torch.exp(u) * ek + 1e-9)
            else:
                wkv = (num + torch.exp(u) * ek * v[:, t]) / \
                      (den + torch.exp(u) * ek + 1e-9)
            
            outputs.append(wkv)
            
            # Update accumulators
            num = num * w + ek * v[:, t]
            den = den * w + ek
        
        return torch.stack(outputs, dim=1)  # (B, T, H, K)


class RWKV_ChannelMixing(nn.Module):
    """
    RWKV Channel-Mixing Block (FFN Replacement)
    
    Più efficiente di FFN standard con gating.
    """
    
    def __init__(self, config: OdinConfig, layer_id: int):
        super().__init__()
        self.config = config
        dim = config.embedding_dim
        
        # Mixing coefficient
        self.time_mix_k = nn.Parameter(torch.zeros(dim))
        self.time_mix_r = nn.Parameter(torch.zeros(dim))
        
        # Projections
        self.W_k = nn.Linear(dim, config.ffn_dim, bias=False)
        self.W_v = nn.Linear(config.ffn_dim, dim, bias=False)
        self.W_r = nn.Linear(dim, dim, bias=False)
        
        self._init_weights(layer_id)
    
    def _init_weights(self, layer_id: int):
        layer_scale = layer_id / self.config.num_layers
        ratio = 0.5 * (1 - layer_scale)
        with torch.no_grad():
            self.time_mix_k.fill_(ratio)
            self.time_mix_r.fill_(ratio)
    
    def forward(
        self, 
        x: torch.Tensor, 
        prev_x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, dim)
            prev_x: previous token's x for time mixing
        """
        B, T, D = x.shape
        
        if prev_x is None:
            prev_x = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        
        # Shift
        x_prev = torch.cat([prev_x, x[:, :-1, :]], dim=1)
        
        # Time-mix
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        
        # Channel mixing with squared ReLU (RWKV specific)
        k = self.W_k(xk)
        k = torch.square(torch.relu(k))  # Squared ReLU
        v = self.W_v(k)
        
        r = torch.sigmoid(self.W_r(xr))  # Gate
        
        return r * v, x[:, -1:]


class RWKVBlock(nn.Module):
    """Single RWKV Block = TimeMixing + ChannelMixing"""
    
    def __init__(self, config: OdinConfig, layer_id: int):
        super().__init__()
        self.ln1 = LayerNorm(config.embedding_dim, config.layer_norm_eps)
        self.ln2 = LayerNorm(config.embedding_dim, config.layer_norm_eps)
        self.time_mixing = RWKV_TimeMixing(config, layer_id)
        self.channel_mixing = RWKV_ChannelMixing(config, layer_id)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[torch.Tensor] = None,
        prev_x_channel: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # Time mixing (attention equivalent)
        tm_out, new_state = self.time_mixing(self.ln1(x), state)
        x = x + tm_out
        
        # Channel mixing (FFN equivalent)  
        cm_out, new_prev_x = self.channel_mixing(self.ln2(x), prev_x_channel)
        x = x + cm_out
        
        return x, new_state, new_prev_x


class ODIN(nn.Module):
    """
    ODIN-100M: RWKV-based Language Model
    
    100M parametri, complessità O(N), pronto per WASM.
    """
    
    def __init__(self, config: OdinConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.ln_in = LayerNorm(config.embedding_dim, config.layer_norm_eps)
        
        # RWKV Blocks
        self.blocks = nn.ModuleList([
            RWKVBlock(config, layer_id=i) 
            for i in range(config.num_layers)
        ])
        
        # Output
        self.ln_out = LayerNorm(config.embedding_dim, config.layer_norm_eps)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.embedding.weight
        
        # Initialize
        self.apply(self._init_weights)
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"ODIN-100M initialized: {self.num_params / 1e6:.2f}M parameters")
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        states: Optional[list] = None,
        return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            states: list of per-layer states for incremental inference
            return_states: whether to return updated states
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            states: updated states if return_states=True
        """
        B, T = input_ids.shape
        
        # Embedding
        x = self.embedding(input_ids)
        x = self.ln_in(x)
        
        # Initialize states if needed
        if states is None:
            states = [None] * self.config.num_layers
        
        new_states = []
        prev_x_channels = [None] * self.config.num_layers
        
        # Process through blocks
        for i, block in enumerate(self.blocks):
            x, new_state, new_prev_x = block(x, states[i], prev_x_channels[i])
            new_states.append(new_state)
            prev_x_channels[i] = new_prev_x
        
        # Output
        x = self.ln_out(x)
        logits = self.head(x)
        
        if return_states:
            return logits, new_states
        return logits, None
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive generation with sampling.
        """
        self.eval()
        states = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits, states = self(input_ids, states, return_states=True)
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][:, -1:]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Only keep last token for next iteration (RNN-style)
            input_ids = input_ids[:, -1:]
        
        return input_ids


def count_parameters(model: nn.Module) -> dict:
    """Breakdown dettagliato dei parametri"""
    counts = {}
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]
        if module_name not in counts:
            counts[module_name] = 0
        counts[module_name] += param.numel()
    
    total = sum(counts.values())
    print("\n=== Parameter Breakdown ===")
    for name, count in sorted(counts.items()):
        print(f"{name}: {count / 1e6:.2f}M ({100 * count / total:.1f}%)")
    print(f"Total: {total / 1e6:.2f}M")
    
    return counts


# Quick test
if __name__ == "__main__":
    config = OdinConfig()
    model = ODIN(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, _ = model(input_ids)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Parameter breakdown
    count_parameters(model)
    
    # Memory estimate
    param_bytes = sum(p.numel() * 4 for p in model.parameters())  # FP32
    print(f"\nModel size (FP32): {param_bytes / 1e6:.1f} MB")
    print(f"Model size (FP16): {param_bytes / 2 / 1e6:.1f} MB")
    print(f"Model size (INT8): {param_bytes / 4 / 1e6:.1f} MB")
