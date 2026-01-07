"""
ODIN-100M Complete Model
========================
Assemblaggio finale del modello RWKV-v6.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class OdinConfig:
    """Configurazione ODIN-100M"""
    vocab_size: int = 32768
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    ffn_dim: int = 2688
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    max_seq_len: int = 4096


class LayerNorm(nn.Module):
    """LayerNorm senza bias"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        return F.layer_norm(x, (x.size(-1),), self.weight, None, self.eps)


class RWKV_TimeMixing(nn.Module):
    """Time-Mixing block (attenzione O(N))"""
    
    def __init__(self, config: OdinConfig, layer_id: int):
        super().__init__()
        dim = config.embedding_dim
        
        self.time_mix_r = nn.Parameter(torch.zeros(dim))
        self.time_mix_k = nn.Parameter(torch.zeros(dim))
        self.time_mix_v = nn.Parameter(torch.zeros(dim))
        self.time_decay = nn.Parameter(torch.zeros(config.num_heads, config.head_dim))
        self.time_first = nn.Parameter(torch.zeros(config.num_heads, config.head_dim))
        
        self.W_r = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        self.group_norm = nn.GroupNorm(config.num_heads, dim)
        
        self._init(layer_id, config.num_layers)
    
    def _init(self, layer_id, num_layers):
        scale = layer_id / num_layers
        with torch.no_grad():
            ratio = 0.5 * (1 - scale)
            self.time_mix_r.fill_(ratio)
            self.time_mix_k.fill_(ratio)
            self.time_mix_v.fill_(ratio)
            self.time_decay.fill_(-5 + 8 * (scale ** 0.7))
            self.time_first.fill_(0.3 - scale * 0.3)
    
    def forward(self, x, state=None):
        B, T, D = x.shape
        H, K = self.time_decay.shape
        
        if state is None:
            state = torch.zeros(B, H, K, 3, device=x.device, dtype=x.dtype)
        
        x_prev = torch.cat([state[:, 0, :, 2:3].transpose(1,2), x[:, :-1]], dim=1)
        
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1 - self.time_mix_v)
        
        r = torch.sigmoid(self.W_r(xr)).view(B, T, H, K)
        k = self.W_k(xk).view(B, T, H, K)
        v = self.W_v(xv).view(B, T, H, K)
        
        w = torch.exp(-torch.exp(self.time_decay))
        u = self.time_first
        
        # WKV sequential
        num = state[:, :, :, 0].clone()
        den = state[:, :, :, 1].clone()
        outputs = []
        
        for t in range(T):
            ek = torch.exp(k[:, t])
            wkv = (num + torch.exp(u) * ek * v[:, t]) / (den + torch.exp(u) * ek + 1e-9)
            outputs.append(wkv)
            num = num * w + ek * v[:, t]
            den = den * w + ek
        
        out = torch.stack(outputs, dim=1)
        out = (r * out).view(B, T, D)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        out = self.W_o(out)
        
        new_state = state.clone()
        new_state[:, :, :, 0] = num
        new_state[:, :, :, 1] = den
        new_state[:, 0, :, 2] = x[:, -1]
        
        return out, new_state


class RWKV_ChannelMixing(nn.Module):
    """Channel-Mixing block (FFN con gating)"""
    
    def __init__(self, config: OdinConfig, layer_id: int):
        super().__init__()
        dim = config.embedding_dim
        
        self.time_mix_k = nn.Parameter(torch.zeros(dim))
        self.time_mix_r = nn.Parameter(torch.zeros(dim))
        
        self.W_k = nn.Linear(dim, config.ffn_dim, bias=False)
        self.W_v = nn.Linear(config.ffn_dim, dim, bias=False)
        self.W_r = nn.Linear(dim, dim, bias=False)
        
        self._init(layer_id, config.num_layers)
    
    def _init(self, layer_id, num_layers):
        scale = layer_id / num_layers
        ratio = 0.5 * (1 - scale)
        with torch.no_grad():
            self.time_mix_k.fill_(ratio)
            self.time_mix_r.fill_(ratio)
    
    def forward(self, x, prev_x=None):
        B, T, D = x.shape
        
        if prev_x is None:
            prev_x = torch.zeros(B, 1, D, device=x.device, dtype=x.dtype)
        
        x_prev = torch.cat([prev_x, x[:, :-1]], dim=1)
        
        xk = x * self.time_mix_k + x_prev * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + x_prev * (1 - self.time_mix_r)
        
        k = torch.square(torch.relu(self.W_k(xk)))
        v = self.W_v(k)
        r = torch.sigmoid(self.W_r(xr))
        
        return r * v, x[:, -1:]


class RWKVBlock(nn.Module):
    """Singolo blocco RWKV"""
    
    def __init__(self, config: OdinConfig, layer_id: int):
        super().__init__()
        self.ln1 = LayerNorm(config.embedding_dim)
        self.ln2 = LayerNorm(config.embedding_dim)
        self.time_mix = RWKV_TimeMixing(config, layer_id)
        self.channel_mix = RWKV_ChannelMixing(config, layer_id)
    
    def forward(self, x, tm_state=None, cm_prev=None):
        tm_out, new_tm_state = self.time_mix(self.ln1(x), tm_state)
        x = x + tm_out
        
        cm_out, new_cm_prev = self.channel_mix(self.ln2(x), cm_prev)
        x = x + cm_out
        
        return x, new_tm_state, new_cm_prev


class ODIN(nn.Module):
    """ODIN-100M Language Model"""
    
    def __init__(self, config: OdinConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.ln_in = LayerNorm(config.embedding_dim)
        self.blocks = nn.ModuleList([RWKVBlock(config, i) for i in range(config.num_layers)])
        self.ln_out = LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.head.weight = self.embedding.weight  # Weight tying
        
        self.apply(self._init_weights)
        print(f"ODIN: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, input_ids, states=None):
        B, T = input_ids.shape
        x = self.ln_in(self.embedding(input_ids))
        
        if states is None:
            states = [(None, None)] * self.config.num_layers
        
        new_states = []
        for i, block in enumerate(self.blocks):
            x, tm_s, cm_s = block(x, states[i][0], states[i][1])
            new_states.append((tm_s, cm_s))
        
        return self.head(self.ln_out(x)), new_states
    
    @torch.no_grad()
    def generate(self, input_ids, max_tokens=100, temperature=0.8, top_k=40):
        self.eval()
        states = None
        
        for _ in range(max_tokens):
            logits, states = self(input_ids[:, -1:], states)
            logits = logits[:, -1] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# Test
if __name__ == "__main__":
    config = OdinConfig()
    model = ODIN(config)
    
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits, _ = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {logits.shape}")
    print(f"FP32 size: {sum(p.numel()*4 for p in model.parameters())/1e6:.1f} MB")
