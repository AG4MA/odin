# 🏛️ Architecture Decision Record: Mamba vs RWKV

## Executive Summary

**DECISIONE: RWKV-v6 (Eagle Architecture)**

Per un modello da 100M parametri destinato a browser/WASM, RWKV vince per:
- Implementazione più semplice (no selective scan complesso)
- Già testato a scale piccole (RWKV-4 169M funziona bene)
- Kernels più facili da portare a WASM
- Community attiva con implementazioni reference

---

## Analisi Comparativa

### 1. Complessità Computazionale

| Aspetto | Transformer | Mamba | RWKV |
|---------|-------------|-------|------|
| Attention | O(N²) | O(N) | O(N) |
| Memory per token | O(N) | O(1) | O(1) |
| Training | Parallelizzabile | Semi-parallelo | Semi-parallelo |
| Inference | Lenta (KV-cache) | Veloce | Veloce |

### 2. Mamba - Pro e Contro

**Pro:**
- State-of-the-art su benchmark
- Selective state space (adattivo)
- Ottima performance su sequenze lunghe

**Contro:**
- Selective scan richiede kernel CUDA custom
- Difficile da implementare in WASM puro
- Meno testato a scale piccole (<1B)
- Complessità implementativa alta

**Equazione Core Mamba:**
```
h(t) = Ā·h(t-1) + B̄·x(t)
y(t) = C·h(t)

dove Ā, B̄ sono discretizzati da parametri continui (Δ, A, B)
e Δ, B, C sono input-dependent (selective)
```

### 3. RWKV - Pro e Contro

**Pro:**
- Implementazione semplice (puro Python possibile)
- Testato da 169M a 14B parametri
- Community open-source attiva
- Facilmente portabile a WASM
- Time-mixing elegante e stabile

**Contro:**
- Leggermente inferiore a Mamba sui benchmark
- Meno "elegante" matematicamente
- Training richiede attenzione alla stabilità

**Equazione Core RWKV (Time-Mixing v6):**
```
r(t) = σ(x(t) @ Wr)           # Receptance gate
k(t) = x(t) @ Wk              # Key
v(t) = x(t) @ Wv              # Value
w = exp(-exp(decay))          # Time decay

wkv(t) = Σ(i<t) w^(t-i-1) · exp(k(i)) · v(i)
        ─────────────────────────────────────
         Σ(i<t) w^(t-i-1) · exp(k(i))

output = r(t) ⊙ wkv(t)        # Gated output
```

---

## Decisione Finale: RWKV-v6

### Configurazione per 100M Parametri

```yaml
model_config:
  name: "ODIN-100M"
  architecture: "RWKV-v6"
  
  # Dimensioni
  vocab_size: 32768
  embedding_dim: 768
  num_layers: 12
  hidden_dim: 768          # FFN intermedio = 768 * 3.5 ≈ 2688
  
  # State
  state_size: 64           # Per head state size
  num_heads: 12            # 768 / 64 = 12 heads
  
  # Parametri calcolati
  embedding_params: 25.2M  # 32768 * 768
  layer_params: 6.2M       # Per layer
  total_params: ~100M      # 25.2M + 12*6.2M + overhead
```

### Breakdown Parametri per Layer

```
Per ogni layer RWKV:
├── Time-Mixing Block
│   ├── Wr, Wk, Wv, Wo: 768 × 768 × 4 = 2.36M
│   ├── time_decay: 768 = 0.001M
│   └── time_first: 768 = 0.001M
├── Channel-Mixing Block
│   ├── Wk: 768 × 2688 = 2.06M
│   ├── Wv: 2688 × 768 = 2.06M
│   └── Wr: 768 × 768 = 0.59M
└── LayerNorm × 2: 768 × 2 × 2 = 0.003M

Totale per layer: ~6.2M parametri
```

---

## Implicazioni per Altri Ruoli

### → Data Chef
- Sequence length: fino a 4096 token
- Vocab: 32K (BPE tokenizer)
- Il modello ha capacità limitata: focus su reasoning step-by-step

### → Builder  
- Export: ONNX supporta tutte le operazioni
- Nessun kernel custom richiesto
- State size: 12 × 768 = 9216 float per layer (OK per browser)

### → Optimizer
- Operazioni critiche: matmul, exp, sigmoid, element-wise
- Nessun selective scan (più facile di Mamba)
- Quantizzazione: LayerNorm sensibile, attenzione a INT8

---

## Rischi e Mitigazioni

| Rischio | Probabilità | Mitigazione |
|---------|-------------|-------------|
| Performance insufficiente | Media | Fallback a Mamba se necessario |
| Instabilità training | Bassa | Warm-up lungo, gradient clipping |
| WASM troppo lento | Media | WebGPU come piano B |

---

## Riferimenti

1. RWKV Paper: "RWKV: Reinventing RNNs for the Transformer Era"
2. Mamba Paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
3. RWKV-v6 (Eagle): https://github.com/BlinkDL/RWKV-LM
4. RWKV Community: https://wiki.rwkv.com

---

**Approvato:** ✅ Architetto Matematico
**Data:** 2026-01-07
**Versione:** 1.0
