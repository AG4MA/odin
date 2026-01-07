# 🔄 ODIN Phase 1 - Piano Iterativo Micro-Task

## Obiettivo
100M RWKV model → Browser WASM → Reasoning su Math/Code

---

## TASK 1: Setup & Foundations (Tutti in parallelo)

| Ruolo | Micro-Task 1.1 |
|-------|----------------|
| 🏛️ Architetto | Definire config.yaml con dimensioni esatte |
| 🧪 Data Chef | Creare struttura cartelle generatori |
| 🌐 Builder | Setup progetto Python + requirements.txt |
| ⚙️ Optimizer | Creare benchmark baseline (matmul puro) |

---

## TASK 2: Core Implementation

| Ruolo | Micro-Task 2.1 |
|-------|----------------|
| 🏛️ Architetto | Scrivere classe RWKV TimeMixing |
| 🧪 Data Chef | Generatore aritmetica base (+, -, *, /) |
| 🌐 Builder | Script training loop base |
| ⚙️ Optimizer | Kernel WASM per matmul (Rust) |

---

## TASK 3: Expansion

| Ruolo | Micro-Task 3.1 |
|-------|----------------|
| 🏛️ Architetto | Scrivere classe RWKV ChannelMixing |
| 🧪 Data Chef | Generatore algebra (equazioni lineari) |
| 🌐 Builder | Data loader per dataset sintetico |
| ⚙️ Optimizer | Kernel WASM per exp/sigmoid |

---

## TASK 4: Integration

| Ruolo | Micro-Task 4.1 |
|-------|----------------|
| 🏛️ Architetto | Assemblare modello completo + test |
| 🧪 Data Chef | Generatore code (Python base) |
| 🌐 Builder | Export ONNX pipeline |
| ⚙️ Optimizer | Quantizzazione INT8 |

---

## TASK 5: Browser Runtime

| Ruolo | Micro-Task 5.1 |
|-------|----------------|
| 🏛️ Architetto | Validazione numerica WASM vs PyTorch |
| 🧪 Data Chef | Dataset finale packaged |
| 🌐 Builder | WASM runtime + JS API |
| ⚙️ Optimizer | Ottimizzazione memoria browser |

---

## TASK 6: Demo & Polish

| Ruolo | Micro-Task 6.1 |
|-------|----------------|
| 🏛️ Architetto | Benchmark reasoning accuracy |
| 🧪 Data Chef | Test set per evaluation |
| 🌐 Builder | Web UI demo |
| ⚙️ Optimizer | Performance tuning finale |

---

## Stato Attuale

✅ = Completato | 🔄 = In corso | ⬜ = Da fare

| Task | Architetto | Data Chef | Builder | Optimizer |
|------|------------|-----------|---------|-----------|
| 1.1  | ✅ | ✅ | ✅ | ✅ |
| 2.1  | ✅ | ✅ | ✅ | ✅ |
| 3.1  | ✅ | ✅ | ✅ | ✅ |
| 4.1  | ✅ | ✅ | ✅ | ✅ |
| 5.1  | ✅ | ✅ | ✅ | ✅ |
| 6.1  | ✅ | ✅ | ✅ | ✅ |

---

**🎉 PHASE 1 COMPLETATA!**

## Files Creati

### 🏛️ Architetto Matematico
- `config.yaml` - Configurazione modello
- `src/time_mixing.py` - RWKV Time-Mixing block
- `src/channel_mixing.py` - RWKV Channel-Mixing block
- `src/odin_model.py` - Modello completo 100M
- `tests/validate_wasm.py` - Validazione numerica
- `benchmarks/reasoning_benchmark.py` - Benchmark reasoning

### 🧪 Data Chef
- `generators/math/arithmetic.py` - Generatore aritmetica
- `generators/math/algebra.py` - Generatore algebra
- `generators/code/python_basic.py` - Generatore code
- `build_dataset.py` - Builder dataset completo
- `evaluation/generate_testset.py` - Test set evaluation

### 🌐 Builder
- `requirements.txt` - Dipendenze Python
- `src/train.py` - Training loop
- `src/dataloader.py` - Data loader
- `src/export_onnx.py` - Export ONNX
- `browser/src/runtime.ts` - WASM runtime JS
- `browser/demo/index.html` - Demo web UI

### ⚙️ Optimizer
- `benchmarks/baseline_matmul.py` - Benchmark baseline
- `wasm_kernels/src/matmul.rs` - Kernel matmul Rust
- `wasm_kernels/src/activations.rs` - Kernel activations
- `quantization/int8_quantize.py` - Quantizzazione INT8
- `memory/browser_memory.py` - Ottimizzazione memoria
- `tuning/performance_tuner.py` - Performance tuning
