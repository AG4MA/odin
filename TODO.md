# 🏛️ Project ODIN: The Linux of AI

> *"Democratizing Intelligence, One Parameter at a Time"*

## 🎯 Mission

Create a decentralized, sovereign, and accessible AI ecosystem that runs on consumer hardware without requiring billion-dollar datacenters.

## 📋 Phase 1: Proof of Concept — ✅ SCAFFOLD COMPLETE 

**Objective:** Build a 100 Million parameter model based on RWKV-v6 architecture, trained exclusively on synthetic math & coding data, running in the browser via WebAssembly.

> "If this toy can reason, the theory works."

### 🏆 Architecture Decision: **RWKV-v6** (chosen over Mamba)
- Simpler WASM portability (no selective scan complexity)
- O(N) complexity like Mamba
- Active community & proven at scale
- Better suited for 100M parameter range

---

## 🏗️ Project Structure (Implemented)

```
odin/
├── 01_architect_mathematician/    # 🏛️ The Visionary
│   ├── config.yaml               # Model hyperparameters (768 dim, 12 layers)
│   ├── src/
│   │   ├── time_mixing.py        # RWKV Time-Mixing with WKV computation
│   │   ├── channel_mixing.py     # RWKV Channel-Mixing with squared ReLU
│   │   └── odin_model.py         # Complete 100M model assembly
│   ├── tests/validate_wasm.py    # Numerical validation suite
│   └── benchmarks/reasoning_benchmark.py
│
├── 02_data_chef/                  # 🧪 The Alchemist  
│   ├── generators/
│   │   ├── math/arithmetic.py    # +, -, *, / with step-by-step reasoning
│   │   ├── math/algebra.py       # Linear/quadratic equations (SymPy)
│   │   └── code/python_basic.py  # Function implementation problems
│   ├── build_dataset.py          # Full dataset builder (1M target)
│   └── evaluation/generate_testset.py
│
├── 03_distributed_builder/        # 🌐 The Swarm Architect
│   ├── requirements.txt          # torch, onnx, sympy, tokenizers, wandb
│   ├── src/
│   │   ├── train.py              # Training loop with gradient clipping
│   │   ├── dataloader.py         # SyntheticMathDataset class
│   │   └── export_onnx.py        # PyTorch→ONNX export pipeline
│   └── browser/
│       ├── src/runtime.ts        # OdinRuntime with streaming API
│       └── demo/index.html       # Full demo UI with chat interface
│
├── 04_lowlevel_optimizer/         # ⚙️ The Surgeon
│   ├── benchmarks/baseline_matmul.py
│   ├── wasm_kernels/src/
│   │   ├── matmul.rs             # Tiled WASM matmul kernel
│   │   └── activations.rs        # sigmoid, relu, gelu, softmax
│   ├── quantization/int8_quantize.py  # INT8 PTQ pipeline
│   ├── memory/browser_memory.py  # <400MB memory optimizer
│   └── tuning/performance_tuner.py
│
├── ITERATION_PLAN.md             # Task tracking (24/24 complete)
├── pyrightconfig.json            # Pylance configuration
└── TODO.md                       # This file
```

---

## 📊 Phase 1 Progress

| Task Block | Architect | Data Chef | Builder | Optimizer | Status |
|------------|-----------|-----------|---------|-----------|--------|
| 1.1 Setup  | ✅ | ✅ | ✅ | ✅ | Complete |
| 2.1 Core   | ✅ | ✅ | ✅ | ✅ | Complete |
| 3.1 Expand | ✅ | ✅ | ✅ | ✅ | Complete |
| 4.1 Integrate | ✅ | ✅ | ✅ | ✅ | Complete |
| 5.1 Browser | ✅ | ✅ | ✅ | ✅ | Complete |
| 6.1 Demo   | ✅ | ✅ | ✅ | ✅ | Complete |

**Total: 24/24 micro-tasks complete** 🎉

---

## 🔧 Model Configuration (100M Parameters)

| Parameter | Value |
|-----------|-------|
| `vocab_size` | 32,768 |
| `embedding_dim` | 768 |
| `num_layers` | 12 |
| `num_heads` | 12 |
| `head_dim` | 64 |
| `ffn_dim` | 2,688 |
| `max_seq_len` | 4,096 |
| **Total Params** | ~100M |

---

## 🚀 Next Steps (Execution Phase)

```bash
# 1. Install dependencies
pip install -r 03_distributed_builder/requirements.txt

# 2. Generate synthetic dataset (1M examples)
python 02_data_chef/build_dataset.py

# 3. Train the model
python 03_distributed_builder/src/train.py

# 4. Export to ONNX
python 03_distributed_builder/src/export_onnx.py

# 5. Quantize to INT8
python 04_lowlevel_optimizer/quantization/int8_quantize.py

# 6. Compile WASM kernels
cd 04_lowlevel_optimizer/wasm_kernels
cargo build --target wasm32-unknown-unknown --release

# 7. Launch demo
# Open 03_distributed_builder/browser/demo/index.html
```

---

## 🎯 Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Model Size | 100M ± 5M parameters | ✅ Configured |
| Download Size | < 200MB (INT8) | 🔄 Ready to quantize |
| Inference Speed | > 10 tokens/sec (laptop) | ⬜ Pending training |
| Memory Usage | < 400MB browser | ✅ Optimizer ready |
| Math Accuracy | > 70% on GSM8K-style | ⬜ Pending evaluation |
| Code Accuracy | > 40% on HumanEval-style | ⬜ Pending evaluation |
| Browser Support | Chrome, Firefox, Safari, Edge | ✅ Runtime ready |

---

## 🧬 Core Philosophy

### Why Not Transformers?
- O(N²) attention is expensive and memory-hungry
- Mamba/RWKV achieve O(N) complexity
- Linear scaling = runs on consumer hardware

### Why Synthetic Data?
- No copyright issues
- No bias from web scraping  
- 100% verified correctness
- Focused on reasoning, not memorization

### Why Browser?
- Zero installation friction
- True decentralization (no server needed)
- Privacy by default (runs locally)
- Proves it works on weak hardware

---

## 🔮 Future Phases

- **Phase 2:** Distributed training via P2P swarm
- **Phase 3:** Scale to 1B+ parameters
- **Phase 4:** Multi-modal (vision, audio)
- **Phase 5:** Federated learning network

---

## 📜 Manifesto

```
We believe:
  - Intelligence should not be controlled by few
  - Privacy is a fundamental right
  - Knowledge should be free and verifiable
  - Small, efficient models beat bloated giants
  - The edge is the future, not the cloud

We reject:
  - Dependency on proprietary hardware
  - Black-box AI systems
  - Surveillance capitalism
  - Artificial scarcity of intelligence
```

---

## 🤝 Contributing

See `ITERATION_PLAN.md` for detailed task breakdown and completion status.

**Phase 1 scaffold is complete.** Ready for:
1. Dataset generation
2. Model training (GPU recommended)
3. ONNX export & quantization
4. Browser deployment

**The revolution will be decentralized.** 🔥

---

## 📅 Timeline

| Milestone | Status |
|-----------|--------|
| Architecture Design | ✅ Complete (RWKV-v6) |
| Code Scaffold | ✅ Complete (24 files) |
| Data Generators | ✅ Complete |
| Training Pipeline | ✅ Complete |
| WASM Kernels | ✅ Complete |
| Dataset Generation | ⬜ Ready to run |
| Model Training | ⬜ Needs GPU |
| Browser Demo | ⬜ Needs trained model |

---

*Project ODIN - Named after the Norse god who sacrificed an eye for wisdom.
We sacrifice centralization for freedom.*
