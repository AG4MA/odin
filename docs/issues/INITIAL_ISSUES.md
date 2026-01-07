# Initial Issue Set

Pre-defined issues for project launch. Copy these to GitHub Issues.

---

## Good First Issues (5)

### Issue #1: Add unit tests for arithmetic generator

**Labels:** `type/bug` `difficulty/good-first` `domain/data` `status/ready`

**Description:**
The arithmetic generator in `02_data_chef/generators/math/arithmetic.py` lacks comprehensive unit tests. Add tests covering edge cases.

**Scope:**
- Create `02_data_chef/generators/math/test_arithmetic.py`
- Test all operations: addition, subtraction, multiplication, division
- Test edge cases: zero, negative numbers, large numbers, division by zero handling
- Test output format matches expected schema

**Acceptance Criteria:**
- [ ] All operations have at least 3 test cases each
- [ ] Edge cases documented and tested
- [ ] Tests pass with `pytest`
- [ ] Coverage > 80% for `arithmetic.py`

**Expected Skills:** Python, pytest basics

**Estimated Effort:** 2-4 hours

**Owner Archetype:** Data Chef (B)

---

### Issue #2: Add type hints to dataloader.py

**Labels:** `type/refactor` `difficulty/good-first` `domain/training` `status/ready`

**Description:**
`03_distributed_builder/src/dataloader.py` has incomplete type hints. Add full type annotations.

**Scope:**
- Add type hints to all function signatures
- Add type hints to class attributes
- Ensure Pyright passes with no errors

**Acceptance Criteria:**
- [ ] All functions have parameter and return type hints
- [ ] `pyright dataloader.py` returns 0 errors
- [ ] No use of `Any` except where unavoidable (document why)

**Expected Skills:** Python typing module, basic PyTorch types

**Estimated Effort:** 1-2 hours

**Owner Archetype:** Builder (C)

---

### Issue #3: Document config.yaml parameters

**Labels:** `type/docs` `difficulty/good-first` `domain/model` `status/ready`

**Description:**
`01_architect_mathematician/config.yaml` lacks inline documentation explaining each parameter.

**Scope:**
- Add YAML comments explaining each parameter
- Document valid ranges and defaults
- Document dependencies between parameters

**Acceptance Criteria:**
- [ ] Every parameter has a comment
- [ ] Comments explain what the parameter controls
- [ ] Comments include valid ranges where applicable
- [ ] Example configurations for different model sizes (50M, 100M, 200M)

**Expected Skills:** YAML, basic understanding of transformer hyperparameters

**Estimated Effort:** 1-2 hours

**Owner Archetype:** Architect (A)

---

### Issue #4: Add validation to algebra generator output

**Labels:** `type/bug` `difficulty/good-first` `domain/data` `status/ready`

**Description:**
`02_data_chef/generators/math/algebra.py` generates solutions but doesn't verify them. Add verification that generated solutions are correct.

**Scope:**
- After generating equation and solution, substitute solution back into equation
- Verify LHS equals RHS
- Add `verified: bool` field to output
- Log and skip any failed verifications

**Acceptance Criteria:**
- [ ] All generated examples are verified
- [ ] Verification uses SymPy symbolic comparison
- [ ] Failed verifications are logged with details
- [ ] `verified` field added to output schema

**Expected Skills:** Python, SymPy basics

**Estimated Effort:** 2-3 hours

**Owner Archetype:** Data Chef (B)

---

### Issue #5: Add CLI help text to build_dataset.py

**Labels:** `type/docs` `difficulty/good-first` `domain/data` `status/ready`

**Description:**
`02_data_chef/build_dataset.py` can be run from command line but has no `--help` output or argument documentation.

**Scope:**
- Add `argparse` with proper help text
- Add arguments for: output directory, number of examples per type, random seed
- Add `--dry-run` option that shows what would be generated

**Acceptance Criteria:**
- [ ] `python build_dataset.py --help` shows all options
- [ ] All arguments have descriptive help text
- [ ] `--dry-run` prints statistics without generating files
- [ ] Default values documented in help

**Expected Skills:** Python argparse

**Estimated Effort:** 1-2 hours

**Owner Archetype:** Data Chef (B)

---

## Intermediate Issues (5)

### Issue #6: Implement gradient checkpointing in OdinModel

**Labels:** `type/feature` `difficulty/intermediate` `domain/model` `status/ready`

**Description:**
The ODIN model in `01_architect_mathematician/src/odin_model.py` does not support gradient checkpointing, limiting the batch size on memory-constrained GPUs.

**Scope:**
- Add optional gradient checkpointing using `torch.utils.checkpoint`
- Checkpoint every N layers (configurable)
- Add config parameter `gradient_checkpointing: bool`
- Benchmark memory savings vs compute overhead

**Acceptance Criteria:**
- [ ] Gradient checkpointing can be enabled via config
- [ ] Memory usage reduced by at least 30% with checkpointing
- [ ] Training produces identical results with/without checkpointing (within tolerance)
- [ ] Benchmark results documented

**Expected Skills:** PyTorch autograd internals, memory profiling

**Estimated Effort:** 4-6 hours

**Owner Archetype:** Architect (A)

---

### Issue #7: Implement TopK gradient compression

**Labels:** `type/feature` `difficulty/intermediate` `domain/swarm` `status/ready`

**Description:**
The swarm network needs gradient compression to reduce bandwidth. Implement TopK sparsification.

**Scope:**
- Create `05_swarm_network/src/compression.py`
- Implement TopK selection (keep top 10% of gradients by magnitude)
- Implement error feedback (accumulate dropped gradients)
- Serialize compressed gradients efficiently

**Acceptance Criteria:**
- [ ] TopK compression reduces gradient size by 90%
- [ ] Error feedback prevents accuracy degradation
- [ ] Unit tests with known gradient tensors
- [ ] Benchmark compression/decompression speed

**Expected Skills:** PyTorch tensor operations, numerical stability

**Estimated Effort:** 4-6 hours

**Owner Archetype:** Optimizer (D)

---

### Issue #8: Add learning rate scheduler to training loop

**Labels:** `type/feature` `difficulty/intermediate` `domain/training` `status/ready`

**Description:**
`03_distributed_builder/src/train.py` uses constant learning rate. Implement configurable schedulers.

**Scope:**
- Add support for: cosine annealing, linear warmup, cosine with restarts
- Add config parameters for scheduler type and arguments
- Log learning rate to tensorboard/wandb
- Add `--resume` flag that restores scheduler state

**Acceptance Criteria:**
- [ ] At least 3 scheduler types implemented
- [ ] Scheduler state saved in checkpoints
- [ ] Learning rate logged every N steps
- [ ] Resume correctly restores scheduler state
- [ ] Unit test for scheduler state save/restore

**Expected Skills:** PyTorch optimizers, training dynamics

**Estimated Effort:** 3-5 hours

**Owner Archetype:** Builder (C)

---

### Issue #9: Implement JSONL streaming for large datasets

**Labels:** `type/perf` `difficulty/intermediate` `domain/data` `status/ready`

**Description:**
`SyntheticMathDataset` in `03_distributed_builder/src/dataloader.py` loads entire dataset into memory. Implement streaming for datasets larger than RAM.

**Scope:**
- Add `StreamingSyntheticDataset` class
- Use memory-mapped file or line-by-line reading
- Maintain shuffle capability (shuffle buffer)
- Benchmark memory usage vs current implementation

**Acceptance Criteria:**
- [ ] Memory usage constant regardless of dataset size
- [ ] Shuffle buffer of configurable size
- [ ] Iteration speed within 10% of in-memory version
- [ ] Benchmark with 10M example dataset

**Expected Skills:** Python file I/O, memory profiling, DataLoader internals

**Estimated Effort:** 4-6 hours

**Owner Archetype:** Builder (C)

---

### Issue #10: Add INT4 quantization option

**Labels:** `type/feature` `difficulty/intermediate` `domain/lowlevel` `status/ready`

**Description:**
`04_lowlevel_optimizer/quantization/int8_quantize.py` only supports INT8. Add INT4 for further size reduction.

**Scope:**
- Implement symmetric INT4 quantization
- Pack two INT4 values per byte
- Add dequantization for inference
- Benchmark accuracy degradation vs INT8

**Acceptance Criteria:**
- [ ] INT4 quantization reduces model size by 50% vs INT8
- [ ] Accuracy within 5% of INT8 on test set
- [ ] Dequantization produces correct values
- [ ] Works with existing ONNX export pipeline

**Expected Skills:** Quantization theory, bitwise operations, NumPy

**Estimated Effort:** 6-8 hours

**Owner Archetype:** Optimizer (D)

---

## Advanced Issues (3)

### Issue #11: Implement Byzantine-tolerant gradient aggregation

**Labels:** `type/research` `difficulty/advanced` `domain/swarm` `status/needs-design`

**Description:**
The swarm network is vulnerable to malicious nodes submitting poisoned gradients. Implement Byzantine-tolerant aggregation.

**Scope:**
- Research: Compare Krum, Multi-Krum, Trimmed Mean, Median
- Implement at least 2 aggregation methods
- Create attack simulation (random noise, sign flip, scaling)
- Benchmark resilience to different attack percentages (10%, 30%, 50%)

**Acceptance Criteria:**
- [ ] Design doc comparing aggregation methods
- [ ] At least 2 methods implemented
- [ ] Convergence maintained with 30% malicious nodes
- [ ] Attack simulation framework reusable for future testing
- [ ] Benchmark results documented

**Expected Skills:** Distributed systems, ML security, statistical aggregation

**Estimated Effort:** 15-25 hours

**Owner Archetype:** Architect (A) + Builder (C)

**Design Required:** Yes — submit ADR before implementation

---

### Issue #12: Port WKV kernel to WebGPU

**Labels:** `type/feature` `difficulty/advanced` `domain/lowlevel` `status/needs-design`

**Description:**
The current WASM runtime is CPU-only. Implement WebGPU compute shaders for WKV (the core RWKV operation).

**Scope:**
- Write WGSL compute shader for WKV operation
- Handle state management across tokens
- Integrate with existing TypeScript runtime
- Benchmark vs WASM CPU implementation

**Acceptance Criteria:**
- [ ] WKV kernel runs on WebGPU
- [ ] Numerical parity with PyTorch reference (within FP32 tolerance)
- [ ] At least 5x speedup over WASM CPU on discrete GPU
- [ ] Graceful fallback when WebGPU unavailable
- [ ] Works in Chrome and Firefox

**Expected Skills:** WGSL, GPU compute programming, WebGPU API, TypeScript

**Estimated Effort:** 20-30 hours

**Owner Archetype:** Optimizer (D)

**Design Required:** Yes — submit ADR with shader design

---

### Issue #13: Implement curriculum learning for math training

**Labels:** `type/research` `difficulty/advanced` `domain/data` `status/needs-design`

**Description:**
Current training uses uniform sampling from all difficulty levels. Implement curriculum learning that progresses from easy to hard problems.

**Scope:**
- Define difficulty metrics for each generator type
- Implement curriculum scheduler (linear, exponential, self-paced)
- Modify DataLoader to support dynamic difficulty
- Benchmark: curriculum vs uniform on held-out test set

**Acceptance Criteria:**
- [ ] Design doc with difficulty metrics per problem type
- [ ] At least 2 curriculum schedules implemented
- [ ] A/B comparison: curriculum vs uniform
- [ ] Final accuracy improvement of at least 5%
- [ ] Reproducible training script

**Expected Skills:** ML training dynamics, curriculum learning literature, experiment design

**Estimated Effort:** 20-30 hours

**Owner Archetype:** Data Chef (B) + Architect (A)

**Design Required:** Yes — submit ADR with experiment plan

---

## Issue Template

When creating new issues, use this template:

```markdown
## Description
[What needs to be done and why]

## Scope
- [ ] Item 1
- [ ] Item 2

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Expected Skills
[List required knowledge]

## Estimated Effort
[X-Y hours]

## Owner Archetype
[A/B/C/D]

## Notes
[Any additional context]
```
