# Issue Taxonomy

All issues must be labeled. Unlabeled issues will not be triaged.

---

## Label Categories

### `type/*` — What kind of work is this?

| Label | Description | Examples |
|-------|-------------|----------|
| `type/bug` | Something is broken | Incorrect output, crash, regression |
| `type/feature` | New functionality | New generator, new kernel, new API |
| `type/research` | Experimental work with uncertain outcome | Architecture experiments, benchmarks |
| `type/infra` | Build, CI, tooling, deployment | GitHub Actions, Docker, packaging |
| `type/docs` | Documentation only | README, docstrings, comments |
| `type/refactor` | Code improvement without behavior change | Reorganization, naming, deduplication |
| `type/perf` | Performance improvement | Speed, memory, model size |

**Rules:**
- Every issue has exactly one `type/*` label.
- `type/research` issues may have unclear acceptance criteria initially.
- `type/perf` requires before/after benchmarks.

---

### `difficulty/*` — How hard is this?

| Label | Description | Expected Contributor |
|-------|-------------|---------------------|
| `difficulty/good-first` | Self-contained, clear scope, minimal context needed | New contributors, students |
| `difficulty/intermediate` | Requires understanding of one module | Contributors with 1-2 merged PRs |
| `difficulty/advanced` | Cross-module, architectural, or research-heavy | Core contributors |

**Rules:**
- Maintainers assign difficulty labels.
- `good-first` issues must have clear acceptance criteria.
- `advanced` issues may require design discussion first.

---

### `domain/*` — What part of the system?

| Label | Module | Skills |
|-------|--------|--------|
| `domain/model` | `01_architect_mathematician` | PyTorch, RWKV, linear algebra |
| `domain/data` | `02_data_chef` | SymPy, data generation, testing |
| `domain/training` | `03_distributed_builder` | Training loops, ONNX, distributed |
| `domain/lowlevel` | `04_lowlevel_optimizer` | Rust, WASM, memory, quantization |
| `domain/swarm` | `05_swarm_network` | P2P, networking, consensus |
| `domain/browser` | Browser runtime | TypeScript, WebAssembly, Web APIs |

**Rules:**
- Issues may have multiple `domain/*` labels if cross-cutting.
- Domain determines who reviews the PR.

---

### `status/*` — What's the current state?

| Label | Meaning | Action |
|-------|---------|--------|
| `status/needs-triage` | New issue, not yet reviewed | Maintainer will review |
| `status/needs-design` | Requires ADR or design doc before implementation | Do not start coding |
| `status/ready` | Fully specified, ready for implementation | Can be picked up |
| `status/in-progress` | Someone is working on it | Check assignee |
| `status/blocked` | Waiting on external dependency | See blocking issue |
| `status/stale` | No activity for 14+ days | Will be closed soon |

**Rules:**
- Only maintainers change `status/*` labels.
- `status/ready` issues must have acceptance criteria.
- Picking up a `status/ready` issue requires commenting first.

---

### `priority/*` — How urgent?

| Label | Response Time | Use Case |
|-------|---------------|----------|
| `priority/critical` | Same day | Security, data loss, broken master |
| `priority/high` | This week | Blocking other work |
| `priority/medium` | This month | Normal feature/bug work |
| `priority/low` | When convenient | Nice-to-have improvements |

**Rules:**
- Only maintainers assign priority.
- Most issues are `priority/medium` by default.
- `priority/critical` requires maintainer confirmation.

---

## Label Combinations

### Good First Issues
```
type/bug + difficulty/good-first + domain/X + status/ready
type/docs + difficulty/good-first + status/ready
type/feature + difficulty/good-first + domain/X + status/ready
```

### Research Issues
```
type/research + difficulty/advanced + domain/X + status/needs-design
```

### Blocked Issues
```
type/X + status/blocked + (link to blocking issue in description)
```

---

## How to Use Labels

### For Contributors

1. Filter by `difficulty/*` to find appropriate issues
2. Filter by `domain/*` to find issues matching your skills
3. Only pick up `status/ready` issues
4. Comment "I'll take this" before starting

### For Maintainers

1. Apply `status/needs-triage` to new issues automatically
2. Review and apply all required labels
3. Move to `status/ready` when fully specified
4. Assign when someone commits to working on it

---

## Label Colors

| Category | Color | Hex |
|----------|-------|-----|
| `type/*` | Blue | `#1d76db` |
| `difficulty/*` | Green | `#0e8a16` |
| `domain/*` | Purple | `#5319e7` |
| `status/*` | Yellow | `#fbca04` |
| `priority/*` | Red | `#d93f0b` |

---

## Creating New Labels

New labels require:
1. Issue explaining the need
2. Approval from maintainer
3. Update to this document

Do not create ad-hoc labels.
