# Contributing to ODIN

This document defines how to contribute to ODIN. Read it completely before opening an issue or PR.

## Core Principles

1. **Code over talk.** Ideas without implementation are noise.
2. **Issues and PRs are the only valid contribution channels.** No Discord conversations, no emails, no "quick chats."
3. **Every contribution must be reviewable.** If it can't be reviewed, it doesn't exist.
4. **Quality over speed.** A rejected PR wastes everyone's time. Do it right.

---

## What We Accept

| Accepted | Not Accepted |
|----------|--------------|
| Bug reports with reproduction steps | "I think X might be broken" |
| Feature PRs with tests | Feature requests without implementation |
| Documentation fixes | Vague improvement suggestions |
| Benchmark results with methodology | "This feels slow" |
| Research with reproducible code | Papers without code |

---

## Contribution Process

### 1. Issues

**Before opening an issue:**
- Search existing issues. Duplicates will be closed without comment.
- Verify the problem exists on `master`.
- Prepare reproduction steps or concrete proposal.

**Issue requirements:**

For bugs:
```
Environment: OS, Python version, GPU (if applicable)
Steps to reproduce: numbered list
Expected behavior: what should happen
Actual behavior: what happens
Error output: full traceback
```

For features:
```
Problem: what limitation exists
Proposed solution: concrete implementation approach
Scope: what files/modules are affected
Acceptance criteria: how to verify completion
```

**Issues that will be closed immediately:**
- "What if we..." without implementation plan
- "Have you considered..." without PR
- Discussions better suited for a blog post
- Requests for mentorship or explanations

### 2. Pull Requests

**Before opening a PR:**
- Open an issue first (except for typo fixes)
- Get issue assigned to you
- Fork and branch from `master`
- One logical change per PR

**PR requirements:**
- Descriptive title: `[module] verb object` (e.g., `[time_mixing] fix state initialization`)
- Link to issue: `Closes #123`
- Tests for new functionality
- Documentation for public APIs
- No unrelated changes

**PR checklist:**
```markdown
- [ ] Tests pass locally
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] No commented-out code
- [ ] No debug prints
- [ ] Commit history is clean (squash if needed)
```

**PRs that will be rejected:**
- No linked issue
- Failing tests
- Unrelated changes bundled in
- Poor commit hygiene
- No response to review comments within 7 days

### 3. Code Review

- All PRs require at least one approval from a maintainer.
- Address all review comments. "I disagree" is not a resolution.
- Maintainers may push commits to your branch to unblock merging.
- Stale PRs (no activity for 14 days) will be closed.

---

## Code Standards

### Python

```python
# Required
- Type hints on all function signatures
- Docstrings on all public functions (Google style)
- 88 character line limit (Black)
- isort for imports

# Forbidden
- `from module import *`
- Mutable default arguments
- Bare `except:`
- `print()` for logging (use `logging` module)
```

### Rust (WASM kernels)

```rust
// Required
- rustfmt formatting
- clippy clean (no warnings)
- #[inline] on hot paths
- no_std compatible where possible
```

### Commit Messages

```
Format: <type>(<scope>): <subject>

Types: feat, fix, docs, test, refactor, perf, chore
Scope: module name (time_mixing, dataloader, wasm, etc.)

Examples:
  feat(algebra): add quadratic equation generator
  fix(wkv): correct state accumulation overflow
  perf(matmul): tile for L1 cache (2.3x speedup)
  docs(readme): add browser compatibility table
```

**Bad commits that will require squashing:**
- "WIP"
- "fix"
- "address review comments"
- "oops"

---

## Contributor Path

Progression is based on demonstrated competence, not time or enthusiasm.

### Level 0: Observer
- Read the codebase
- Read closed issues and PRs
- Understand the architecture
- **Duration:** As long as you need

### Level 1: Issue Commenter
- Comment on issues with:
  - Reproduction confirmations
  - Technical analysis
  - Concrete suggestions
- **Do not:** Ask questions answerable by reading code
- **Promotion criteria:** 3+ useful technical comments

### Level 2: Small PR Contributor
- Fix bugs
- Add tests
- Improve documentation
- Small, well-scoped features
- **Promotion criteria:** 3+ merged PRs with no major revisions

### Level 3: Core Contributor
- Larger features
- Architectural input (via ADR process)
- Review others' PRs
- Triage issues
- **Promotion criteria:** Invitation from maintainers based on track record

---

## For University Students

This section is specifically for CS/ML students who want to contribute as part of coursework, thesis, or skill development.

### Before You Start

**Required baseline:**
- Comfortable reading Python without documentation
- Can navigate a multi-module codebase
- Understand git branching and rebasing (not just `git add -A && git commit && git push`)
- Can run tests and debug failures independently

**Recommended preparation:**
- Read all files in one module completely (e.g., `01_architect_mathematician/src/`)
- Run the test suite
- Read 5 closed PRs to understand review expectations

### Expected Skills by Domain

| Domain | Minimum Skills |
|--------|----------------|
| Model (01) | PyTorch, linear algebra, attention mechanisms |
| Data (02) | SymPy, data structures, property-based testing |
| Training (03) | Distributed systems, ONNX, DataLoader internals |
| Optimization (04) | Rust, WASM, memory layouts, profiling |

### How to Learn by Contributing

1. **Pick a `good-first-issue`** tagged with your domain interest
2. **Read the related code** before asking questions
3. **Propose your approach** in the issue before coding
4. **Submit a draft PR** early for directional feedback
5. **Iterate based on review** - this is where learning happens

You will learn more from one rejected PR with detailed feedback than from tutorials.

### Behaviors That Get PRs Rejected

| Behavior | Why It's a Problem |
|----------|-------------------|
| Submitting without reading the issue | Wastes reviewer time |
| "Can you explain X?" in PR comments | Google it or read the code |
| Ignoring review comments | Shows you're not learning |
| Large PRs without prior discussion | Scope creep, hard to review |
| Copy-pasting from ChatGPT without understanding | Obvious and embarrassing |
| Arguing instead of iterating | This isn't a debate club |
| Disappearing after review | Abandoned PRs get closed |

### What We Will Do for You

- Provide specific, actionable review feedback
- Explain *why* something is wrong, not just *that* it's wrong
- Credit all contributors in release notes
- Write recommendations for students who demonstrate consistent quality

### What We Won't Do

- Teach you Python/PyTorch/Git basics
- Extend deadlines because of your coursework
- Merge substandard work because you're a student
- Have synchronous calls to explain things

---

## What Not To Do

### Don't: Open "Discussion" Issues
```
❌ "Should we consider using JAX instead of PyTorch?"
❌ "I have some ideas about the architecture..."
❌ "What's the roadmap for Phase 3?"
```
These will be closed. If you have a proposal, write an ADR with implementation plan.

### Don't: Submit "Vision" PRs
```
❌ PRs that "lay groundwork" without concrete functionality
❌ Refactors that "will enable future work"
❌ "I reorganized the code to be cleaner"
```
Every PR must have measurable impact.

### Don't: Expect Collaboration Theater
```
❌ Expecting praise for effort
❌ Expecting mentorship
❌ Expecting discussion of your ideas
❌ Expecting exceptions to the process
```

---

## File Locations

| File | Purpose |
|------|---------|
| `CONTRIBUTING.md` | This file |
| `GOVERNANCE.md` | Decision-making process |
| `COMMUNICATION.md` | Communication policy |
| `docs/ISSUE_TAXONOMY.md` | Label definitions |
| `docs/issues/` | Pre-defined issue templates |

---

## Summary

1. Read the code.
2. Open an issue with concrete scope.
3. Get assigned.
4. Submit a clean PR.
5. Address review feedback.
6. Get merged.

No shortcuts. No exceptions.
