# Governance Model

Lightweight decision-making framework for ODIN. Optimized for speed, not politics.

---

## Roles

### Maintainer
- Merge PRs
- Triage issues
- Make architectural decisions
- Assign issues
- Enforce standards

**Current Maintainers:** See CODEOWNERS file

### Core Contributor
- Review PRs
- Propose architectural changes
- Triage issues (label, not close)
- Mentor new contributors through review

**Becoming Core Contributor:** Invitation after 5+ quality PRs merged

### Contributor
- Submit PRs
- Comment on issues
- Report bugs

**Becoming Contributor:** First merged PR

---

## Decision Making

### Code Changes (Normal PRs)

| Decision | Who Decides | Process |
|----------|-------------|---------|
| Bug fix | Any maintainer | Approve + merge |
| Feature (single module) | Module owner | Approve + merge |
| Feature (cross-module) | 2 maintainers | Both approve + merge |
| Refactor (no behavior change) | Any maintainer | Approve + merge |
| Dependency addition | 2 maintainers | Both approve + merge |

**Timeout:** If no review in 7 days, author may ping in issue. If no response in 14 days, author may request different reviewer.

### Architectural Changes (ADR Required)

Changes that require an Architecture Decision Record:
- New module/component
- API changes affecting multiple modules
- New external dependency (non-trivial)
- Protocol changes (swarm network)
- File format changes (dataset, checkpoints)
- Build system changes

**ADR Process:**
1. Create `docs/adr/NNNN-title.md` using template
2. Open PR with ADR
3. Minimum 3-day discussion period
4. All maintainers must approve or abstain
5. Merge ADR, then implement

**ADR Template:** See `docs/adr/0000-template.md`

### Breaking Changes

Any change that:
- Breaks existing API
- Changes file formats incompatibly
- Removes functionality

**Process:**
1. ADR required
2. All maintainers must approve
3. Deprecation warning in previous release (if applicable)
4. Migration guide required

---

## PR Approval Requirements

| Change Type | Approvals Needed | Who Can Approve |
|-------------|------------------|-----------------|
| Docs only | 1 | Any maintainer |
| Tests only | 1 | Any maintainer |
| Single module | 1 | Module owner or any maintainer |
| Cross-module | 2 | Must include affected module owners |
| Architecture | All maintainers | - |
| Dependencies | 2 | Any maintainers |

---

## Conflict Resolution

### Technical Disagreements

1. Discuss in PR/issue with technical arguments
2. Each party provides concrete evidence (benchmarks, code examples, references)
3. If unresolved after 3 rounds, maintainer with most relevant expertise decides
4. Decision is documented in PR/issue

**Forbidden Arguments:**
- "I prefer..."
- "It feels like..."
- "Other projects do..."
- "We should discuss this more..."

**Valid Arguments:**
- Benchmarks
- Complexity analysis
- Security analysis
- Maintenance burden analysis
- Concrete failure modes

### Process Disagreements

1. Raise in GitHub issue (not PR)
2. Reference specific policy in CONTRIBUTING.md or GOVERNANCE.md
3. Maintainers discuss and update policy if needed
4. Policy change requires PR to relevant .md file

### Personal Conflicts

1. Don't have them.
2. If unavoidable, one party steps back from the specific PR/issue
3. Repeated conflicts result in reduced review assignments

---

## Module Ownership

| Module | Owner | Backup |
|--------|-------|--------|
| `01_architect_mathematician` | TBD | TBD |
| `02_data_chef` | TBD | TBD |
| `03_distributed_builder` | TBD | TBD |
| `04_lowlevel_optimizer` | TBD | TBD |
| `05_swarm_network` | TBD | TBD |
| Browser runtime | TBD | TBD |
| CI/Infrastructure | TBD | TBD |

Ownership means:
- Primary reviewer for module PRs
- Decides module-local architectural questions
- Maintains module documentation

Ownership does not mean:
- Exclusive commit access
- Veto power over cross-module changes
- Immunity from review

---

## Inactivity

### Maintainer Inactivity
- 30 days no activity: ping via issue
- 60 days no activity: maintainer status suspended
- Rights restored upon return and catching up on backlog

### PR/Issue Inactivity
- 14 days no response: warning comment
- 21 days no response: closed as stale
- Can be reopened with explanation

---

## Emergency Procedures

### Security Vulnerability
1. Report via security@[project-email] (not public issue)
2. Maintainers assess severity
3. Fix developed in private branch
4. Coordinated disclosure after fix available

### Broken Master
1. Any maintainer can revert immediately
2. Post-mortem issue opened
3. Fix must include test preventing regression

### Maintainer Unavailability
If all maintainers unavailable for 30+ days:
1. Core contributors may merge critical fixes
2. No new features until maintainer returns
3. Document all decisions for review

---

## Amendments

This document can be changed via PR:
1. Propose change with rationale
2. 7-day comment period
3. All active maintainers must approve
4. Changes effective upon merge
