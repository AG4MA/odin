# Communication Policy

This document defines how project communication works. Violations result in ignored messages.

---

## Source of Truth

**GitHub is the only source of truth.**

| Authoritative | Non-Authoritative |
|---------------|-------------------|
| GitHub Issues | Discord messages |
| GitHub PRs | Email threads |
| GitHub Discussions (if enabled) | Twitter/X posts |
| Merged documentation | Verbal agreements |
| ADRs | Meeting notes (unless committed) |

If it's not in a GitHub issue, PR, or committed document, it doesn't exist.

---

## Communication Channels

### GitHub Issues
**Purpose:** Bug reports, feature specifications, task tracking

**Use for:**
- Reporting bugs with reproduction steps
- Proposing features with concrete scope
- Tracking work items
- Technical discussions tied to specific work

**Do not use for:**
- General questions (read the docs)
- Brainstorming (write a proposal instead)
- Status updates (use PR)

### GitHub Pull Requests
**Purpose:** Code review, implementation discussion

**Use for:**
- Submitting code changes
- Discussing implementation details
- Iterating on feedback

**Do not use for:**
- Design discussions before implementation (use issue)
- Questions unrelated to the specific PR

### GitHub Discussions (if enabled)
**Purpose:** Q&A, broader technical discussions

**Use for:**
- Technical questions requiring discussion
- RFCs before they become ADRs
- Cross-cutting concerns

**Do not use for:**
- Bug reports (use issues)
- Feature requests (use issues)
- Anything actionable (use issues)

### Discord (if exists)
**Purpose:** Informal community chat

**Status:** Non-authoritative

**Rules:**
- Nothing discussed on Discord is binding
- No decisions are made on Discord
- If it matters, it must be written in an issue
- Maintainers are not obligated to read Discord
- "We discussed this on Discord" is not valid

### Email
**Purpose:** Security reports only

**Use for:**
- Security vulnerability reports
- Sensitive matters requiring privacy

**Do not use for:**
- Technical questions
- Feature requests
- Anything that should be public

---

## Communication Principles

### 1. Async by Default
- Assume 24-48 hour response time
- Don't ping for faster response
- Write complete context in first message
- Don't require synchronous interaction

### 2. Written Over Verbal
- No decisions in calls/meetings without written summary
- Written summaries must be committed or posted to issue
- "We agreed in the call that..." is not valid without documentation

### 3. Public Over Private
- Default to public issues
- DMs for sensitive matters only
- If you DM a question, expect to be redirected to issue

### 4. Concrete Over Abstract
- Include code snippets
- Include error messages
- Include reproduction steps
- "It doesn't work" is not useful

---

## What Not To Do

### Don't: Ask Before Doing
```
❌ "Can I work on this?"
✅ Comment "I'll take this" and start working
```

### Don't: Announce Intentions
```
❌ "I'm thinking about working on X"
✅ Open issue with concrete proposal, or just submit PR
```

### Don't: Request Sync Calls
```
❌ "Can we hop on a call to discuss?"
✅ Write your complete thoughts in the issue
```

### Don't: Use Multiple Channels
```
❌ Post question in Discord, then issue, then email
✅ Pick one channel, wait for response
```

### Don't: Bump Without Substance
```
❌ "Any updates?"
✅ "I can help unblock this by doing X"
```

---

## Response Time Expectations

| Channel | Expected Response | Escalation |
|---------|-------------------|------------|
| Security email | 24 hours | - |
| Issue (bug) | 7 days | Ping in issue |
| Issue (feature) | 14 days | Ping in issue |
| PR review | 7 days | Ping in PR |
| Discord | Never guaranteed | Don't expect one |

**Escalation Path:**
1. Comment in issue/PR (after timeout)
2. Tag specific maintainer (if pattern of non-response)
3. Open meta-issue about responsiveness (last resort)

---

## Notifications

Maintainers will be notified via GitHub for:
- @mentions
- Review requests
- Issues in owned modules

Maintainers are not obligated to:
- Monitor Discord
- Respond to DMs
- Check social media
- Attend to notifications outside working hours

---

## Meetings

ODIN does not have regular meetings. If a meeting is necessary:

1. Meeting must have written agenda posted 24h in advance
2. Meeting must produce written summary
3. Summary must be committed or posted to relevant issue
4. Decisions in meeting are not valid until summary is approved

Meetings are the last resort, not the first option.

---

## Language

All project communication is in English.

- Keep language simple and direct
- Avoid idioms and cultural references
- Use technical terms precisely
- Proofread before posting

---

## Summary

1. GitHub is truth
2. Async is default
3. Written is required
4. Discord is optional
5. Calls are rare
6. Document everything
