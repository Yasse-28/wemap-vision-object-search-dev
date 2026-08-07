# AI Rules for This Repository

This file defines how AI assistants (ChatGPT, Cursor, etc.) should behave when generating or modifying code in this project.

The goal is consistency, minimal disruption, and respect for existing architecture.

---

## Core Principles

1. **Respect the existing architecture**
   - Do not introduce new patterns if a clear one already exists.
   - Follow current layering and module boundaries.
   - Reuse existing services, repos, and utilities when possible.

2. **Make minimal, targeted changes**
   - Modify only what is necessary to solve the task.
   - Avoid rewriting entire files unless explicitly requested.
   - Preserve naming conventions and structure.

3. **Prefer integration over invention**
   - Before adding new abstractions, check if something similar exists.
   - Do not create parallel systems (e.g. second ORM layer, new config system).
   - Avoid adding dependencies unless clearly justified.

4. **Be explicit and predictable**
   - Write readable, straightforward code.
   - Avoid clever or overly abstract solutions.
   - Favor clarity over novelty.

---

## When Generating Code

- Follow the patterns used in nearby files.
- Match existing error handling and logging style.
- Keep functions and classes focused and single-purpose.
- Add type hints if the project uses them.
- Do not introduce unused helpers or speculative code.

If unsure about architecture or intent:
- Ask for clarification instead of guessing.
- Propose options rather than committing to a large change.

---

## When Modifying Code

- Preserve public interfaces unless told otherwise.
- Avoid breaking changes across modules.
- Do not silently change behavior.
- If refactoring, keep scope limited and explain why.

Never:
- Rename large sections of code without instruction.
- Move files across layers arbitrarily.
- Replace working systems with new frameworks.

---

## Documentation Behavior

When adding documentation or comments:

- Be concise and factual.
- Explain *why* something exists, not just *what* it does.
- Avoid long narrative comments.
- Do not generate large markdown files unless requested.

For AI context files:
- Keep them under ~200 lines.
- Use bullet points and clear sections.
- Avoid describing trivial CRUD logic.

For CHANGELOG entries:
- Write for two audiences: non-technical stakeholders and frontend devs.
- One bullet per feature/fix, max 2 lines. No implementation details.
- For new API endpoints: name the route, one sentence on what it does. No internal plumbing.
- Internal refactors: include only if they affect a public interface, a cross-module contract, or a convention others must follow. Skip local cleanups and minor renames.

---

## Multi-Agent Coordination (coder/reviewer handoff)

When more than one agent works on this repo at the same time (e.g. a coder
agent and a reviewer agent), use `scripts/agent-lock.sh` to avoid two agents
editing the working tree concurrently. See the script's header for the full
state machine and command reference.

- **Coder agent**: before writing any file, run
  `scripts/agent-lock.sh acquire <your-label> "<task>"`. If it fails, the repo
  is locked — wait, don't edit. Call `heartbeat <your-label>` periodically
  during long tasks. When the change is ready for review, run
  `scripts/agent-lock.sh ready <your-label> <commit-or-"uncommitted"> "<comma,separated,files>"`
  instead of editing further — the lock stays held (state `review_ready`)
  until a reviewer picks it up.
- **Reviewer agent**: check `scripts/agent-lock.sh status` — only start a
  review when state is `review_ready`. Claim it with
  `scripts/agent-lock.sh review-start <your-label>`, then finish with either
  `approve <your-label>` (releases the lock) or
  `reject <your-label> "<note>"` (hands it back to the coder, state `coding`).
- Never edit files while the lock is held by another label. Never call
  `release --force` unless the holder is confirmed gone (stale heartbeat
  reported by `status`) — it is a recovery path, not a way to skip the queue.

---

## Testing and Safety

- Add or update tests if the project already uses tests.
- Do not remove tests unless explicitly instructed.
- Avoid introducing breaking changes without warning.
- Assume production code requires stability.

---

## Output Style

Default expectations:

- Propose minimal diffs.
- Show only relevant code snippets when possible.
- Do not reprint entire files unnecessarily.
- Highlight important decisions or assumptions.

If multiple approaches exist:
- Briefly list options.
- Default to the simplest solution consistent with current architecture.

---

## Absolute Don'ts

- Do not redesign the architecture without being asked.
- Do not introduce new frameworks casually.
- Do not generate placeholder code that is never used.
- Do not over-engineer simple features.
- Do not produce 500 lines of documentation for a small change.

---

## Mental Model

Act like a careful contributor joining an existing codebase:
- You are not the original architect.
- You are not here to "improve everything".
- You are here to implement specific changes safely and coherently.

Consistency beats cleverness.
Stability beats novelty.
Small diffs beat grand rewrites.