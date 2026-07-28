<objective>
Display GSD help at the tier the user asked for: brief (one-line refresher), default (one-page tour), full (complete reference), a single topic section, or a compact scoped lookup of one topic (`--brief <topic>`: signature + one-line summary).

Output ONLY the reference content of the chosen tier. Do NOT add:
- Project-specific analysis
- Git status or file context
- Next-step suggestions
- Any commentary beyond the reference
</objective>

<execution_context>
@/Users/glestryc/personal/github_repos/claude-scratch-work/.cursor/gsd-core/workflows/help.md
</execution_context>

<context>
Arguments: {{GSD_ARGS}}
</context>

<process>
Follow /Users/glestryc/personal/github_repos/claude-scratch-work/.cursor/gsd-core/workflows/help.md with {{GSD_ARGS}}.
</process>
