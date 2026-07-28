<objective>
GSD smart entry — the state-aware front door. Detect what's going on in this project, then present a short menu of the right next actions and dispatch to one.

This is a launcher/router only. It never does the work itself. It reads project + workflow state via `gsd-tools smart-entry --json`, shows a situation-appropriate menu, and hands off to an existing GSD command.
</objective>

<execution_context>
@/Users/glestryc/personal/github_repos/claude-scratch-work/.cursor/gsd-core/workflows/smart-entry.md
@/Users/glestryc/personal/github_repos/claude-scratch-work/.cursor/gsd-core/references/ui-brand.md
</execution_context>

<context>
Arguments: {{GSD_ARGS}}
</context>

<process>
Follow /Users/glestryc/personal/github_repos/claude-scratch-work/.cursor/gsd-core/workflows/smart-entry.md. Detect the situation, present the menu, and dispatch exactly one command. Then stop.
</process>
