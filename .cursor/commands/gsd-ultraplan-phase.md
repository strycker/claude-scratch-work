<objective>
Offload GSD's plan phase to Cursor's ultraplan cloud infrastructure.

Ultraplan drafts the plan in a remote cloud session while your terminal stays free.
Review and comment on the plan in your browser, then import it back via /gsd-import --from.

⚠ BETA: ultraplan is in research preview. Use /gsd-plan-phase for stable local planning.
Requirements: Cursor v2.1.91+, claude.ai account, GitHub repository.
</objective>

<execution_context>
@/Users/glestryc/personal/github_repos/claude-scratch-work/.cursor/gsd-core/workflows/ultraplan-phase.md
@/Users/glestryc/personal/github_repos/claude-scratch-work/.cursor/gsd-core/references/ui-brand.md
</execution_context>

<context>
{{GSD_ARGS}}
</context>

<process>
Execute the ultraplan-phase workflow end-to-end.
</process>
