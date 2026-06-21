---
name: new-chapter
description: Scaffold a new top-level numbered chapter in this curriculum (e.g. "12_claude_code") following the repo's existing conventions — README.md, numbered sub-folders, per-sub-folder README.md.
---

# New Chapter Scaffolder

Use this skill when the user asks to add a new top-level chapter to this
curriculum (distinct from adding a single app under `08_ai_apps/` or a project
under `09_projects/` — those have their own narrower conventions).

## Steps

1. Determine the next chapter number by listing top-level `NN_*` directories and
   incrementing the highest one. Confirm the name with the user if it's
   ambiguous — chapter names are permanent, renumbering later is disruptive.
2. Create `NN_chapter_name/README.md` modeled on an existing chapter
   (e.g. `10_mcp/README.md` or `11_azure_ai_foundry/README.md`): a one-line
   mission statement, a Prerequisites section linking earlier chapters, a
   Learning Path tree, and a sub-chapter breakdown table.
3. Create numbered sub-folders (`00_setup/`, `01_.../`, …), each with its own
   `README.md` — see `10_mcp/01_mcp_concepts/README.md` for the expected depth
   (architecture diagrams, code blocks, a "Common Mistakes" table, a "Ready to
   move on" link to the next sub-folder).
4. Update the root `CLAUDE.md`: add a row to the main chapter table, add a
   "Chapter NN Sub-chapters" table if there are more than ~3 sub-folders, and
   add the chapter's key README to the "Key Paths" table.
5. Do not touch the directories listed under `CLAUDE.md` → "Do Not Touch".

## Naming Conventions to Enforce

- Directories: `lowercase_underscore`, zero-padded two-digit prefix.
- Each sub-folder is self-contained — don't share state across them.
