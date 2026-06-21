---
description: Scaffold a new standalone app under 08_ai_apps/ following this repo's app conventions.
argument-hint: <app-number> <app-name> "<one-line pattern description>"
---

Scaffold a new app at `08_ai_apps/$1_$2/` for this curriculum repo.

Read `CLAUDE.md` → "Chapter 08 Apps" table first to see the existing numbering
and pattern before creating anything.

Create:
- `08_ai_apps/$1_$2/main.py` — a minimal runnable entry point, not a stub. It
  should actually run with `python main.py` after `pip install -r requirements.txt`.
- `08_ai_apps/$1_$2/README.md` — what the app demonstrates: $3

Then update the "Chapter 08 Apps" table in the root `CLAUDE.md` with the new
row.

Do not add a `docker-compose.yml` unless the app genuinely needs a service
(database, queue, etc.) — most apps in this chapter are standalone.
