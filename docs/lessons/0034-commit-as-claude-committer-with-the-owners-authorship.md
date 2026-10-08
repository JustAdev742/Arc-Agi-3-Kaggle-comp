Summary: commit with committer Claude <noreply@anthropic.com> and author JustAdev742 <owner's email>; the session's push check flags commits whose committer email is not noreply@anthropic.com (GitHub shows them as Unverified).

# Commit identity (2026-10-08)

What happened: the stop hook (~/.claude/stop-hook-git-check.sh) flagged an unpushed commit whose author and committer
were both JustAdev742 <the owner's email>, the identity this repo had used throughout: "GitHub will show as
Unverified (missing signature, or committer email is not noreply@anthropic.com)". Amending only the committer (keeping
the author) cleared it.

How to apply:
- Commit with
  `git -c user.name=Claude -c user.email=noreply@anthropic.com commit --author="JustAdev742 <owner's email>" -m "..."`,
  messages ending with the Co-Authored-By and Claude-Session lines.
- To fix an unpushed tip: `git -c user.name=Claude -c user.email=noreply@anthropic.com commit --amend --no-edit`
  (keeps the author). Never rewrite history that is already pushed.
- Brief subagents with the same command: their branches are merged and pushed from here.
