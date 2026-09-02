# GitHub release checklist

Upload this source folder only. The parent research workspace, manuscript, figures and private experiment directories are excluded.

## Before a public push

- Select a project license and confirm permission to license included code. No license grant, author identity, DOI, repository URL or citation metadata is invented here.
- Add accurate citation metadata once authors and publication identifier are finalized.
- Ensure the paper describes this model and feasibility contract. Revise numerical claims from evidence separately; no result tables or manuscript copy are distributed here.
- Run `python -m pytest -q` and the README smoke command.
- Review `git status --short` and `git diff --cached --stat`. Do not force-add ignored data, secrets, results or checkpoints.
- Archive the exact commit, protocol and environment with private experiment evidence. Changing constraints requires new experiments, not relabeling results.

## Upload

Create an empty GitHub repository in your account. In this source folder, commit reviewed files, add the remote URL provided by GitHub, and push `main`. No remote or public release is configured by this preparation step. Alternatively, upload the generated ZIP's contents through GitHub.

Build the ZIP with `python scripts/package_repository.py`. Packaging uses an explicit source allowlist, excludes generated files, and verifies archive hashes. Review the inventory and SHA-256 file before sharing.
