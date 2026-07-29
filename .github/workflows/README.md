# GitHub Actions — storyland-ai

Catalog of this repo's CI/CD workflows. Last audited **2026-07-25**. Org-wide CI policy
lives in `storyland-team-os/conventions/engineering-standards.md`; this file is the
repo-local companion to it. This is the most workflow-heavy repo in the fleet — **12
workflow files** plus GitHub-managed **Dependabot Updates**.

> **Public-repo constraint:** storyland-ai is **public**, and a public repo cannot `uses:` a
> reusable workflow that lives in the **private** `storyland-infrastructure`. So this repo's
> deploy and gitleaks are **inlined copies** that must be kept in sync manually with the infra
> originals — noted per row below.

| Workflow | File | Triggers | Purpose | Verdict |
|---|---|---|---|---|
| **Unit Tests** | `codex.yml` | push/PR `main`+`develop`, dispatch, call | `make test-ci` — unit tests + coverage ratchet. Core PR gate. | **Keep** |
| **Integration Tests** | `integration-tests.yml` | push/PR `main`+`develop`, dispatch, call | Integration suite against **mocked / VCR cassettes** — no live API spend. Runs on every push/PR. | **Keep** |
| **Integration Tests (Live API)** | `integration-live-tests.yml` | dispatch (requires `reason`) | The **live-API** counterpart — spends real quota/cost. Manual-only by design; run to validate real API behavior before a release. Intentionally distinct from the mocked suite above. | **Keep** |
| **CI/CD Pipeline** | `ci-cd.yml` | push `main`, dispatch (`skip-tests`) | **Test orchestration only** — fans out to `codex.yml` (unit) + `integration-tests.yml`. Contains **no build/publish/deploy job**; a `main` push does not produce or publish an artifact. | **Keep** |
| **Gitleaks** | `gitleaks.yml` | push/PR `main` | Full-history secret scan. **Inline** (mirrors `infra/reusable-gitleaks.yml` step-for-step; cannot `uses:` the private reusable — public-repo constraint). Re-point to the reusable version if infra ever goes public. | **Keep** |
| **pip-audit** | `pip-audit.yml` | push/PR `main`+`develop` | Python dependency vulnerability gate. | **Keep** |
| **Deploy AI (prod)** | `deploy-ai-prod.yml` | dispatch | G5-gated production release. **Self-contained / inlined** — mirrors `infra/deploy-service.yml` step-for-step (stage gate, build, SSH deploy) because it cannot call the private reusable. **Keep-in-sync** with the infra original. Supports build-and-deploy or `image_tag` redeploy. | **Keep** |
| **Claude Code** | `claude.yml` | issue_comment, PR review (+comment), issues opened/assigned | The `@claude` agent. Fires only when mentioned — **`skipped` runs on unrelated comments are normal**, not failures. | **Keep** |
| **PR body sections** | `pr-body-sections.yml` | PR opened/edited/reopened/synchronize | Enforces the four PR template headings (`## Why / ## What / ## Docs / ## Verification`, MYS-617). **Required check** (Dependabot / `Revert …` PRs skip). | **Keep** |
| **Evaluation (manual dispatch)** | `scheduled-eval.yml` | dispatch (runner/datasets/max_cases) | Runs an eval runner (itinerary / local_atmosphere / expansion / place_to_book) against Langfuse datasets. **Manual-only — has no cron.** Renamed 2026-07-25 (was "Scheduled Evaluation", which implied a schedule it never had). Filename kept as-is to avoid breaking references; `evaluation/README.md` updated to match. | **Keep** |
| **Re-record VCR Cassettes** | `re-record-cassettes.yml` | dispatch | Manual utility to regenerate VCR cassettes (MYS-440 re-record split). No automatic trigger; run when live responses change. | **Keep** (manual tool) |
| **Sync rotated env keys to box** | `sync-env-keys.yml` | dispatch | Manual utility to push rotated env keys to the prod box (MYS-440 key rotation). | **Keep** (manual tool) |
| **Dependabot Updates** | *(GitHub-managed, `.github/dependabot.yml`)* | Dependabot schedule | Automated dependency-update PRs. Not a YAML in `.github/workflows/`. | **Keep** |
| **CI summary comment** | `ci-summary.yml` | `workflow_run` completed on this repo's PR-reporting workflows + dispatch | Upserts ONE sticky comment per PR summarising every workflow's conclusion for the PR's current head SHA. Exists because agents cannot read CI at all — the Checks API 403s (no `checks` scope on the connector app, and our own PAT 403s too) and the Actions API is refused by the agent sandbox's outbound proxy (MYS-672). Runs inside Actions, so it has the access agents lack, and writes into a channel they can read. **Informational — never add to the required-checks list; it reports CI, it does not gate it.** Takes the newest run per workflow name (a re-run leaves stale failures on the same SHA) and reports "no runs" as explicitly NOT green. | **Keep** |

## Health notes (2026-07-25)
- `main` **green**; security lane (Gitleaks, pip-audit, Dependabot) all passing.
- Naming defect **fixed** in this PR: `scheduled-eval.yml` was titled "Scheduled Evaluation"
  but only ever ran on `workflow_dispatch` (no `schedule:`) — the name implied a cron that
  did not exist. Manual-only is intended (cost); renamed to make the name honest, and the
  Actions-tab reference in `evaluation/README.md` updated to the new name.

## Conventions
- Two integration workflows are **intentionally split**: mocked/VCR on every push vs.
  live-API on manual dispatch. Do not "consolidate" them.
- Deploy + gitleaks are **inlined** here (public-repo constraint) and must be kept in sync
  with the infra originals — change both places.
- Deploys are `workflow_dispatch`-only and G5-gated; there is no push-to-deploy.
