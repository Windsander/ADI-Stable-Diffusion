# Contributing to ADI

Thanks for your interest in Agile Diffusers Inference! This document is the
short path from "I want to help" to a merged PR.

## Development setup

```bash
# one-shot local build (picks the default ORT provider for your platform)
bash ./auto_build.sh --platform macos --build-type debug   # or windows / linux / android
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the project structure, and the
`engine/` + `sd/` READMEs if you need to prepare the inference engine or
convert models manually.

## Testing

- Compile gates run on every PR (`test-native`, `test-cross`) — keep them green.
- For inference behavior changes, run the smoke matrix locally:

```bash
bash sd/io-test/run_smoke_matrix.sh --quick    # 19 fast cases
```

- Hard gates: zero ORT exceptions, correct output size, no flat-pixel outputs.

## Branch & PR discipline

- Work happens on a topic branch cut from the **latest `main`**
  (`feat/...`, `fix/...`, `ci/...`, `docs/...`, `chore/...`); `main` is
  protected and only reachable via PR with green gates.
- Keep PRs focused: one logical change per PR.
- Fill in the PR template — especially the verification evidence section.

## Commit style

Conventional-ish prefixes, imperative mood:

```
feat: add flow_heun scheduler
fix: skip formula entry when the release asset is missing
ci: pin actions to SHAs
docs: sync README_CN with README
```

## Committing rules

- **Never commit local working docs** (`PLAN-*.md`, `GOAL_*.md`) or generated
  artifacts (build trees, model files, smoke outputs).
- Update `CHANGELOG.md` for user-facing changes.
- Keep `README.md` and `README_CN.md` in sync for user-facing changes.

## Code style

`.clang-format` and `.editorconfig` at the repo root define the baseline
(4-space indent, attached braces, 100-column limit). Format the files you
touch; do **not** reformat the whole tree in a feature PR.

## Releasing (maintainers)

`include/adi.h` (`CURRENT_ADI_VERSION`) is the **single source of truth** for
the ADI version. To cut a release:

```bash
scripts/bump_version.sh v2.1.0           # updates adi.h + scaffolds CHANGELOG
# fill in the CHANGELOG section, commit, then:
git checkout -b release/release-v2.1.0
git push -u origin release/release-v2.1.0
```

`auto-publish` refuses to build a release whose branch name disagrees with
`CURRENT_ADI_VERSION` or lacks a matching CHANGELOG section, so versions
cannot drift apart.
