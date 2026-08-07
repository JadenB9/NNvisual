# Contributing

## Setup

```sh
git clone https://github.com/JadenB9/NNvisual.git
cd NNvisual
npm install          # dev tooling only — the app itself has zero dependencies
npm run serve        # http://localhost:4173
```

## Workflow

`main` is protected: changes land through pull requests with green CI.

1. Branch from `main` (`feat/…`, `fix/…`, `docs/…`).
2. Make the change. Keep the engine (`public/js/nn/`) free of DOM code so it
   stays testable in Node.
3. `npm run lint && npm test` locally.
4. Open a PR — the template asks what changed, why, and how to verify it.
5. Squash-merge once CI is green. Branches are deleted on merge.

## Code style

- Plain vanilla JavaScript, ES modules, no frameworks and no build step.
- 4-space indent, semicolons, single quotes (ESLint enforces the basics).
- Prefer a readable loop over a clever one-liner — this project exists to make
  the math legible.
- New engine behavior needs a test. UI-only changes need a manual check in the
  browser (note it in the PR).

## Commits

Short imperative subject lines ("Add spiral dataset", "Fix relu gradient at
init"). Reference issues where one exists.
