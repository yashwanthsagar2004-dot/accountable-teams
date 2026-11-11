# Intervention Bench (interactive supplement)

The page in `docs/index.html` is built from this folder. Its numbers come from `src/core.ts`, which is checked
against the Python in `src/accteams` and `theory/`:

```bash
python web/tests/make_ref.py                                  # reference values from the repo's Python
node --experimental-strip-types web/tests/core.test.ts        # 15 checks, Node 22.18 or later
```

The page reads `claims/claims.json` and the records in `experiments/` directly, so
it cannot show a number the records do not contain. Build: `cd web && pnpm install && pnpm exec parcel build index.html`.
