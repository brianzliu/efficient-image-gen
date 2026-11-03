# Repository instructions

## Commit messages

Use Conventional Commit style for commit messages:

```text
type: short description
type(scope): short description
```

Choose a type that describes the change, such as `feat`, `fix`, `hotfix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, or `chore`. A scope is optional. Keep the description concise, use the imperative mood, and omit the trailing period.

Examples:

- `feat: add accelerated image generation`
- `feat(sampling): add adaptive refresh policy`
- `fix(memory): release cached features after generation`
- `hotfix: correct model loading failure`
- `docs: define image acceleration project and research direction`

Use `hotfix` only for an urgent repair; use `fix` for ordinary bug fixes.
