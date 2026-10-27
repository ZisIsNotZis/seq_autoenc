# Contributing

This is a research archive, not a maintained library. Focused fixes and
reproducibility notes are welcome; large rewrites are unlikely to be merged.

## Issues

Open an issue for:

- a notebook that fails in a way the README does not already describe;
- a data path or tokenisation step that is documented incorrectly;
- a result plot that can be tied to a run configuration.

Please include the notebook name, the cell, the error, and the Python / PyTorch
versions you used. The corpora are private and missing, so a report that needs
the original data is still useful as a description of the failure.

## Pull requests

- Keep the change focused. One notebook, one fix.
- Clear notebook outputs before committing. Do not add the private corpora,
  derived binaries, or model checkpoints; `.gitignore` already covers the
  common extensions.
- Do not add CI. This archive intentionally has none.
- Say what you ran and on what hardware. There is no test command to run.
- If you rename or add a notebook, update the notebook map in `README.md` and
  `README.zh-CN.md`, and the evidence in `docs/project-status.md`.

Useful contributions: a CPU fallback, a data-prep recipe, a synthetic smoke
test, and extracting the repeated containers into an importable module.

## License

By contributing, you agree that your contribution is licensed under
[AGPL-3.0-only](LICENSE).
