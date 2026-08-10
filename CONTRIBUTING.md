# Contributing to mosaix-pde

Thank you for your interest in contributing to `mosaix-pde`. Contributions may
include bug reports, feature proposals, documentation improvements, new examples,
numerical methods, and implementations of additional equations.

## Reporting bugs

Please report bugs using
[GitHub issues](https://github.com/acoh64/mosaix-pde/issues). Before opening an
issue, search the existing issues to avoid duplicates. A useful bug report includes:

- A clear description of the problem and the expected behavior
- A minimal reproducible example
- The complete error message and traceback
- Your operating system and Python version
- Relevant package versions

Please do not include passwords, access tokens, private data, or other sensitive
information in an issue.

## Feature requests and support

Feature requests and usage questions are also welcome through
[GitHub issues](https://github.com/acoh64/mosaix-pde/issues). State whether the
issue is a feature proposal or support question, and describe the scientific use
case. For new equations or solvers, include references to the relevant numerical
method when possible.

## Development setup

Clone the repository and install the package with its development dependencies in
a virtual environment:

```bash
git clone https://github.com/acoh64/mosaix-pde.git
cd mosaix-pde
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Testing and style

Before submitting a pull request, run the linter and test suite from the repository
root:

```bash
ruff check .
pytest
```

New functionality should include tests. Changes to user-facing behavior should also
include corresponding documentation or examples.

## Pull requests

1. Create a branch from the latest version of `main`.
2. Keep the change focused and use descriptive commit messages.
3. Add or update tests and documentation as appropriate.
4. Confirm that the tests and lint checks pass.
5. Open a pull request describing the motivation, implementation, and verification.

Maintainers may request changes before merging. Reviews should focus on correctness,
clarity, maintainability, and the scientific use case.

## Community expectations

Communicate respectfully and constructively. Contributors should welcome different
backgrounds and levels of experience, give actionable feedback, and focus discussion
on improving the project. Harassment, personal attacks, and discriminatory behavior
are not acceptable in project spaces.
