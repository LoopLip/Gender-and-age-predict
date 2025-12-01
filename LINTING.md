Linting and CI

Added:
- .flake8 configuration (max-line-length=88, ignore E203,W503)
- GitHub Actions workflow .github/workflows/lint.yml to run black and flake8 on PRs and pushes.

How to run locally:
- pip install black flake8
- black .
- flake8
