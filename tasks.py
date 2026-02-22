from invoke import task


@task
def clean(c):
    c.run("find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true", warn=True)
    c.run("rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage", warn=True)


@task
def install(c):
    c.run("uv sync --all-extras")


@task
def format(c):
    c.run("uv run ruff format src tests tasks.py", pty=True)


@task
def lint(c, fix=False):
    fix_flag = "--fix" if fix else ""
    c.run(f"uv run ruff check src tests {fix_flag}", pty=True)


@task
def typecheck(c):
    c.run("uv run mypy src", pty=True)


@task
def test(c):
    c.run("uv run pytest", pty=True)


@task
def check(c):
    format(c)
    lint(c)
    typecheck(c)
    test(c)
