from scripts.package_repository import source_files


def test_archive_excludes_results_measurements_and_credentials(tmp_path):
    names = ['README.md', '.gitignore', 'pyproject.toml', 'algorithms/example.py',
             'configs/principal.json', 'data/singapore_greenhouse_schema.csv',
             'data/README.md', '.github/workflows/tests.yml',
             '.env', '.env.local', 'private.key', 'data/nested/observations.csv',
             'results/run.json', 'configs/token.json', 'paper/manuscript.tex',
             'algorithms/__pycache__/example.pyc', 'algorithms/checkpoint.pt']
    for name in names:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('test')
    included = {p.relative_to(tmp_path).as_posix() for p in source_files(tmp_path)}
    assert included == set(names[:8])
