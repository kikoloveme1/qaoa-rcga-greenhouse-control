"""Create a verified source-only archive using an explicit file allowlist."""
from hashlib import sha256
from pathlib import Path
import json
import zipfile

ROOT_FILES = {'.gitignore', '.gitattributes', 'README.md', 'pyproject.toml', 'requirements-lock.txt',
              'LICENSE', 'CITATION.cff', 'CITATION.md'}
PYTHON_DIRS = {'algorithms', 'environment', 'paper_protocol', 'tests', 'scripts'}
CONFIG_FILES = {'principal.json', 'matched.json', 'sensitivity.json'}


def source_files(root):
    selected = []
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if p.is_symlink() or any((root.joinpath(*rel.parts[:i])).is_symlink() for i in range(1, len(rel.parts))):
            continue
        if any(part.startswith('.') or part == '__pycache__' or part.endswith('.egg-info')
               for part in rel.parts[:-1]) and rel.parts[:2] != ('.github', 'workflows'):
            continue
        allowed = (len(rel.parts) == 1 and rel.name in ROOT_FILES
                   or rel.parts[0] in PYTHON_DIRS and p.suffix == '.py'
                   or rel.parts[0] == 'docs' and p.suffix == '.md'
                   or len(rel.parts) == 2 and rel.parts[0] == 'configs' and rel.name in CONFIG_FILES
                   or rel.as_posix() in ('data/README.md', 'data/singapore_greenhouse_schema.csv',
                                          '.github/workflows/tests.yml'))
        if allowed:
            selected.append(p)
    return selected


def main():
    root = Path(__file__).resolve().parents[1]
    output = root.parent / 'qaoa-rcga-source.zip'
    files = source_files(root)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in files:
            archive.write(p, 'qaoa-rcga/' + p.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('archive CRC check failed')
        for p in files:
            name = 'qaoa-rcga/' + p.relative_to(root).as_posix()
            if sha256(archive.read(name)).digest() != sha256(p.read_bytes()).digest():
                raise RuntimeError(f'archive content mismatch: {name}')
    digest = sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n', encoding='ascii')
    print(json.dumps({'archive': str(output), 'files': len(files), 'bytes': output.stat().st_size,
                      'sha256': digest, 'all_member_hashes_verified': True}, indent=2))


if __name__ == '__main__':
    main()
