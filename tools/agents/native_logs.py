"""Copy a harness's own session transcripts into the run record as raw evidence.

Transcripts are richer than the streamed stdout (tool outputs, reasoning, per-call usage) and support later
ATIF conversion. They never decide a score. A file containing the model credential is withheld, not redacted:
the record then names the file and why it is missing."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil


def decompressed(path: Path) -> bytes:
    """A file's bytes as a reader sees them: .gz and .zstd/.zst are expanded (zstandard, Python's compression.zstd,
    or the zstd CLI); other files are returned as they are."""
    data = Path(path).read_bytes()
    suffix = Path(path).suffix
    if suffix == '.gz':
        import gzip
        return gzip.decompress(data)
    if suffix not in ('.zstd', '.zst'):
        return data
    try:
        import zstandard
        return zstandard.ZstdDecompressor().decompressobj().decompress(data)
    except ImportError:
        pass
    try:
        from compression import zstd
        return zstd.decompress(data)
    except ImportError:
        pass
    import subprocess
    return subprocess.run(['zstd', '-dc'], input=data, capture_output=True, check=True).stdout


def collect_native_logs(source: Path | None, destination: Path, secrets=()) -> dict:
    record = {'files': [], 'withheld': [], 'bytes': 0}
    if not source or not Path(source).is_dir():
        return record
    secrets = [s.encode() for s in secrets if s and len(s) >= 8]
    for path in sorted(Path(source).rglob('*')):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        data = path.read_bytes()
        try:
            plain = decompressed(path)        # a compressed session log hides the credential from a byte search
        except Exception:
            record['withheld'].append({'path': relative, 'reason': 'compressed and could not be checked for the credential'})
            continue
        if any(secret in data or secret in plain for secret in secrets):
            record['withheld'].append({'path': relative, 'reason': 'contains the model credential'})
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        record['files'].append({'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        record['bytes'] += len(data)
    return record
