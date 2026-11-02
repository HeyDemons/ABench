"""Operator source paths, with explicit manifests and compatibility for older source descriptions."""
from pathlib import Path
import re

HF_TREE = re.compile(r'^https://huggingface\.co/(datasets/)?([^/]+/[^/]+)/tree/([^/]+)/?$')
GITHUB_BLOB = re.compile(r'^https://github\.com/([^/]+)/([^/]+)/blob/(.+)$')
CHECKS = ('sha256', 'md5', 'git_blob_sha1', 'md5_first_mib', 'size')


def relative(text):
    text = str(text).strip().rstrip('/')
    if not text or ' ' in text or text.startswith(('/', '~')) or '..' in Path(text).parts:
        return None
    return text


def checked_path(text):
    value = str(text).strip().rstrip('/')
    if (not value or value.startswith(('/', '~')) or '..' in Path(value).parts
            or any(character in value for character in '*?[]')):
        raise ValueError('source raw_path must be a relative file path without traversal or wildcards')
    return value


def files(source):
    """Yield URL, raw-relative path, checksum fields; explicit raw_path always wins over legacy notes.

    A tree's files accepts legacy strings, {remote: raw_path-or-mapping}, or mappings with path/raw_path/checksums.
    """
    url = str(source.get('url') or '')
    if not url.startswith('http'):
        return []
    tree = HF_TREE.match(url)
    note = re.search(r'\bRAW/(\S+)', str(source.get('note') or ''))
    explicit = source.get('raw_path')
    if explicit is not None:
        explicit = checked_path(explicit)
    if tree:
        entries = source.get('files') or []
        if isinstance(entries, dict):
            entries = [dict(value, path=remote) if isinstance(value, dict)
                       else {'path': remote, 'raw_path': value} for remote, value in entries.items()]
        base = explicit or (note.group(1).rstrip('/') if note else None) \
            or relative(source.get('name') or '') or tree.group(2).split('/')[1]
        prefix = f'https://huggingface.co/{tree.group(1) or ""}{tree.group(2)}/resolve/{tree.group(3)}/'
        result = []
        for entry in entries:
            item = {'path': entry} if isinstance(entry, str) else dict(entry)
            remote = checked_path(item['path'])
            target = checked_path(item.get('raw_path') or f'{base}/{remote}')
            checks = {key: item[key] for key in CHECKS if key in item}
            if len(entries) == 1:
                checks = {**{key: source[key] for key in CHECKS if key in source}, **checks}
            result.append((prefix + remote, target, checks))
        return result
    blob = GITHUB_BLOB.match(url)
    if blob:
        url = f'https://raw.githubusercontent.com/{blob.group(1)}/{blob.group(2)}/{blob.group(3)}'
    elif re.fullmatch(r'https://github\.com/[^/]+/[^/]+/?(\.git)?', url):
        return []
    name = relative(source.get('name') or '')
    if explicit:
        target = explicit
    elif note and not note.group(1).endswith('/'):
        target = note.group(1)
    elif note:
        target = note.group(1) + (name or Path(url.split('?')[0]).name)
    elif source.get('path') and name:
        target = f'{source["path"].strip("/")}/{name}'
    else:
        target = name or Path(url.split('?')[0]).name
    # Legacy glob notes describe operator downloads, not an explicit executable manifest.
    if target.startswith('/') or '..' in Path(target).parts:
        raise ValueError('source path must remain inside the raw directory')
    return [(url, target, {key: source[key] for key in CHECKS if key in source})]


def resolve_paths(raw, sources):
    """Resolve unambiguous explicit/legacy destinations before starting a costly build."""
    raw = Path(raw)
    resolved = []
    for source in sources:
        for _, relative_path, _ in files(source):
            target = raw / relative_path
            if target.exists() or any(character in relative_path for character in '*?[]'):
                continue
            hits = [path for path in raw.rglob(target.name) if path.is_file() and path != target
                    and not path.name.endswith(('.part', '.partial'))]
            if len(hits) == 1:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(hits[0].resolve())
                resolved.append(relative_path)
    return resolved
