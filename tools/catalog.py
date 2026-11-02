"""One case catalog for CLI selection, scheduling and the static website."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TIERS = ('low', 'medium', 'high', 'pro')
SNAPSHOT_FIELDS = ('id', 'category', 'difficulty', 'resource_tier', 'area', 'subarea', 'subset', 'type', 'leaderboard',
                   'environment')
# Copies of tools/harness_kit in every case; the judge's tools tree hash covers them.
KIT_FILES = ('abench_stage.py', 'selfcheck.py', 'abench_metrics.py')


def spec_path(case: str, root: Path = ROOT) -> Path:
    for group in ('standard_cases', 'cases', 'handcrafted'):
        p = root / 'spec' / group / (case + '.yaml')
        if p.is_file():
            return p
    return root / 'cases' / case / 'case' / 'oj.yaml'


def spec_hash(case: str, root: Path = ROOT) -> str:
    return hashlib.sha256(spec_path(case, root).read_bytes()).hexdigest()


def judge_hash(case: str, root: Path = ROOT) -> str:
    """Hash of the case's hand-written judge: everything under harness/ (stages, metrics, helper modules,
    baselines, submission rules) except the shared kit copies and documentation. Frozen data lives elsewhere, so
    freezing data never changes it."""
    harness = Path(root) / 'cases' / case / 'harness'
    digest = hashlib.sha256()
    for path in sorted(harness.rglob('*')):
        rel = path.relative_to(harness)
        if (not path.is_file() or '__pycache__' in rel.parts or path.suffix in ('.pyc', '.md')
                or (len(rel.parts) == 1 and rel.name in KIT_FILES)):
            continue
        digest.update(rel.as_posix().encode() + b'\0' + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def categories(root: Path = ROOT) -> dict[str, str]:
    """Track -> category (spec/categories.yaml), in category order; every index.yaml track must have exactly one."""
    tracks = {}
    for category in yaml.safe_load((Path(root) / 'spec/categories.yaml').read_text())['categories']:
        for track in category['tracks']:
            if track in tracks:
                raise ValueError(f'track {track} is in two categories')
            tracks[track] = category['id']
    return tracks


def category_names(root: Path = ROOT) -> list[str]:
    return [c['id'] for c in yaml.safe_load((Path(root) / 'spec/categories.yaml').read_text())['categories']]


def category_of(case: str, root: Path = ROOT) -> str:
    """The category a case's runs are filed under (runs/<harness>-<model>/<category>/<case>/<run>)."""
    index = yaml.safe_load((Path(root) / 'index.yaml').read_text())
    track = next((t['id'] for t in index['tracks'] if any(c['id'] == case for c in t['cases'])), None)
    if track is None:
        raise ValueError(f'unknown case: {case}')
    category = categories(root).get(track)
    if category is None:
        raise ValueError(f'track {track} of {case} is in no category (spec/categories.yaml)')
    return category


def load_catalog(root: Path = ROOT, *, only: set[str] | None = None) -> list[dict]:
    root = Path(root)
    index = yaml.safe_load((root / 'index.yaml').read_text())
    with (root / 'verify.csv').open(newline='') as f:
        registry = {r['case_id']: r for r in csv.DictReader(f)}
    areas = {}
    for area in yaml.safe_load((root / 'spec/areas.yaml').read_text())['areas']:
        for sub in area['subareas']:
            for case in sub['cases']:
                if case in areas:
                    raise ValueError(f'{case}: duplicate area membership')
                areas[case] = (area['id'], sub['id'])
    track_category = categories(root)
    rows = []
    for track in index['tracks']:
        if track['id'] not in track_category:
            raise ValueError(f"track {track['id']} is in no category (spec/categories.yaml)")
        for case in track['cases']:
            cid = case['id']
            if only is not None and cid not in only:
                continue
            oj = yaml.safe_load((root / 'cases' / cid / 'case/oj.yaml').read_text())
            area, subarea = areas[cid]
            rows.append(dict(case, track=track['id'], track_name=track['name'], group=track.get('group'),
                             category=track_category[track['id']],
                             resource_tier=case['budget'], area=area, subarea=subarea, registry=registry[cid],
                             environment=oj.get('environment') or {}, resources=oj.get('resources') or {},
                             leaderboard=(oj.get('resources') or {}).get('leaderboard'), oj=oj,
                             spec_sha256=spec_hash(cid, root), judge_sha256=judge_hash(cid, root)))
    return rows


def snapshot(row: dict) -> dict:
    return {k: row.get(k) for k in SNAPSHOT_FIELDS}


def _values(value) -> set:
    if value is None:
        return set()
    if isinstance(value, str):
        return {s for s in value.split(',') if s}
    return {str(s) for item in value for s in str(item).split(',') if s}


def select_cases(rows: list[dict], *, category=None, difficulty=None, resources=None, area=None, type=None,
                 subset=None, cases=None, **_) -> list[dict]:
    filters = {'category': category, 'difficulty': difficulty, 'resource_tier': resources, 'area': area, 'type': type,
               'subset': subset, 'id': cases}
    values = {k: _values(v) for k, v in filters.items()}
    requested = values['id']
    unknown = requested - {r['id'] for r in rows}
    if unknown:
        raise ValueError('unknown cases: ' + ', '.join(sorted(unknown)))
    return [r for r in rows if all(not v or str(r.get(k)) in v for k, v in values.items())]


def distribution(rows: list[dict]) -> dict:
    counts = Counter((r['difficulty'], r['resource_tier']) for r in rows)
    return {d: {t: counts[d, t] for t in TIERS} for d in ('low', 'medium', 'high')}


def suite_definition(name: str = 'all', root: Path = ROOT) -> tuple[list[dict], str]:
    """Frozen case entries of a suite version (name or name@version), without checking the working tree."""
    doc = yaml.safe_load((Path(root) / 'spec/suites.yaml').read_text())
    suite, _, version = name.partition('@')
    if suite not in doc['default_versions'] and not version:
        raise ValueError(f'unknown suite: {suite}')
    version = version or str(doc['default_versions'][suite])
    key = suite + '@' + version
    definition = doc['suites'].get(key)
    if definition is None:
        raise ValueError(f'unknown suite: {key}')
    return definition['cases'], key


def suite_cases(name: str = 'all', root: Path = ROOT, rows: list[dict] | None = None, *,
                only: set[str] | None = None) -> tuple[list[dict], str]:
    """Catalog rows of a suite version, each checked against its frozen identity (spec, judge code, environment).

    only restricts both the result and the check to those cases, so a change to one case does not block running
    the others; a suite version is never edited, a changed case needs a new version (`abench suites freeze`)."""
    frozen_cases, key = suite_definition(name, root)
    if only is not None:
        missing = set(only) - {f['id'] for f in frozen_cases}
        if missing:
            raise ValueError(f'{key} does not contain: ' + ', '.join(sorted(missing)))
        frozen_cases = [f for f in frozen_cases if f['id'] in only]
    by = {r['id']: r for r in (rows if rows is not None else load_catalog(root, only=only))}
    selected = []
    for frozen in frozen_cases:
        row = by.get(frozen['id'])
        if (row is None or row['spec_sha256'] != frozen['spec_sha256'] or row['environment'] != frozen['environment']
                or frozen.get('judge_sha256', row['judge_sha256']) != row['judge_sha256']):
            raise ValueError(f'{key}: {frozen["id"]} differs from the frozen suite; '
                             f'freeze a new suite version with `abench suites freeze`')
        selected.append(row)
    return selected, key


def harness_hash(case: str, root: Path = ROOT) -> str:
    """Hash of every file under the case's harness/ directory, the shared kit copies the judge executes included."""
    harness = Path(root) / 'cases' / case / 'harness'
    digest = hashlib.sha256()
    for path in sorted(harness.rglob('*')):
        rel = path.relative_to(harness)
        if path.is_file() and '__pycache__' not in rel.parts and path.suffix != '.pyc':
            digest.update(rel.as_posix().encode() + b'\0' + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def case_lock(case: str, root: Path = ROOT) -> dict:
    """What a run is judged with: the case spec and its own judge code (the suite identity), the effective oj.yaml
    as the evaluator reads it (judge, seeds, submission rules, resources, frozen data hashes), the whole harness
    directory and the published judge image of the case's environment. Recorded in run_lock.json when a run starts
    and compared item by item before the run is judged."""
    from images import image_key, local_lock_entries, lock_entries
    oj = yaml.safe_load((Path(root) / 'cases' / case / 'case' / 'oj.yaml').read_text(encoding='utf-8'))
    effective = json.dumps(oj, sort_keys=True, separators=(',', ':'), default=str)
    environment = oj.get('environment') or {}
    key = f"{image_key(environment.get('profile'))}@{environment.get('version')}"
    # A host-local image ID lock takes part like a published one, so re-locking a rebuilt image never scores older runs.
    image = lock_entries().get(key) or local_lock_entries().get(key)
    return {'spec_sha256': spec_hash(case, root), 'judge_sha256': judge_hash(case, root),
            'environment': oj.get('environment'), 'oj_sha256': hashlib.sha256(effective.encode()).hexdigest(),
            'harness_sha256': harness_hash(case, root), 'judge_image_lock': image}


def check_run_lock(record: dict, root: Path = ROOT) -> None:
    """Refuse to judge a run with anything but what it ran against: its suite version must still hold the case
    unchanged, and the case lock recorded at the start (records from before it existed have only the suite) must
    equal the case now. A changed oj.yaml, metrics.py, stages.py, kit copy, spec, data lock or judge image lock
    therefore never scores an older run."""
    config = record.get('run_config') or {}
    case = record.get('case_id') or config.get('case')
    if not config.get('suite'):
        raise ValueError('agent record names no suite version, so the case it ran against is unknown')
    suite_cases(config['suite'], root, only={case})
    recorded = (config.get('run_lock') or {}).get('case')
    if recorded is not None:
        now = case_lock(case, root)
        changed = sorted(k for k in set(recorded) | set(now) if recorded.get(k) != now.get(k))
        if changed:
            raise ValueError(f'{case} changed since the run started ({", ".join(changed)}); '
                             'judge it with the checkout it ran against')


def freeze_suite(name: str, root: Path = ROOT, *, cases: list[str] | None = None, default: bool = False) -> dict:
    """Add a new immutable suite version (name@version) frozen from the working tree.

    Members: the given cases; otherwise every catalog case for `all`, or the members of the suite's current
    default version."""
    root = Path(root)
    suite, _, version = name.partition('@')
    if not suite or not version:
        raise ValueError('name the new version: SUITE@VERSION')
    path = root / 'spec/suites.yaml'
    doc = yaml.safe_load(path.read_text())
    key = suite + '@' + version
    if key in doc['suites']:
        raise ValueError(f'{key} already exists; suite versions are immutable')
    if not re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)', version):
        raise ValueError('new suites require a semantic version: SUITE@MAJOR.MINOR.PATCH')
    rows = {r['id']: r for r in load_catalog(root)}
    if cases:
        ids = list(dict.fromkeys(cases))
    elif suite == 'all':
        ids = list(rows)
    elif suite in doc['default_versions']:
        ids = [f['id'] for f in suite_definition(suite, root)[0]]
    else:
        raise ValueError(f'new suite {suite}: list its cases')
    unknown = [c for c in ids if c not in rows]
    if unknown:
        raise ValueError('unknown cases: ' + ', '.join(unknown))
    doc['suites'][key] = {'created_at': date.today().isoformat(),
                         'cases': [{'id': c, 'spec_sha256': rows[c]['spec_sha256'],
                                     'judge_sha256': rows[c]['judge_sha256'],
                                     'environment': rows[c]['environment']} for c in ids]}
    if default or suite not in doc['default_versions']:
        doc['default_versions'][suite] = version
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return {'suite': key, 'cases': len(ids), 'default': doc['default_versions'][suite] == version}


def catalog_errors(rows: list[dict], root: Path = ROOT, *, check_site: bool = True) -> list[str]:
    errors = []
    profiles = yaml.safe_load((root / 'spec/env/profiles.yaml').read_text())['profiles']
    policy = yaml.safe_load((root / 'spec/container_policy.yaml').read_text())
    from container_policy import bytes_value
    from images import known_versions
    for row in rows:
        cid, r, env = row['id'], row['resources'], row['environment']
        if 'budget_tier' in row:
            errors.append(f'{cid}: duplicate index budget_tier; use budget for resource tier')
        required = ('budget_tier', 'hardware', 'wall_clock_limit_h', 'leaderboard', 'gpus', 'gpu_class', 'cpus', 'memory', 'shm_size')
        missing = [k for k in required if k not in r]
        if missing or not (r.get('hardware') or {}).get('reference'):
            errors.append(f'{cid}: incomplete resources: {missing}')
        if env.get('profile') not in profiles or not env.get('version'):
            errors.append(f'{cid}: unknown or unversioned environment')
        elif str(env['version']) not in known_versions(env['profile']):
            errors.append(f'{cid}: profile version is neither the current definition nor published')
        if r.get('budget_tier') != ('pro' if row['resource_tier'] == 'pro' else 'budget'):
            errors.append(f'{cid}: budget_tier disagrees with resource tier')
        merged = {**policy['defaults'].get(env.get('profile'), {}), **r}
        for key, cap in policy['limits'].items():
            try:
                actual = bytes_value(merged[key]) if key in ('memory', 'shm_size') else float(merged[key])
                limit = bytes_value(cap) if key in ('memory', 'shm_size') else float(cap)
                if actual > limit or actual < 0 or (actual == 0 and key != 'gpus'):
                    errors.append(f'{cid}: {key} exceeds container policy')
            except (KeyError, ValueError, TypeError):
                errors.append(f'{cid}: invalid container resource {key}')
        if r.get('leaderboard') != ('none' if cid in ('TC-FT-001', 'DS-PIPE-002') else 'all'):
            errors.append(f'{cid}: incorrect leaderboard membership')
        for key in ('budget', 'type', 'area', 'subarea'):
            if row['registry'].get(key) != row.get(key):
                errors.append(f'{cid}: verify.csv {key} differs from catalog')
        source = yaml.safe_load(spec_path(cid, root).read_text())
        for key in ('environment', 'resources'):
            if source.get(key) != row.get(key):
                errors.append(f'{cid}: generated {key} differs from spec')
    site = root / 'site/data/abench-data.js'
    if check_site and site.is_file():
        text = site.read_text()
        start, end = text.find('{'), text.rfind('}')
        try:
            data = json.loads(text[start:end + 1])
            by = {c['id']: c for t in data['index']['tracks'] for c in t['cases']}
            for row in rows:
                c = by.get(row['id'], {})
                if any(c.get(k) != row[k] for k in ('difficulty', 'budget')):
                    errors.append(f'{row["id"]}: website classification differs from catalog')
        except (ValueError, KeyError):
            errors.append('website data cannot be read')
    return errors
