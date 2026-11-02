#!/usr/bin/env python3
"""ABench command line: Docker runs, calibration and durable jobs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import secrets
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from catalog import ROOT, catalog_errors, distribution, freeze_suite, load_catalog, select_cases, suite_cases, suite_definition
import jobs


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def call(tool: str, arguments: list[str], root: Path = ROOT, env: dict | None = None) -> int:
    return subprocess.call([sys.executable, str(root / 'tools' / tool), *arguments], cwd=root, env=env)


def load_env(path: Path = ROOT / '.env') -> list[str]:
    """Set the operator's NAME=value lines from `.env` (template: .env.example) that the shell has not already set,
    so that subcommands and their child processes inherit them. Values are literal (no variable expansion or inline
    comments); empty values are skipped. Errors name the line, never its value."""
    if not path.exists():
        return []
    loaded = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, sep, value = line.removeprefix('export ').partition('=')
        name, value = name.strip(), value.strip()
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            raise ValueError(f'{path}:{number}: expected NAME=value')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        if value and name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def add_filters(parser):
    parser.add_argument('--suite', default='all')
    for name in ('category', 'difficulty', 'resources', 'area', 'type', 'subset', 'cases'):
        parser.add_argument('--' + name, help='comma-separated values')


def selected(args):
    """Filter the suite's members, then check only the chosen cases against their frozen identity."""
    members = {f['id'] for f in suite_definition(args.suite)[0]}
    rows = [r for r in load_catalog() if r['id'] in members]
    chosen = select_cases(rows, **vars(args))
    return suite_cases(args.suite, rows=rows, only={r['id'] for r in chosen})


PAPER_INDEX_MIN_PAPERS = 100_000      # an arXiv snapshot has millions; a handful of papers is a test index


def paper_index_check(path: str | None = None) -> dict:
    """Whether the offline paper search that task statements offer will work: the index the runner would mount
    (--papers-index, else ABENCH_PAPERS_INDEX, else ~/.abench/papers/current.sqlite) exists, opens read-only,
    answers a full-text query and covers an arXiv snapshot rather than a test sample."""
    import sqlite3
    from agents.workspace import DEFAULT_PAPER_INDEX, paper_index_info
    source = 'argument' if path else 'environment' if os.environ.get('ABENCH_PAPERS_INDEX') else 'default'
    p = Path(path or os.environ.get('ABENCH_PAPERS_INDEX') or DEFAULT_PAPER_INDEX).expanduser()
    check = {'path': str(p), 'source': source}
    if not p.is_file():
        return {**check, 'passed': False, 'detail': "no paper index: agents' offline paper search fails; build one "
                '(tools/paper_search/README.md) or set ABENCH_PAPERS_INDEX'}
    try:
        info = paper_index_info(str(p))
        con = sqlite3.connect(f'file:{p.resolve()}?mode=ro', uri=True)
        try:
            con.execute("SELECT rowid FROM papers_fts WHERE papers_fts MATCH 'learning' LIMIT 1").fetchall()
        finally:
            con.close()
    except sqlite3.Error as error:
        return {**check, 'passed': False, 'detail': f'unreadable paper index: {error}'}
    papers = int(info.get('papers') or 0)
    check.update(passed=papers >= PAPER_INDEX_MIN_PAPERS, papers=papers, cutoff=info.get('cutoff'),
                 snapshot_sha256=info.get('snapshot_sha256'))
    if not check['passed']:
        check['detail'] = f'only {papers} papers: a test index, not an arXiv metadata snapshot'
    return check


def doctor() -> int:
    checks = {'python': sys.version.split()[0], 'dependencies': {n: importlib.util.find_spec(n) is not None for n in ('numpy', 'yaml')},
              'key_file': str(Path.home() / '.abench/hidden.key'),
              'key_present': (Path.home() / '.abench/hidden.key').is_file(),
              'paper_index': paper_index_check()}
    try:
        r = subprocess.run(['docker', 'info', '--format', '{{json .}}'], check=True, capture_output=True, text=True, timeout=20)
        info = json.loads(r.stdout)
        checks['docker'] = {'available': True, 'version': info.get('ServerVersion'), 'platform': info.get('OSType', 'linux') + '/' +
                            {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(info.get('Architecture'), info.get('Architecture', 'unknown')),
                            'rootless': any('rootless' in s for s in info.get('SecurityOptions', [])),
                            'desktop': 'docker desktop' in str(info.get('OperatingSystem', '')).lower(),
                            'runtimes': list((info.get('Runtimes') or {}).keys()),
                            'resources_enforced': all(info.get(key) for key in ('MemoryLimit','CpuCfsQuota','PidsLimit')) and
                                (not any('rootless' in value for value in info.get('SecurityOptions',[])) or
                                 info.get('CgroupDriver') == 'systemd' and str(info.get('CgroupVersion')) == '2')}
        from container_policy import gpu_mode, gpu_options
        mode = gpu_mode(info)
        checks['gpu_access'] = {'mode': mode,
                                'source': 'environment' if os.environ.get('ABENCH_GPU_MODE') not in (None, '', 'auto') else 'docker_discovery',
                                'suggested_command': 'docker run --rm --network none ' +
                                    ('--device nvidia.com/gpu=<idle-id>' if mode == 'cdi' else '--gpus device=<idle-id>') +
                                    ' <profile-image> nvidia-smi',
                                'probe_policy': 'only probe a GPU with no active compute users'}
        from images import resolve_image
        try:
            image = resolve_image('cpu-sci', allow_unlocked=True)
            probe = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--user', '1000:1000', '--cap-drop', 'ALL',
                                    '--security-opt', 'no-new-privileges', image, 'python3', '-c',
                                    'import socket; print(socket.if_nameindex())'], capture_output=True, text=True, timeout=60)
            checks['network_none'] = {'passed': probe.returncode == 0, 'output': probe.stdout.strip(), 'detail': probe.stderr[-600:]}
        except (ValueError, RuntimeError, subprocess.SubprocessError) as e:
            checks['network_none'] = {'passed': False, 'detail': str(e)}
        # Run real transport/cgroup/GPU probes when a locally built or locked image exists.
        try:
            from runtime.probe import probe
            checks['runtime_probe'] = probe(resolve_image('cpu-sci', allow_unlocked=True))
        except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
            checks['runtime_probe'] = {'passed': False, 'detail': str(error)}
        lock_check = subprocess.run([sys.executable, str(ROOT / 'tools/images.py'), 'verify'], capture_output=True, text=True, timeout=30)
        checks['image_lock'] = {'passed': lock_check.returncode == 0, 'detail': (lock_check.stdout + lock_check.stderr)[-2000:]}
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        checks['docker'] = {'available': False, 'detail': str(e)}
    inventory = jobs.gpu_inventory()
    checks['gpus'] = inventory
    idle = next((gpu for gpu in inventory if gpu['idle']), None)
    if checks['docker']['available'] and idle:
        try:
            from images import resolve_image
            image = resolve_image('cpu-sci', allow_unlocked=True)
            gpu_args = gpu_options([idle['index']], info)
            r = subprocess.run(['docker', 'run', '--rm', '--network', 'none', *gpu_args, image,
                                'nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
            checks['gpu_runtime'] = {'mode': checks['gpu_access']['mode'], 'passed': r.returncode == 0 and idle['uuid'] in r.stdout,
                                     'selected': idle['index'], 'output': r.stdout.strip(), 'detail': r.stderr[-1000:]}
        except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
            checks['gpu_runtime'] = {'passed': False, 'detail': str(error)}
    else:
        checks['gpu_runtime'] = {'mode': checks.get('gpu_access', {}).get('mode'), 'skipped': 'no idle GPU or Docker unavailable'}
    supported = []
    skipped = []
    for row in load_catalog():
        reason = jobs.skip_reason(row, inventory)
        (skipped if reason else supported).append({'case': row['id'], 'reason': reason} if reason else row['id'])
    checks['supported_cases'] = supported
    checks['skipped_cases'] = skipped
    checks['resource_tiers'] = sorted({r['resource_tier'] for r in load_catalog() if r['id'] in supported})
    checks['ready_for_scoring'] = bool(checks['docker']['available'] and checks.get('image_lock', {}).get('passed')
                                      and checks.get('network_none', {}).get('passed') and checks['paper_index']['passed'])
    emit(checks)
    return 0 if checks['docker']['available'] and all(checks['dependencies'].values()) else 1


def judge_run(argv: list[str]) -> int:
    """Complete deferred judging, or judge again after the infrastructure failed judging, while preserving
    immutable final records and original agent restrictions."""
    ap = argparse.ArgumentParser(prog='abench judge')
    ap.add_argument('--run', required=True)
    ap.add_argument('--hidden-key-file', default=str(Path.home() / '.abench/hidden.key'))
    ap.add_argument('--backend', choices=('docker', 'host'))
    ap.add_argument('--gpus')
    ap.add_argument('--image')
    ap.add_argument('--rejudge', metavar='REASON',
                    help='judge a final evaluation again (e.g. after a judge rule changed); it is archived as '
                         'evaluation.superseded-N.json and the new record names it and the reason')
    args = ap.parse_args(argv)
    run = Path(args.run).expanduser().resolve()
    out = run / 'evaluation.json'
    previous = archive = None
    if out.is_symlink():
        raise ValueError('evaluation record must not be a symlink')
    if out.exists():
        from agents.errors import retryable_judge_failure
        previous = json.loads(out.read_text())
        if previous.get('verdict') == 'not_evaluated':
            archive = run / 'evaluation.deferred.json'
        elif retryable_judge_failure(previous):
            # Judging failed on the infrastructure: judge the collected submission again, keeping the failure.
            archive = run / f'evaluation.judge-error-{len(list(run.glob("evaluation.judge-error-*.json"))) + 1}.json'
        elif args.rejudge:
            archive = run / f'evaluation.superseded-{len(list(run.glob("evaluation.superseded-*.json"))) + 1}.json'
        else:
            raise ValueError(f'evaluation record already exists and is immutable: {out}')
        if archive.exists() or archive.is_symlink():
            raise ValueError(f'evaluation archive already exists: {archive}')
    agent_path = run / 'agent_run.json'
    if agent_path.is_symlink() or not agent_path.is_file():
        raise ValueError('agent_run.json must be a regular file')
    record = json.loads(agent_path.read_text())
    mode = record.get('run_mode')
    if mode not in ('benchmark', 'smoke'):
        raise ValueError('agent record has no supported run_mode')
    import oj_eval
    from agents.scoring import apply_agent_attestation
    from sandbox_runner import sha256_file, validate_hidden_key
    submission = run / 'submission'
    expected = (record.get('submission') or {}).get('tree_sha256')
    if not expected or expected != oj_eval.tree_hash(submission):
        raise ValueError('collected submission differs from agent_run.json tree hash')
    from catalog import check_run_lock
    check_run_lock(record)
    sandbox = record.get('sandbox') or {}
    backend = args.backend or sandbox.get('backend') or ('docker' if record.get('isolation') == 'container' else 'host')
    if backend == 'host' and mode != 'smoke':
        raise ValueError('host judging requires the original run to be smoke')
    key = validate_hidden_key(args.hidden_key_file) if mode == 'benchmark' else None
    gpus = args.gpus if args.gpus is not None else sandbox.get('allocated_gpus')
    from runtime.cleanup import sigterm_interrupts
    with sigterm_interrupts():          # jobs stop a re-judge with SIGTERM; its judge containers are still removed
        evaluation = oj_eval.evaluate(record.get('case_id') or record.get('case'), submission, key, mode,
                                      backend=backend, gpus=gpus, image=args.image,
                                      run_id=record.get('run_id') or run.name)
    apply_agent_attestation(evaluation, record)
    evaluation['agent_run_sha256'] = sha256_file(agent_path)
    if args.rejudge and archive is not None and archive.name.startswith('evaluation.superseded-'):
        evaluation['supersedes'] = {'record': archive.name, 'verdict': previous.get('verdict'), 'reason': args.rejudge}
    text = json.dumps(evaluation, ensure_ascii=False, indent=2) + '\n'
    if previous is not None:
        # Preserve the placeholder or failed judging, and never overwrite an existing archive or final evaluation.
        if json.loads(out.read_text()) != previous:
            raise ValueError('evaluation record changed during judging')
        os.link(out, archive)
        out.unlink()
    with out.open('x', encoding='utf-8') as target:
        target.write(text)
    emit(evaluation)
    return 0 if evaluation.get('verdict') in ('evaluated', 'judge_pending', 'smoke_unscored') else 1


def calibrate(argv: list[str]) -> int:
    """Optional shortcut: the oracle runs the case's reference submission with this checkout's code and the data
    already frozen in it, and the judge evaluates it. Nothing is copied or prepared; the key is passed by path."""
    ap = argparse.ArgumentParser(prog='abench calibrate')
    ap.add_argument('case')
    ap.add_argument('--hidden-key-file', default=str(Path.home() / '.abench/hidden.key'))
    ap.add_argument('--run-id')
    ap.add_argument('--gpus')
    ap.add_argument('--device')
    ap.add_argument('--backend', choices=('docker', 'host'), default='docker')
    ap.add_argument('--allow-nonreference-gpu', action='store_true')
    ap.add_argument('--allow-shared-gpu', action='store_true', help='unscored Budget validation on a sufficiently free low-load GPU')
    ap.add_argument('--image')
    if any(a.split('=')[0] in ('--data', '--rows', '--allow-unverified') for a in argv):
        ap.error('calibrate no longer prepares data; freeze it in this checkout with tools/prepare/run.py first')
    args = ap.parse_args(argv)
    row = next((r for r in load_catalog() if r['id'] == args.case), None)
    if row is None:
        ap.error('unknown case')
    if row['resources']['budget_tier'] == 'pro':
        ap.error('Pro calibration needs the audit workflow on an exclusive 4 x H100 node')
    data = row['oj'].get('data') or {}
    if not re.fullmatch(r'[0-9a-f]{64}', str((data.get('hidden') or {}).get('sha256_plain', ''))):
        tool = 'tools/prepare_ic_spd.py' if args.case == 'IC-SPD-001' else f'tools/prepare/run.py --case {args.case}'
        ap.error(f'{args.case} has no frozen data in this checkout; prepare it first: python3 {tool} --raw-dir RAW '
                 '--key-file KEY (or ABENCH_ALLOW_SYNTHETIC=1 ... --synthetic N for an unscored plumbing check)')
    key = Path(args.hidden_key_file).expanduser().resolve()
    if not key.is_file():
        ap.error(f'hidden key not found: {key}; pass the key the data was frozen with')
    run_id = args.run_id or 'ojcal-' + time.strftime('%Y%m%d-%H%M%S-') + secrets.token_hex(3)
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        env[name] = str(row['resources']['cpus'])
    if args.image:
        env['ABENCH_JUDGE_IMAGE'] = args.image
    if data.get('synthetic'):
        # Data frozen with --synthetic is an explicit plumbing check; the judge records it as never scorable.
        env['ABENCH_ALLOW_SYNTHETIC'] = '1'
    config_path = ROOT / 'sandbox/calibration/cases.yaml'
    import yaml
    config = (yaml.safe_load(config_path.read_text()) or {}).get(args.case) or {}
    validation_device = args.device or config.get('device') or row['oj'].get('judge',{}).get('device','cpu')
    if validation_device == 'cpu' and int(row['resources']['gpus']):
        env['ABENCH_CALIBRATION_CPU'] = '1'
        env['ABENCH_JUDGE_DEVICE'] = 'cpu'
    command = ['--case', args.case, '--agent', 'oracle', '--backend', args.backend, '--hidden-key-file', str(key),
               '--result-root', str(ROOT / 'runs'), '--run-id', run_id, '--network', 'none', '--device',
               args.device or config.get('device') or row['oj'].get('judge', {}).get('device', 'cpu')]
    if args.gpus:
        command += ['--gpus', args.gpus]
    if args.allow_nonreference_gpu:
        command += ['--allow-nonreference-gpu']
    if args.allow_shared_gpu:
        command += ['--allow-shared-gpu']
    if args.image:
        command += ['--image', args.image]
    if args.backend == 'host':
        command += ['--smoke']
    from agents import run_dir
    print(json.dumps({'run': str(run_dir(ROOT / 'runs', 'oracle', None, args.case, run_id)), 'synthetic_data': bool(data.get('synthetic'))}), flush=True)
    return call('sandbox_runner.py', command, env=env)


def prepare_command(argv: list[str]) -> int:
    """Download, freeze, calibrate and activate the selected cases (tools/prepare/batch.py)."""
    sys.path.insert(0, str(ROOT / 'tools' / 'prepare'))
    import batch
    ap = argparse.ArgumentParser(prog='abench prepare', description=batch.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_filters(ap)
    ap.add_argument('--raw-root', required=True, help='raw files go to RAW_ROOT/<case>/')
    ap.add_argument('--hidden-key-file', default=str(Path.home() / '.abench/hidden.key'))
    ap.add_argument('--steps', default=','.join(batch.STEPS), help='comma-separated subset of ' + ','.join(batch.STEPS))
    ap.add_argument('--gpus', help='comma-separated GPUs for calibrating GPU cases, one case per GPU at a time')
    ap.add_argument('--jobs', type=int, default=4, help='cases downloaded and frozen at a time (default 4)')
    ap.add_argument('--allow-nonreference-gpu', action='store_true')
    ap.add_argument('--allow-unverified', action='store_true', help='passed to tools/prepare/run.py')
    ap.add_argument('--run-id', default='ojcal-prepare', help='calibration run id')
    args = ap.parse_args(argv)
    steps = [s for s in args.steps.split(',') if s]
    if unknown := set(steps) - set(batch.STEPS):
        ap.error('unknown steps: ' + ', '.join(sorted(unknown)))
    key = Path(args.hidden_key_file).expanduser()
    if {'freeze', 'calibrate'} & set(steps) and not key.is_file():
        ap.error(f'no key at {key}; create one with: python3 tools/hidden_store.py keygen {key}')
    rows, suite = selected(args)
    results = batch.prepare(rows, Path(args.raw_root).expanduser().resolve(), key, steps, gpus=args.gpus,
                            nonreference=args.allow_nonreference_gpu, allow_unverified=args.allow_unverified,
                            run_id=args.run_id, log=lambda line: print(line, file=sys.stderr, flush=True), jobs=args.jobs)
    emit({'suite': suite, 'status': dict(Counter(r['status'] for r in results)),
          'needs_operator': {r['case']: r['reason'] for r in results if r['status'] != 'done'}, 'cases': results})
    return int(any(r['status'] == 'failed' for r in results))


def jobs_command(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog='abench jobs')
    sub = ap.add_subparsers(dest='action', required=True)
    run = sub.add_parser('run')
    add_filters(run)
    run.add_argument('--agent', action='append', default=[])
    run.add_argument('--model', action='append', default=[])
    run.add_argument('--attempts', type=int, default=1)
    run.add_argument('-n', '--n-concurrent', type=int, default=1)
    run.add_argument('--max-gpu-hours', type=float)
    run.add_argument('--dry-run', action='store_true')
    run.add_argument('--allow-nonreference-gpu', action='store_true')
    run.add_argument('--allow-shared-gpu', action='store_true', help='unscored Budget validation; Pro remains exclusive')
    run.add_argument('--order', choices=('category', 'resources', 'case', 'input'), default='category')
    run.add_argument('--job-id')
    run.add_argument('--retries', type=int, default=1)
    run.add_argument('--provider-concurrency', type=int, default=1)
    for action in ('resume', 'status', 'report'):
        p = sub.add_parser(action)
        p.add_argument('job')
        if action == 'resume':
            p.add_argument('-n', '--n-concurrent', type=int)
            p.add_argument('--provider-concurrency', type=int)
            p.add_argument('--allow-shared-gpu', action='store_true', default=None)
            p.add_argument('--allow-nonreference-gpu', action='store_true', default=None)
    args, extra = ap.parse_known_args(argv)
    if args.action == 'run':
        rows, suite = selected(args)
        agents = [s for v in (args.agent or ['claude']) for s in v.split(',')]
        models = [s for v in args.model for s in v.split(',')] or [None]
        from agents import get_agent
        from agents.models import resolve
        pairs, incompatible = [], []
        for agent_name in agents:
            adapter = get_agent(agent_name)
            for model in models:
                try:
                    for row in rows:
                        adapter.options(resolve(adapter, model, oj=row['oj']))   # protocol, parameters, effort
                    pairs.append((agent_name, model))
                except ValueError as error:
                    # A harness × model matrix keeps the combinations whose protocol the harness speaks.
                    incompatible.append({'agent': agent_name, 'model': model, 'reason': str(error)})
        if not pairs:
            ap.error('no runnable harness/model combination: ' + '; '.join(x['reason'] for x in incompatible))
        scheduler_info = jobs.scheduler_docker_info(extra) if not args.dry_run else None
        daemon_cpus = ((scheduler_info or {}).get('raw') or {}).get('NCPU')
        doc = jobs.plan(rows, agents, models, args.attempts, allow_nonreference=args.allow_nonreference_gpu,
                        order=args.order, allow_shared=args.allow_shared_gpu, pairs=pairs, cpu_count=daemon_cpus)
        doc['skipped_combinations'] = incompatible
        doc.update(id=args.job_id or time.strftime('%Y%m%d-%H%M%S-') + secrets.token_hex(3), suite=suite,
                   created_at=time.time(), runner_options=extra, allow_nonreference_gpu=args.allow_nonreference_gpu,
                   allow_shared_gpu=args.allow_shared_gpu, max_gpu_hours=args.max_gpu_hours, retries=args.retries, n_concurrent=args.n_concurrent, provider_concurrency=args.provider_concurrency)
        if args.dry_run:
            emit(doc)
            return 0
        path = ROOT / 'jobs' / doc['id'] / 'job.json'
        if path.exists():
            ap.error('job already exists; use jobs resume')
        jobs.save(path, doc)
        doc = jobs.execute(doc, path, concurrency=args.n_concurrent, max_gpu_hours=args.max_gpu_hours,
                           retries=args.retries, runner_options=extra, allow_nonreference=args.allow_nonreference_gpu,
                           allow_shared=args.allow_shared_gpu, provider_concurrency=args.provider_concurrency)
    else:
        path = Path(args.job)
        if not path.is_file():
            path = ROOT / 'jobs' / args.job / 'job.json'
        doc = jobs.load(path)
        if args.action == 'resume':
            suite_cases(doc['suite'], only={e['case'] for e in doc['entries']})
            if args.allow_shared_gpu is not None:
                doc['allow_shared_gpu'] = args.allow_shared_gpu
            if args.allow_nonreference_gpu is not None:
                doc['allow_nonreference_gpu'] = args.allow_nonreference_gpu
            doc = jobs.execute(doc, path, concurrency=args.n_concurrent or doc.get('n_concurrent', 1), max_gpu_hours=doc.get('max_gpu_hours'),
                               retries=doc.get('retries', 1), runner_options=doc.get('runner_options'),
                               allow_nonreference=doc.get('allow_nonreference_gpu', False), allow_shared=doc.get('allow_shared_gpu', False),
                               provider_concurrency=args.provider_concurrency if args.provider_concurrency is not None else doc.get('provider_concurrency', 1))
        elif extra:
            ap.error('unexpected options: ' + ' '.join(extra))
    emit(jobs.report(doc) if args.action == 'report' else {'job': str(path), 'suite': doc['suite'],
         'status': dict(Counter(e['state'] for e in doc['entries'])), 'entries': doc['entries']})
    return int(any(e['state'] == 'failed' for e in doc['entries']))


def cases_command(argv: list[str]) -> int:
    """List selected cases using the shared catalog and frozen suite."""
    ap = argparse.ArgumentParser(prog='abench cases')
    ap.add_argument('action', choices=('list',))
    add_filters(ap)
    rows, suite = selected(ap.parse_args(argv))
    for row in rows:
        print('\t'.join(str(row[k]) for k in ('id', 'category', 'difficulty', 'resource_tier', 'area', 'type', 'subset')))
    from collections import Counter
    emit({'suite': suite, 'count': len(rows), 'categories': dict(Counter(r['category'] for r in rows)),
          'distribution': distribution(rows)})
    return 0


def case_command(argv: list[str]) -> int:
    """Check a case and optionally calibrate its frozen data."""
    ap = argparse.ArgumentParser(prog='abench case')
    ap.add_argument('action', choices=('check',))
    ap.add_argument('case')
    ap.add_argument('--no-calibrate', '--skip-calibration', action='store_true', help='only static checks; omit the oracle calibration')
    a = ap.parse_args(argv)
    rows = select_cases(load_catalog(), cases=a.case)
    errors = catalog_errors(rows, check_site=False)
    for tool, options in [('build_standard_cases.py', ['--case', a.case, '--check']),
                          ('validate_task_statement.py', ['--case', a.case, '--strict'])]:
        if call(tool, options):
            errors.append(tool + ' failed')
    prepare = ROOT / 'tools/prepare' / (a.case.lower().replace('-', '_') + '.py')
    if not prepare.is_file() and a.case != 'IC-SPD-001':
        errors.append('data preparation module missing')
    from egress_proxy import policy_for_case
    try:
        policy_for_case(rows[0]['oj'], 'claude')
        from container_policy import resources_for
        resources_for(rows[0]['oj'])
    except (ValueError, KeyError, RuntimeError) as e:
        errors.append(str(e))
    emit({'case': a.case, 'errors': errors})
    if errors:
        return 1
    return 0 if a.no_calibrate else calibrate([a.case])


def suites_command(argv: list[str]) -> int:
    """List frozen suites or add one immutable suite version."""
    ap = argparse.ArgumentParser(prog='abench suites', description='Frozen, versioned case suites.')
    ap.add_argument('action', choices=('list', 'freeze'))
    ap.add_argument('name', nargs='?', help='freeze: the new SUITE@MAJOR.MINOR.PATCH, e.g. all@2.0.1')
    ap.add_argument('--cases', help='freeze: comma-separated members (default: all cases for `all`, '
                                    'otherwise the members of the current default version)')
    ap.add_argument('--default', action='store_true', help='freeze: make the new version the default')
    a = ap.parse_args(argv)
    if a.action == 'freeze':
        if not a.name:
            ap.error('freeze needs SUITE@VERSION')
        emit(freeze_suite(a.name, cases=[c for c in (a.cases or '').split(',') if c] or None, default=a.default))
        return 0
    import yaml
    doc = yaml.safe_load((ROOT / 'spec/suites.yaml').read_text())
    emit({'default_versions': doc['default_versions'],
          'suites': {k: len(v['cases']) for k, v in doc['suites'].items()}})
    return 0


def gc_command(argv: list[str]) -> int:
    """Collect Docker objects using the explicit run or stale-object policy."""
    from containers import Docker
    ap = argparse.ArgumentParser(prog='abench gc', description='Remove Docker objects left by ABench runs.')
    ap.add_argument('--run-id', help='every object of this run (agent, sidecar, judge), whatever its state')
    ap.add_argument('--caches', action='store_true', help='also remove unused shared data caches (abench-data-*)')
    ap.add_argument('--min-age-minutes', type=float, default=10,
                    help='without --run-id: only touch stopped objects at least this old (default 10)')
    ap.add_argument('--discard-recovery', action='store_true', help='with --run-id: discard retained stopped collection and private staging')
    a = ap.parse_args(argv)
    if a.discard_recovery and not a.run_id:
        ap.error('--discard-recovery requires --run-id')
    from agents.lifecycle import registry_root, read_json
    retained = [read_json(path) for path in registry_root().glob('*.json')]
    matches = [record for record in retained if record.get('run_id') == a.run_id]
    if matches:
        if not a.discard_recovery:
            ap.error('run has retained collection; use recover, or explicitly --discard-recovery')
        from agents.recovery import discard, discard_registered, pending_path, run_lock
        for record in matches:
            with run_lock(record['run']):
                if pending_path(record['run']).is_file():
                    discard(record['run'])
                else:
                    discard_registered(record)
    emit(Docker().gc(a.run_id, caches=a.caches, min_age=a.min_age_minutes * 60))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('doctor', 'images', 'cases', 'case', 'suites', 'harnesses', 'prepare', 'run', 'calibrate',
                                            'judge', 'recover', 'jobs', 'gc', 'summarize'))
    args, rest = parser.parse_known_args(argv)
    if args.command == 'doctor':
        return doctor()
    if args.command == 'images':
        return call('images.py', rest)
    if args.command == 'run':
        default_key = Path.home() / '.abench/hidden.key'
        if '--hidden-key-file' not in rest and default_key.is_file():
            rest += ['--hidden-key-file', str(default_key)]
        return call('sandbox_runner.py', rest)
    if args.command == 'calibrate':
        return calibrate(rest)
    if args.command == 'prepare':
        return prepare_command(rest)
    if args.command == 'jobs':
        return jobs_command(rest)
    if args.command == 'cases':
        return cases_command(rest)
    if args.command == 'case':
        return case_command(rest)
    if args.command == 'harnesses':
        from agents import HARNESSES, get_agent
        emit([get_agent(name).capabilities() for name in HARNESSES])
        return 0
    if args.command == 'suites':
        return suites_command(rest)
    if args.command == 'judge':
        return judge_run(rest)
    if args.command == 'recover':
        ap = argparse.ArgumentParser(prog='abench recover', description='Resume stopped-container collection without running the agent again.')
        ap.add_argument('--run', required=True)
        ap.add_argument('--hidden-key-file')
        ap.add_argument('--gpus')
        ap.add_argument('--collect-only', action='store_true', help='finalize the original agent record, then judge separately')
        options = ap.parse_args(rest)
        from agents.runner import recover
        return recover(options.run, hidden_key_file=options.hidden_key_file, gpus=options.gpus, collect_only=options.collect_only)
    if args.command == 'gc':
        return gc_command(rest)
    return subprocess.call([sys.executable, str(ROOT / 'sandbox/calibration/summarize.py'), *rest])


if __name__ == '__main__':
    try:
        load_env()
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f'abench: {error}', file=sys.stderr)
        raise SystemExit(2)
