"""Persistent jobs, independent run processes and hardware-aware resource allocation."""
from __future__ import annotations

import hashlib
import ast
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

from agents.errors import PROVIDER_ERRORS, RETRYABLE, retryable_judge_failure
from catalog import ROOT, TIERS, snapshot

PROVIDER_BACKOFF_BASE = 5.0

INFRA_ERRORS = ('cannot connect to the docker daemon', 'error response from daemon', 'no space left on device',
                'failed to create task', 'connection reset by peer', 'docker daemon', 'context deadline exceeded',
                'docker create failed', 'docker start failed', 'docker exec failed', 'docker cp failed',
                'docker run failed', 'docker inspect failed', 'could not select device driver', 'tls handshake timeout')


def gpu_inventory() -> list[dict]:
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in r.stdout.splitlines():
        values = [v.strip() for v in line.split(',')]
        if len(values) != 6:
            continue
        index, uuid, name, total, used, util = values
        try:
            total, used, util = float(total), float(used), float(util)
        except ValueError:
            continue
        kind = 'h100' if 'H100' in name and total >= 75000 else 'reference24' if '4090' in name and 23000 <= total <= 26000 else 'other'
        gpus.append({'index': index, 'uuid': uuid, 'name': name, 'memory_mb': total,
                     'used_memory_mb': used, 'free_memory_mb': max(0., total - used), 'utilization': util, 'class': kind,
                     'idle': used < 256 and util < 5})
    return gpus


def pipeline_limits(path: Path) -> list[dict]:
    """Read repository pipeline constants without importing case training dependencies."""
    context = {'__builtins__': {'range': range, 'int': int, 'float': float, 'list': list,
                              'tuple': tuple, 'dict': dict, 'len': len}}
    def value(node):
        for item in ast.walk(node):
            if isinstance(item, ast.Attribute) and item.attr not in ('values', 'keys', 'items'):
                raise ValueError('nonconstant attribute')
            if isinstance(item, ast.Name) and item.id.startswith('__'):
                raise ValueError('nonconstant name')
        return eval(compile(ast.Expression(node), str(path), 'eval'), context)
    def assign(target, val):
        if isinstance(target, ast.Name):
            context[target.id] = val
        elif isinstance(target, (ast.List, ast.Tuple)):
            for child, item in zip(target.elts, val):
                assign(child, item)
    def statements(body):
        for node in body:
            try:
                if isinstance(node, ast.Assign):
                    val = value(node.value)
                    for target in node.targets:
                        assign(target, val)
                elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
                    context[node.target.id] += value(node.value)
                elif isinstance(node, ast.For):
                    for item in value(node.iter):
                        assign(node.target, item)
                        statements(node.body)
            except (ValueError, NameError, TypeError, KeyError, AttributeError):
                # Some dictionaries hold training Config() instances. Their literal keys are enough for
                # comprehensions over evaluation settings; never construct the training objects on the host.
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                    try:
                        val = {value(key): None for key in node.value.keys}
                        for target in node.targets:
                            assign(target, val)
                    except (ValueError, NameError, TypeError, KeyError, AttributeError):
                        pass
                continue
    statements(ast.parse(path.read_text()).body)
    pipeline = context.get('PIPELINE')
    if not pipeline or any('timeout' not in stage for stage in pipeline):
        raise ValueError(f'{path}: cannot determine pipeline time limits')
    return pipeline


def evaluation_hours(row: dict, root: Path = ROOT) -> float:
    judge = row['oj'].get('judge') or {}
    pipeline = pipeline_limits(root / 'cases' / row['id'] / judge.get('stages', 'harness/stages.py'))
    seconds = sum(float(stage['timeout']) + float(stage.get('tail_slack', 5)) for stage in pipeline)
    return seconds * max(1, len(judge.get('seeds') or [])) * (2 if judge.get('baseline') else 1) / 3600


def estimate(row: dict) -> dict:
    resources, budget = row['resources'], row['oj'].get('agent_budget') or {}
    count = int(resources['gpus'])
    verify = evaluation_hours(row)
    return {'reference_gpu_hours': float(budget.get('gpu_hours', 0)) + verify * count if count and row['resource_tier'] != 'pro' else 0.,
            'h100_hours': float(budget.get('h100_hours', 0)) + verify * count if row['resource_tier'] == 'pro' else 0.,
            'cpu_core_hours': float(budget.get('cpu_core_hours', 0)) + verify * int(resources['cpus']) if not count else 0.}


def available_for_case(row: dict, gpu: dict, allow_shared: bool = False) -> bool:
    """Keep formal allocation idle; explicit Budget validation can use 24 GiB of free low-load memory."""
    resources = row['resources']
    if not int(resources['gpus']):
        return True
    if allow_shared and resources['budget_tier'] == 'pro':
        return False
    if gpu.get('idle', False):
        return True
    if not allow_shared or resources['budget_tier'] == 'pro' or resources.get('gpu_class') != 'reference24':
        return False
    return float(gpu.get('free_memory_mb', 0)) >= 24576 and float(gpu.get('utilization', float('inf'))) <= 10


def compatible_gpus(row: dict, inventory: list[dict], allow_nonreference: bool = False,
                    allow_shared: bool = False) -> list[str]:
    resources = row['resources']
    if not int(resources['gpus']):
        return []
    # Any GPU model is scored and reported (Budget and Pro); using another model than the case's class is explicit.
    return [g['index'] for g in inventory if available_for_case(row, g, allow_shared) and
            (g['class'] == resources['gpu_class'] or allow_nonreference)]


def skip_reason(row: dict, inventory: list[dict], *, allow_nonreference: bool = False,
                allow_shared: bool = False, cpu_count: int | None = None) -> str | None:
    """Why this machine can never run the case. A compatible GPU that is busy right now is not a reason: the job
    waits for it (or leaves the entry pending for resume)."""
    inventory = [dict(gpu, idle=True) for gpu in inventory]
    resources = row['resources']
    if int(resources['cpus']) > (cpu_count or os.cpu_count() or 1):
        return f'needs {resources["cpus"]} CPU cores'
    n = int(resources['gpus'])
    if allow_shared and resources['budget_tier'] == 'pro':
        return 'Pro requires an exclusive node; shared GPU validation is unsupported'
    available = compatible_gpus(row, inventory, allow_nonreference, allow_shared)
    if n and len(available) < n:
        return f'needs {n} {resources["gpu_class"]} GPU(s); this machine has {len(available)}'
    if resources['budget_tier'] == 'pro' and len(inventory) != 4:
        return 'Pro requires an exclusive node with 4 GPUs'
    return None


def plan(rows: list[dict], agents: list[str], models: list[str | None], attempts: int = 1,
         inventory: list[dict] | None = None, allow_nonreference: bool = False, order: str = 'category',
         allow_shared: bool = False, pairs: list[tuple[str, str | None]] | None = None,
         cpu_count: int | None = None) -> dict:
    """pairs, when given, are the harness × model combinations to run (e.g. only protocol-compatible ones)."""
    if inventory is None:
        inventory = gpu_inventory() if any(int(row['resources']['gpus']) for row in rows) else []
    if attempts < 1:
        raise ValueError('attempts must be positive')
    if order == 'category':                 # the five categories in turn, each from low to high resources
        from catalog import category_names
        names = category_names()
        rows = sorted(rows, key=lambda r: (names.index(r['category']), TIERS.index(r['resource_tier']),
                                           int(r['resources']['gpus']), r['id']))
    elif order == 'resources':
        rows = sorted(rows, key=lambda r: (TIERS.index(r['resource_tier']), int(r['resources']['gpus']), r['id']))
    elif order == 'case':
        rows = sorted(rows, key=lambda r: r['id'])
    entries = []
    totals = dict.fromkeys(('reference_gpu_hours', 'h100_hours', 'cpu_core_hours'), 0.)
    for row in rows:
        reason = skip_reason(row, inventory, allow_nonreference=allow_nonreference, allow_shared=allow_shared,
                             cpu_count=cpu_count)
        estimated = estimate(row)
        for agent, model in (pairs if pairs is not None else [(a, m) for a in agents for m in models]):
            for attempt in range(1, attempts + 1):
                identity = json.dumps([row['id'], agent, model, attempt], separators=(',', ':'))
                entry = {'id': hashlib.sha256(identity.encode()).hexdigest()[:16], 'case': row['id'], 'agent': agent,
                         'model': model, 'attempt': attempt, 'catalog': snapshot(row), 'resources': row['resources'],
                         'estimate': dict(estimated), 'state': 'skipped' if reason else 'pending', 'skip_reason': reason,
                         'retries': 0}
                entries.append(entry)
                for k, v in entry['estimate'].items():
                    totals[k] += v
    runnable = {k: sum(e['estimate'][k] for e in entries if e['state'] != 'skipped') for k in totals}
    return {'entries': entries, 'estimate': totals, 'runnable_estimate': runnable, 'hardware': inventory,
            'allow_shared_gpu': allow_shared}


def run_path(root: Path, document: dict, entry: dict) -> Path:
    """Run directory of an entry's next attempt. Run records are immutable, so a retry after an infrastructure
    failure and a restart after an interrupted attempt each get a new directory; the old one is kept for audit."""
    suffix = ('-retry' + str(entry['retries']) if entry.get('retries') else '') + \
             ('-restart' + str(entry['restarts']) if entry.get('restarts') else '')
    category = (entry.get('catalog') or {}).get('category')
    if category:
        from agents import run_label
        name = 'job-' + document['id'] + '-' + entry['id'] + suffix
        return root / 'runs' / run_label(entry['agent'], entry.get('model')) / category / entry['case'] / name
    from agents import run_dir
    return run_dir(root / 'runs', entry['agent'], entry.get('model'), entry['case'], 'job-' + document['id'] + '-' + entry['id'] + suffix)


def _alive(pid: int | None, marker: str) -> bool:
    """Whether a runner started by an earlier job process (e.g. one that was SIGKILLed) is still executing."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    r = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    return marker in r.stdout


def load(path: Path) -> dict:
    from job_store import load as replay
    return replay(path)


def save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(document, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def infrastructure_failure(returncode: int, log: str, record: dict | None = None) -> bool:
    """A Docker or daemon failure of the runner itself (from its own log, not the agent's output)."""
    if not returncode:
        return False
    if record and record.get('error_type'):                 # the run classified its own failure
        return record['error_type'] == 'infrastructure'
    return any(message in log.lower() for message in INFRA_ERRORS)


def failure_kind(run: Path, record: dict | None) -> str | None:
    """error_type of a finished run (agent_error, timeout, provider_* ...), or of its failed model preflight."""
    if record:
        return record.get('error_type')
    preflight = run / 'preflight.json'
    try:
        if preflight.is_file():
            return json.loads(preflight.read_text()).get('error_type')
        attempt = run / 'attempt.json'
        return ((json.loads(attempt.read_text()).get('failure') or {}).get('kind') if attempt.is_file() else None)
    except ValueError:
        return None


def hidden_key_file(runner_options: list[str] | None) -> str:
    options = runner_options or []
    if '--hidden-key-file' in options[:-1]:
        return options[options.index('--hidden-key-file') + 1]
    return str(Path.home() / '.abench/hidden.key')


def after_judging(entry: dict, state: str | None, retries: int) -> str | None:
    """A judge_failed entry is judged again (state rejudge) while it has retries left."""
    return 'rejudge' if state == 'judge_failed' and entry.get('judge_retries', 0) < retries else state


def finished_state(run: Path, record: dict | None, evaluation: dict | None) -> tuple[str | None, str | None]:
    """Derive the same outcome after a runner exits and when recovering its durable records.

    An evaluation can exist after a provider failure; it does not turn that failure into an agent result.
    judge_failed: the submission was collected but the infrastructure failed judging it, so it can be judged again.
    None means there is no finished outcome yet, so an interrupted attempt keeps the restart path.
    """
    kind = failure_kind(run, record)
    if record is None and (run / 'collection_pending.json').is_file():
        return 'recollect', kind
    if kind in PROVIDER_ERRORS:
        return 'provider_error', kind
    if retryable_judge_failure(evaluation):
        return 'judge_failed', kind
    if evaluation is not None:
        return 'completed', kind
    if record and record.get('evaluation_deferred'):
        return 'awaiting_judge', kind
    if record and not kind and (run / 'attempt.json').is_file():
        attempt = json.loads((run / 'attempt.json').read_text())
        if attempt.get('phase') in ('judge', 'awaiting_judge'):
            return 'judge_failed', kind
    if kind or record and record.get('status') in ('failed', 'timeout'):
        return 'failed', kind
    return None, kind


def _run_records(run: Path) -> tuple[dict | None, dict | None]:
    """Read durable outcomes in the same order for recovery and normal process completion."""
    record = json.loads((run / 'agent_run.json').read_text()) if (run / 'agent_run.json').is_file() else None
    evaluation = json.loads((run / 'evaluation.json').read_text()) if (run / 'evaluation.json').is_file() else None
    return record, evaluation


def _finish_entry(entry: dict, returncode: int, retries: int) -> None:
    """Classify a finished child and schedule only the retries allowed by its recorded outcome."""
    run = Path(entry['run'])
    record, evaluation = _run_records(run)
    attempt_path = run / 'attempt.json'
    attempt = json.loads(attempt_path.read_text()) if attempt_path.is_file() else None
    # Structured failures are authoritative for new runs; only legacy runners need traceback fallback.
    if attempt is not None:
        infrastructure = (attempt.get('failure') or {}).get('kind') == 'infrastructure'
    else:
        with Path(entry['log']).open('rb') as stream:
            stream.seek(0, 2); stream.seek(max(0, stream.tell() - 32768))
            details = stream.read().decode(errors='replace') + '\n' + str((evaluation or {}).get('error', ''))
        infrastructure = infrastructure_failure(returncode, details, record)
    state, kind = finished_state(run, record, evaluation)
    entry['error_type'] = kind
    if state == 'recollect':
        entry['state'] = 'collection_pending' if entry.get('action') == 'collect' else 'recollect'
    elif state == 'judge_failed':
        # Only judging failed, on the infrastructure: never rerun the agent for it.
        entry['state'] = after_judging(entry, state, retries)
    elif (entry.get('action') != 'judge' and (infrastructure or kind in RETRYABLE)
          and entry['retries'] < retries):
        entry['retries'] += 1
        entry['state'] = 'pending'
    else:
        entry['state'] = state or 'failed'
    entry.update(returncode=returncode, finished_at=time.time())


class ResourcePool:
    """Reserve declared slots, or the daemon node when containers receive its full CPU/RAM allowance."""
    def __init__(self, inventory: list[dict], allow_nonreference: bool = False, lock_dir: Path | None = None,
                 *, allow_shared: bool = False, docker_info: dict | None = None):
        self.inventory = inventory
        self.allow_nonreference = allow_nonreference
        self.allow_shared = allow_shared
        raw = (docker_info or {}).get('raw') or {}
        self.cpu_total = int(raw.get('NCPU') or os.cpu_count() or 1)
        self.node = None
        self.exclusive_node = False
        if docker_info is not None:
            from runtime.policy import policy
            if int(raw.get('NCPU') or 0) <= 0:
                raise ValueError('Docker daemon CPU capacity is required for resource leases')
            self.exclusive_node = bool(policy().get('host_scale') and raw.get('NCPU') and raw.get('MemTotal'))
            if not raw.get('ID'):
                raise ValueError('Docker daemon identity is required for resource leases')
            self.node = hashlib.sha256(str(raw['ID']).encode()).hexdigest()
        self.cpu_used = 0
        self.leases = {}
        self.lock_dir = lock_dir or Path.home() / '.abench/gpu-locks'

    def acquire(self, key: str, row: dict) -> list[str] | None:
        import fcntl
        resources = row['resources']
        cpus, n = int(resources['cpus']), int(resources['gpus'])
        if cpus > self.cpu_total:
            return None
        if self.exclusive_node:
            cpus = self.cpu_total
        if self.allow_shared and resources['budget_tier'] == 'pro':
            return None
        if self.cpu_used + cpus > self.cpu_total:
            return None
        allocated = {g for lease in self.leases.values() for g in lease['gpus']}
        available = [g for g in compatible_gpus(row, self.inventory, self.allow_nonreference, self.allow_shared) if g not in allocated]
        handles, chosen = [], []
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        with ExitStack() as rollback:
            if self.node is not None:
                handle = rollback.enter_context((self.lock_dir / ('node-' + self.node + '.lock')).open('a'))
                mode = fcntl.LOCK_EX if self.exclusive_node or resources['budget_tier'] == 'pro' else fcntl.LOCK_SH
                try:
                    fcntl.flock(handle, mode | fcntl.LOCK_NB)
                    handles.append(handle)
                except BlockingIOError:
                    return None
            for gpu in available:
                if len(chosen) == n:
                    break
                handle = rollback.enter_context((self.lock_dir / ('gpu-' + gpu + '.lock')).open('a'))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    handles.append(handle)
                    chosen.append(gpu)
                except BlockingIOError:
                    handle.close()
            if len(chosen) != n:
                return None
            self.cpu_used += cpus
            self.leases[key] = {'gpus': chosen, 'cpus': cpus, 'handles': handles}
            rollback.pop_all()                   # successful lease owns these handles until release()
            return chosen

    def release(self, key: str) -> None:
        lease = self.leases.pop(key)
        self.cpu_used -= lease['cpus']
        for handle in lease['handles']:
            handle.close()


def provider_key(endpoint):
    if not endpoint:
        return None
    credential = os.environ.get(endpoint['key_env'], '')
    identity = [endpoint.get('base_url'), endpoint['key_env'], hashlib.sha256(credential.encode()).hexdigest()]
    return hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()


def _cooldown(document, entry):
    if entry.get('error_type') not in RETRYABLE:
        return
    delay = min(300.0, PROVIDER_BACKOFF_BASE * 2 ** min(10, max(0, entry.get('retries', 0) - 1)))
    eligible = time.time() + delay
    if entry.get('state') == 'pending':
        entry['next_eligible_at'] = eligible
    if entry.get('provider_key'):
        values = document.setdefault('provider_cooldowns', {})
        values[entry['provider_key']] = max(values.get(entry['provider_key'], 0), eligible)


def _provider_active(active, key):
    count = 0
    for _, _, entry in active.values():
        if entry.get('provider_key') != key or entry.get('action') in ('judge', 'collect'):
            continue
        attempt = Path(entry['run']) / 'attempt.json'
        if attempt.is_file():
            phase = json.loads(attempt.read_text()).get('phase')
            if phase in ('collection', 'collected', 'finalize', 'judge', 'awaiting_judge'):
                continue
        count += 1
    return count


def scheduler_docker_info(runner_options):
    """Discover the execution daemon once; a host smoke job never needs Docker."""
    options = runner_options or []
    backend = 'docker'
    for index, option in enumerate(options):
        if option == '--backend' and index + 1 < len(options):
            backend = options[index + 1]
        elif option.startswith('--backend='):
            backend = option.split('=', 1)[1]
    if backend == 'host':
        return None
    from containers import Docker
    return Docker().info()


def execute(document: dict, path: Path, *, concurrency: int = 1, max_gpu_hours: float | None = None,
            retries: int = 1, runner_options: list[str] | None = None, root: Path = ROOT,
            allow_nonreference: bool = False, allow_shared: bool = False, provider_concurrency: int = 1) -> dict:
    if provider_concurrency < 1:
        raise ValueError('provider concurrency must be positive')
    if concurrency < 1:
        raise ValueError('concurrency must be positive')
    from catalog import load_catalog
    rows = {r['id']: r for r in load_catalog(root)}
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    job_lock = (path.parent / 'job.lock').open('a')
    try:
        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        job_lock.close()
        raise RuntimeError(f'job {document["id"]} is already being executed by another process')
    from job_store import Journal
    journal = Journal(path, document)
    document['allow_shared_gpu'] = allow_shared
    document['allow_nonreference_gpu'] = allow_nonreference
    document['provider_concurrency'] = provider_concurrency
    document.pop('waiting_for_resources', None)
    active = {}
    spent = float(document.get('committed_gpu_hours', 0))
    for entry in document['entries']:
        if entry['state'] == 'running':
            run = Path(entry['run'])
            if _alive(entry.get('pid'), run.name):
                job_lock.close()
                raise RuntimeError(f'{run.name} is still executing (pid {entry["pid"]}); wait for it or stop it first')
        elif entry['state'] == 'pending':
            run = run_path(root, document, entry)
        else:
            run = Path(entry.get('run') or run_path(root, document, entry))
        attempt = run / 'attempt.json'
        if attempt.is_file():
            phase = json.loads(attempt.read_text())
            if phase.get('status') == 'running' and _alive(phase.get('pid'), run.name):
                job_lock.close()
                raise RuntimeError('the original runner is still executing; wait before resuming its job')
        record, evaluation = _run_records(run)
        state, kind = finished_state(run, record, evaluation)
        if state:
            entry.update(state=after_judging(entry, state, retries), error_type=kind)
        elif entry['state'] in ('running', 'pending'):
            if run.exists() and any(run.iterdir()):
                # An interrupted attempt left a partial, immutable record: keep it and restart elsewhere.
                entry.setdefault('interrupted_runs', []).append(str(run))
                entry['restarts'] = entry.get('restarts', 0) + 1
                run = run_path(root, document, entry)
            entry['state'] = 'pending'
        entry['run'] = str(run)
    journal.checkpoint(document)
    try:
        info = scheduler_docker_info(runner_options) if any(e['state'] in ('pending', 'rejudge', 'recollect') for e in document['entries']) else None
        pool = ResourcePool([], allow_nonreference, allow_shared=allow_shared, docker_info=info)
    except BaseException:
        job_lock.close()
        raise
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    previous_term = signal.signal(signal.SIGTERM, terminate)
    try:
        while True:
            finished = []
            for key, (process, log, entry) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                log.close()
                pool.release(key)
                del active[key]
                _finish_entry(entry, rc, retries)
                _cooldown(document, entry)
                finished.append(entry)
            if finished:
                journal.update(document, finished)
            pending = [e for e in document['entries'] if e['state'] in ('pending', 'rejudge', 'recollect')]
            if (len(active) < concurrency and pool.cpu_used < pool.cpu_total
                    and any(int(rows[e['case']]['resources']['gpus']) for e in pending)):
                pool.inventory = gpu_inventory()
            changed = []
            cooldowns = []
            for entry in pending:
                if len(active) >= concurrency or pool.cpu_used >= pool.cpu_total:
                    break
                row = rows[entry['case']]
                # rejudge: judge the collected submission again in the same run directory, without the agent
                judge_again = entry['state'] == 'rejudge'
                collecting = entry['state'] == 'recollect'
                from agents import get_agent
                from agents.models import resolve
                endpoint = resolve(get_agent(entry['agent']), entry.get('model'), oj=row['oj'])
                entry['provider_key'] = provider_key(endpoint)
                eligible = max(entry.get('next_eligible_at', 0), (document.get('provider_cooldowns') or {}).get(entry['provider_key'], 0))
                if not judge_again and not collecting and eligible > time.time():
                    cooldowns.append(eligible)
                    continue
                if entry['provider_key'] and not judge_again and not collecting and _provider_active(active, entry['provider_key']) >= provider_concurrency:
                    continue
                if endpoint and not judge_again and not collecting and not os.environ.get(endpoint['key_env']):
                    entry.update(state='skipped', skip_reason='missing credential variable ' + endpoint['key_env'])
                    changed.append(entry)
                    continue
                reason = skip_reason(row, pool.inventory, allow_nonreference=allow_nonreference, allow_shared=allow_shared,
                                     cpu_count=pool.cpu_total)
                if reason:
                    entry.update(state='judge_failed' if judge_again else 'collection_pending' if collecting else 'skipped', skip_reason=reason)
                    changed.append(entry)
                    continue
                cost = 0 if judge_again or collecting else entry['estimate']['reference_gpu_hours'] + entry['estimate']['h100_hours']
                if max_gpu_hours is not None and spent + cost > max_gpu_hours:
                    entry.update(state='skipped', skip_reason='max GPU hours would be exceeded')
                    changed.append(entry)
                    continue
                assigned = pool.acquire(entry['id'], row)
                if assigned is None:
                    continue
                if collecting:
                    run = Path(entry['run'])
                    entry['collection_retries'] = entry.get('collection_retries', 0) + 1
                    name = entry['id'] + '-collect-' + str(entry['collection_retries'])
                elif judge_again:
                    run = Path(entry['run'])
                    entry['judge_retries'] = entry.get('judge_retries', 0) + 1
                    name = entry['id'] + '-' + str(entry['retries']) + '-judge' + str(entry['judge_retries'])
                else:
                    run = run_path(root, document, entry)
                    entry['run'] = str(run)
                    name = entry['id'] + '-' + str(entry['retries'])
                log_path = path.parent / 'logs' / (name + '.log')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                entry['log'] = str(log_path)
                log = log_path.open('ab')
                if collecting:
                    command = [sys.executable, str(root / 'tools/abench.py'), 'recover', '--run', str(run),
                               '--hidden-key-file', hidden_key_file(runner_options)]
                    command += ['--gpus', ','.join(assigned)] if assigned else []
                elif judge_again:
                    command = [sys.executable, str(root / 'tools/abench.py'), 'judge', '--run', str(run),
                               '--hidden-key-file', hidden_key_file(runner_options)]
                    command += ['--gpus', ','.join(assigned)] if assigned else []
                else:
                    command = [sys.executable, str(root / 'tools/sandbox_runner.py'), '--case', entry['case'], '--agent', entry['agent'],
                               '--run-id', run.name, '--result-root', str(root / 'runs'), '--attempt', str(entry['attempt']),
                               '--suite', document['suite']]
                    if entry.get('model'):
                        command += ['--model', entry['model']]
                    if assigned:
                        command += ['--gpus', ','.join(assigned)]
                    if allow_nonreference:
                        command += ['--allow-nonreference-gpu']
                    if allow_shared:
                        command += ['--allow-shared-gpu']
                    command += runner_options or []
                    if '--hidden-key-file' not in command and (Path.home() / '.abench/hidden.key').is_file():
                        command += ['--hidden-key-file', str(Path.home() / '.abench/hidden.key')]
                try:
                    entry.update(state='running', action='collect' if collecting else 'judge' if judge_again else 'run',
                                 gpus=assigned, started_at=time.time(), command=command, pid=None)
                    spent += cost
                    document['committed_gpu_hours'] = spent
                    journal.update(document, [entry])     # durable launch intent before a child can create records
                    process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                except OSError:
                    log.close()
                    pool.release(entry['id'])
                    spent -= cost
                    document['committed_gpu_hours'] = spent
                    entry['state'] = 'recollect' if collecting else 'rejudge' if judge_again else 'pending'
                    journal.update(document, [entry])
                    raise
                entry.update(state='running', action='collect' if collecting else 'judge' if judge_again else 'run', gpus=assigned,
                             started_at=time.time(), command=command, pid=process.pid)
                active[entry['id']] = (process, log, entry)
                journal.update(document, [entry])
            if changed:
                journal.update(document, changed)
            if not active:
                if cooldowns:
                    document['waiting_for_provider_until'] = min(cooldowns)
                    journal.update(document)
                    time.sleep(min(.2, max(0, min(cooldowns) - time.time())))
                    continue
                if any(e['state'] in ('pending', 'rejudge', 'recollect') for e in document['entries']):
                    # Nothing could start: GPUs are busy or leased by another ABench job. Leave durable
                    # pending work for resume.
                    document['waiting_for_resources'] = True
                break
            time.sleep(.2)
    except BaseException:
        for process, log, entry in active.values():
            process.terminate()
        for key, (process, log, entry) in active.items():
            try:
                process.wait(timeout=180)      # time for the child's shielded Docker cleanup
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
            pool.release(key)
            entry['state'] = 'recollect' if entry.get('action') == 'collect' else 'rejudge' if entry.get('action') == 'judge' else 'pending'
        raise
    finally:
        # A failure preparing a log or command can occur after acquire(), before active owns the lease.
        for key in list(pool.leases):
            pool.release(key)
        signal.signal(signal.SIGTERM, previous_term)
        document['updated_at'] = time.time()
        journal.checkpoint(document)
        job_lock.close()
    return document


def report(document: dict) -> dict:
    groups = {k: defaultdict(lambda: {'verdicts': Counter(), 'results': []})
              for k in ('category', 'difficulty_resources', 'area', 'harness_model')}
    excluded = []
    for entry in document['entries']:
        if entry['catalog']['leaderboard'] == 'none':
            excluded.append(entry['case'])
            continue
        evaluation = Path(entry.get('run', '')) / 'evaluation.json'
        result = json.loads(evaluation.read_text()) if evaluation.is_file() else {}
        verdict = result.get('verdict', entry['state'])
        cat = entry['catalog']
        keys = {'category': cat.get('category') or 'unknown',
                'difficulty_resources': cat['difficulty'] + '/' + cat['resource_tier'], 'area': cat['area'],
                'harness_model': entry['agent'] + '/' + (entry.get('model') or 'default')}
        for axis, key in keys.items():
            group = groups[axis][key]
            group['verdicts'][verdict] += 1
            primary = result.get('primary') or {}
            metric = primary.get('metric')
            metrics = result.get('metrics') or {}
            baseline_metrics = (result.get('baseline') or {}).get('metrics') or {}
            score, base = metrics.get(metric), baseline_metrics.get(metric)
            relative = (score / base if primary.get('goal') == 'max' else base / score) if score and base else None
            group['results'].append({'case': entry['case'], 'run': entry.get('run'), 'primary': primary,
                                     'primary_ci95': result.get('primary_ci95'), 'relative_to_baseline': relative,
                                     'metrics': result.get('metrics'), 'aggregate': result.get('aggregate'),
                                     'baseline': result.get('baseline'), 'scorable': result.get('scorable', False),
                                     'gpus': {'agent': [g.get('name') for g in (result.get('agent_hardware') or {}).get('gpus') or []],
                                              'judge': [g.get('name') for g in result.get('judge_gpu_inventory') or [] if g]}})
    return {'suite': document['suite'], 'status': dict(Counter(e['state'] for e in document['entries'])),
            'groups': groups, 'excluded_calibration': sorted(set(excluded)), 'estimate': document.get('estimate'),
            'skipped': [{'case': e['case'], 'reason': e['skip_reason']} for e in document['entries'] if e['state'] == 'skipped']}
