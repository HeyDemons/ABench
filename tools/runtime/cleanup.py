"""Best-effort deletion with evidence of successful and remaining Docker objects."""
from __future__ import annotations

from contextlib import contextmanager
import signal
import threading


@contextmanager
def shielded(restore=None):
    """Let cleanup run to completion: SIGINT and SIGTERM that arrive meanwhile are held, and the first is delivered
    once cleanup is done (like asyncio.shield), so stopping a run never leaves its Docker objects behind.
    restore: handlers to install afterwards instead of the ones in place before (a caller's own handlers that end
    with the cleanup), so the held signal reaches the handler that is current after it."""
    if threading.current_thread() is not threading.main_thread():
        yield                                   # signal handlers exist only in the main thread
        return
    held = []
    previous = {sig: signal.signal(sig, lambda number, frame: held.append(number))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in {**previous, **(restore or {})}.items():
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)
        if held:
            signal.raise_signal(held[0])


@contextmanager
def sigterm_interrupts():
    """SIGTERM raises KeyboardInterrupt, as Ctrl-C does, so a process stopped by a job still runs the finally blocks
    that remove its Docker objects (by default SIGTERM ends Python without running them)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    def interrupt(number, frame):
        raise KeyboardInterrupt(f'signal {number}')
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


@shielded()
def stop_container(docker, container):
    """The attach/exec client exiting is not evidence that container processes stopped."""
    from runtime.transport import DockerError
    state = docker.inspect(container).get('State') or {}
    if state.get('Running') is True:
        docker.kill(container)
        state = docker.inspect(container).get('State') or {}
    if state.get('Running') is not False:
        raise DockerError('container is still running or its stopped state is unverified; refusing to collect mutable artifacts')
    return state


@shielded()
def cleanup_objects(docker, containers, volumes, record):
    uncertain = getattr(docker, 'pending_containers', [])
    if isinstance(uncertain, list):
        containers.extend(name for name in uncertain if name not in containers)
    report = record.setdefault('cleanup', {'containers': 0, 'volumes': 0,
                                         'removed': {'containers': [], 'volumes': []},
                                         'failed': [], 'remaining': {'containers': [], 'volumes': []}})
    pending = {}
    failures = []
    for kind, objects, remove in (('containers', containers, docker.remove),
                                  ('volumes', volumes, docker.remove_volume)):
        pending[kind] = []
        for identifier in reversed(list(objects)):
            try:
                result = remove(identifier)
                if result.returncode:
                    message = getattr(result, 'stderr', '') or f'exit {result.returncode}'
                    raise RuntimeError(str(message).strip())
                if (kind == 'containers' and isinstance(uncertain, list) and identifier in uncertain
                        and 'no such container' in str(getattr(result, 'stderr', '') or '').lower()):
                    # `docker rm -f` succeeds for an absent name; after a timed-out create the daemon may still
                    # be creating it, so absence now is not evidence that it is gone.
                    raise RuntimeError('creation timed out; container not present yet')
            except Exception as exc:
                pending[kind].append(identifier)
                failures.append({'kind': kind, 'id': identifier, 'error': str(exc)[:1000]})
            else:
                if identifier not in report['removed'][kind]:
                    report['removed'][kind].append(identifier)
        report[kind] = len(report['removed'][kind])
    report['failed'] = failures
    report['remaining'] = pending
    containers[:] = pending['containers']
    volumes[:] = pending['volumes']
    if isinstance(uncertain, list):
        uncertain[:] = [name for name in uncertain if name in pending['containers']]
    return report
