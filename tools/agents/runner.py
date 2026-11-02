"""Orchestrate view construction, environment execution, collection and independent judging."""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import yaml
from agents import get_agent, run_dir
from agents.models import resolve
from agents.workspace import resource_limits
from runtime.agent import DockerAgentEnvironment, HostSoftEnvironment
from runtime.cleanup import shielded


def verify_harness_version(adapter, args, environment, env, result):
    """Verify the installed executable before authentication or metering, retaining failed setup evidence."""
    probe = adapter.version_probe(args)
    if probe is None:
        return None
    logs = result / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stdout, stderr = logs / 'harness_version_stdout.log', logs / 'harness_version_stderr.log'
    # Never expose credentials to the version subprocess, or allow it to inherit the model's proxy/endpoint.
    version_env = {k: v for k, v in env.items() if k in ('HOME', 'TMPDIR', 'PATH', 'LANG', 'LC_ALL', 'CODEX_HOME')}
    record = {'harness': adapter.name, 'requested_version': adapter.requested_cli_version,
              'actual_version': None, 'verified': False, 'argv': probe.argv,
              'status': 'unavailable', 'exit_code': None}
    failure = None
    try:
        status, code = environment.execute(probe, version_env, 30, stdout, stderr)
        record.update(status=status, exit_code=code)
        if status != 'succeeded' or code != 0:
            failure = 'version command did not finish successfully'
        else:
            record['actual_version'] = adapter.parse_cli_version(stdout.read_bytes())
            if record['actual_version'] is None:
                failure = 'unrecognized version output'
            elif record['actual_version'] != record['requested_version']:
                failure = 'installed CLI does not match the requested version'
            else:
                record['verified'] = True
    except (OSError, RuntimeError, ValueError) as exc:
        failure = 'version command could not be verified (' + type(exc).__name__ + ')'
    for stream, path in (('stdout', stdout), ('stderr', stderr)):
        record[stream + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if failure:
        record['error'] = failure
    (result / 'harness_version.json').write_text(json.dumps(record, indent=2) + '\n')
    if failure:
        raise ValueError(f"{adapter.name} CLI version verification failed: {failure}; "
                         f"requested {record['requested_version']}, actual {record['actual_version']}")
    return record


def _resolve_endpoint(args, adapter, oj):
    """Resolve and validate model overrides before preparing a run."""
    endpoint = resolve(adapter,args.model,args.base_url,args.allow_private_model_endpoint,oj)
    if args.claude_model:
        if args.harness != 'claude' or endpoint is None:
            raise ValueError('--claude-model requires the Claude harness')
        endpoint['model'] = args.claude_model
        endpoint['model_override_source'] = '--claude-model'
    if getattr(args,'effort',None):
        if endpoint is None:
            raise ValueError('--effort requires a harness with a model')
        endpoint['parameters'] = {**(endpoint.get('parameters') or {}),'effort':args.effort}
        endpoint['effort_source'] = '--effort'
    # Reject unsupported options before building views, copying indexes or preparing images.
    adapter.options(endpoint)
    if adapter.name == 'claude' and getattr(args, 'claude_effort', None):
        adapter.check_effort(endpoint, args.claude_effort)
    if endpoint and not os.environ.get(endpoint['key_env']) and not args.dry_run:
        raise ValueError('missing model credential environment variable: '+endpoint['key_env'])
    return endpoint


def _select_gpus(args, oj, shared_gpu):
    """Validate the requested allocation and retain its hardware evidence."""
    from jobs import gpu_inventory, available_for_case
    ids = str(args.gpus or '').split(',') if args.gpus else []
    inventory = {g['index']:g for g in gpu_inventory()}
    resources = oj['resources']
    if len(ids) != int(resources['gpus']) or len(set(ids)) != len(ids) or any(i not in inventory for i in ids):
        raise ValueError('supply distinct --gpus matching the case resource count')
    if any(not available_for_case({'resources':resources},inventory[i],shared_gpu) for i in ids):
        raise ValueError('selected GPU lacks sufficient free low-load resources; sharing requires --allow-shared-gpu')
    selected_gpus = [inventory[i] for i in ids]
    # The GPU model does not affect scoring; it is recorded and reported with the result.
    gpu_class_matches = all(g['class'] == resources['gpu_class'] for g in selected_gpus)
    hardware_matches = not shared_gpu
    if resources['budget_tier'] == 'pro':
        hardware_matches = hardware_matches and len(inventory) == 4
    if not gpu_class_matches and not args.allow_nonreference_gpu:
        raise ValueError('non-reference GPU requires --allow-nonreference-gpu (scored; the GPU model is reported)')
    return selected_gpus, hardware_matches, gpu_class_matches


def _close_environment(environment, result):
    """Keep cleanup evidence without masking an error already in flight."""
    pending_error = sys.exc_info()[0] is not None
    try:
        environment.close()
    except Exception:
        if not pending_error:
            raise
    finally:
        # Setup can fail before agent_run.json exists. Keep the cleanup outcome independently, and
        # never let writing this diagnostic replace the original authentication/transport failure.
        state = getattr(environment,'record',None)
        if isinstance(state,dict) and state.get('cleanup') is not None and result.is_dir():
            try:
                (result/'cleanup.json').write_text(json.dumps(state['cleanup'],indent=2)+'\n')
            except (OSError,TypeError,ValueError):
                pass


def _finalize_run(result, context, collected, sandbox, key, attempt, *, judge=True):
    import sandbox_runner as s
    import oj_eval
    import egress_proxy
    from catalog import check_run_lock
    from types import SimpleNamespace
    args = SimpleNamespace(**context['args'])
    adapter = get_agent(args.harness)
    oj, endpoint, net, attestation = (context[name] for name in ('oj','endpoint','net','attestation'))
    view, cutoff, papers, run_config, catalog = (context[name] for name in ('view','cutoff','papers','run_config','catalog'))
    restrictions, recorded_argv, passed, cli_version = (context[name] for name in ('restrictions','recorded_argv','passed','cli_version'))
    budget, gpu, status, code, wall, started, mode = (context[name] for name in ('budget','gpu','status','code','wall','started','mode'))
    logs = result / 'logs'
    attempt.phase('finalize')
    from agents.native_logs import collect_native_logs
    credential = os.environ.get(endpoint['key_env'], '') if endpoint else ''
    matches = not endpoint or hashlib.sha256(credential.encode()).hexdigest() == context['credential_sha256']
    if (result/'agent_run.json').exists():
        raise ValueError('agent run record already exists and is immutable')
    if matches:
        native_logs = collect_native_logs(collected.get('native'), result/'native_logs', [credential])
    else:
        native_logs = {'files': [], 'withheld': [{'path': '*', 'reason': 'recovery needs the original credential to inspect native logs'}], 'bytes': 0}
    if collected.get('egress_log') and collected['egress_log'].is_file():
        shutil.copyfile(collected['egress_log'],result/'egress_log.jsonl')
        net['summary'] = egress_proxy.summarize_log(result/'egress_log.jsonl')
        sandbox.setdefault('sidecar',{})['log_sha256'] = net['summary']['log_sha256']
    if (result/'submission').exists():
        shutil.rmtree(result/'submission')  # only an uncommitted prior finalization; agent_run.json is absent
    kept,dropped = s.collect_submission(collected['submission'],result/'submission',oj)
    retrieval = s.collect_retrieval_log(collected['retrieval_log'],result/'retrieval_log.jsonl')
    cpu = context["host_cpu_seconds"]
    if args.backend == 'host' and 'host_cpu_baseline' in context:
        current = os.times()
        cpu = current.children_user + current.children_system - context['host_cpu_baseline']
    if args.backend == 'docker':
        # Host-side children are only the docker CLI; without the container cgroup, CPU use is unknown.
        usage, baseline = sandbox.get('cpu_usage_usec'), sandbox.get('cpu_baseline_usec')
        cpu = max(0,usage-baseline)/1e6 if usage is not None and baseline is not None else None
    out,err = (logs/'agent_stdout.log').read_bytes(),(logs/'agent_stderr.log').read_bytes()
    trajectory = adapter.summarize(out,err)
    native = adapter.native_summary(result/'native_logs')
    if trajectory is not None and native:
        # The harness's own session logs are complete where its stdout was truncated, and name the models.
        trajectory['native'] = native
        if not trajectory.get('actual_models'):
            trajectory['actual_models'] = native['actual_models']
    terminal = (trajectory or {}).get('result') or {}
    if terminal.get('is_error') or str(terminal.get('subtype','')).startswith('error'):
        status = 'failed' if status == 'succeeded' else status
    from agents.errors import run_error_type
    error_type = run_error_type(status,trajectory)
    if endpoint and trajectory:
        endpoint['actual_models'] = trajectory.get('actual_models',[])
    record = {'schema_version':'2.0','case_id':args.case,'run_id':args.run_id,'run_mode':mode,
        'provenance':'smoke_real' if args.smoke else 'benchmark_real','runner_version':'sandbox-runner-v3-docker',
        'harness':args.harness,'isolation':'container' if args.backend == 'docker' else 'host_soft',
        'harness_cli_version':cli_version,
        'wrapper_attestation':{'kind':'container' if args.backend == 'docker' else 'host_soft','scoreable':bool(sandbox.get('verified') and attestation.get('locked')), 'accepted':bool(sandbox.get('verified')),'reason':'Docker policy attestation replaces wrapper policy','policy_version':'abench.container_policy/v1','launcher':'docker' if args.backend == 'docker' else None,'policy':None,'wrapper_id':'docker' if args.backend == 'docker' else 'host_soft'},
        'status':status,'error_type':error_type,'provider_error':(trajectory or {}).get('provider_error'),'exit_code':code,'agent_wall_seconds':round(wall,3),'agent_child_cpu_seconds':None if cpu is None else round(cpu,3),
        'started_at':started,'finished_at':s.now(),'network':net,'cleanup':sandbox.get('cleanup'),
        'host':context['host'],'agent_view':view,
        'prompt_sha256':context['prompt_sha256'],'passed_env_names':passed,'agent_env_names':context['agent_env_names'],
        'argv':recorded_argv,'launch_files':context['launch_files'],'harness_restrictions':restrictions,
        'harness_capabilities':context['harness_capabilities'] if 'harness_capabilities' in context else adapter.capabilities(),'native_logs':native_logs,'submission':{'kept':kept,'dropped_self_reported_or_disallowed':dropped,
          'tree_sha256':oj_eval.tree_hash(result/'submission')},'knowledge_cutoff':cutoff,'paper_index':papers,
        'retrieval_log':retrieval,'stdout_sha256':s.sha256_bytes(out),'stderr_sha256':s.sha256_bytes(err),
        'trajectory':trajectory,'gpu_use':gpu,'budget':s.budget_check(budget,gpu,cpu),
        'sandbox':sandbox,'model_endpoint':endpoint,'run_config':run_config,'catalog':catalog,'evaluation_deferred':bool(args.defer_evaluation)}
    (result/'submission.sha256').write_text(''.join(f'{s.sha256_file(result/"submission"/k)}  submission/{k}\n' for k in kept))
    from agents.lifecycle import atomic_json
    if (result/'agent_run.json').exists():
        raise ValueError('agent run record already exists and is immutable')
    atomic_json(result/'agent_run.json', record)
    attempt.phase('judge')
    if args.defer_evaluation or not judge:
        evaluation = {'verdict':'not_evaluated','scorable':False,'note':'--defer-evaluation'}
    else:
        try:
            check_run_lock(record)
            evaluation = oj_eval.evaluate(args.case,result/'submission',key,mode,backend=args.backend,gpus=args.gpus,
                                          run_id=args.run_id)
        except (SystemExit,RuntimeError,ValueError,OSError,subprocess.SubprocessError) as exc:
            from agents.errors import judge_failure
            # failure.retryable: the infrastructure failed, so jobs judge this submission again (abench judge)
            evaluation = {'schema_version':'2.0','case_id':args.case,'verdict':'judge_error','metrics':None,'scorable':False,
                          'error':str(exc),'failure':judge_failure(exc)}
    from agents.scoring import apply_agent_attestation
    apply_agent_attestation(evaluation,record)
    evaluation['agent_run_sha256'] = s.sha256_file(result/'agent_run.json')
    if not args.defer_evaluation and judge:
        (result/'evaluation.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2))
    elif not judge and not args.defer_evaluation:
        attempt.phase('awaiting_judge')
    print(json.dumps({'run_id':args.run_id,'result_dir':str(result),'agent_status':status,'verdict':evaluation.get('verdict')},indent=2))
    return 0 if status == 'succeeded' and evaluation.get('verdict') in ('evaluated','judge_pending','smoke_unscored','not_evaluated') else 1


def recover(result, *, hidden_key_file=None, gpus=None, collect_only=False):
    """Finish collection/finalization of this exact execution; no agent command or metering is invoked."""
    from agents import recovery
    from agents.lifecycle import Attempt, atomic_json
    from catalog import check_run_lock
    from runtime.cleanup import sigterm_interrupts
    import sandbox_runner as s
    result = Path(result).expanduser().resolve()
    with recovery.run_lock(result), sigterm_interrupts():
        context, environment, retained = recovery.restore(result)
        check_run_lock({'case_id': context['args']['case'], 'run_config': context['run_config']})
        attempt = Attempt(result, context['args']['case'], context['args']['run_id'], resume=True)
        original_error = None
        try:
            attempt.phase('collection')
            artifacts = recovery.collected_paths(result)
            if artifacts is None:
                artifacts = environment.collect(native=get_agent(context['args']['harness']).session_logs)
                recovery.collected(result, environment, artifacts)
            environment.close()
            recovery.cleanup_state(result, environment)
            atomic_json(result / 'cleanup.json', environment.record.get('cleanup') or {})
            if (result/'agent_run.json').exists():
                # The previous process committed its immutable result just before it was interrupted.
                recovery.finish(result, retained)
                if not (result/'evaluation.json').exists() and not context['args']['defer_evaluation'] and not collect_only:
                    from abench import judge_run
                    options = ['--run', str(result)]
                    if hidden_key_file:
                        options += ['--hidden-key-file', str(hidden_key_file)]
                    if gpus:
                        options += ['--gpus', gpus]
                    return judge_run(options)
                return 0
            key = None
            if not collect_only and not context['args']['defer_evaluation'] and context['mode'] == 'benchmark':
                key = s.validate_hidden_key(str(hidden_key_file or Path.home() / '.abench/hidden.key'))
            if gpus is not None:
                context['args']['gpus'] = gpus     # judging allocation only; original agent run_config stays fixed
            return _finalize_run(result, context, artifacts, environment.record, key, attempt, judge=not collect_only)
        except BaseException as error:
            original_error = error
            raise
        finally:
            keep = not (result/'agent_run.json').is_file()
            with shielded():
                if not keep:
                    environment.close()
                    recovery.finish(result, retained)
                attempt.finish(error=original_error, pending=keep, cleanup=environment.record.get('cleanup'))


def _recovery_credentials(args, endpoint):
    """Pass-env configuration values must not rewrite the saved run mode or status."""
    from redact_records import _sensitive
    names = {name for name in args.pass_env if _sensitive(name)}
    if endpoint:
        names.add(endpoint['key_env'])
    return [value for name in names if len(value := os.environ.get(name, '')) >= 8]


def run(args):
    import sandbox_runner as s
    import oj_eval
    import egress_proxy
    from catalog import case_lock, check_run_lock, snapshot, suite_cases
    from images import resolve_image, image_attestation
    oj = yaml.safe_load((s.ROOT / 'cases' / args.case / 'case/oj.yaml').read_text())
    limit = float(oj['resources']['wall_clock_limit_h'])*3600
    args.timeout_seconds = args.timeout_seconds if args.timeout_seconds is not None else limit
    if not 0 < args.timeout_seconds <= limit:
        raise ValueError('timeout must be positive and within the declared session wall-clock limit')
    # Only this case's frozen identity is checked: a change to another case does not block this run.
    selected,args.suite = suite_cases(args.suite,only={args.case})
    locked_case = case_lock(args.case)
    args.device = args.device or (oj.get('judge') or {}).get('device','cpu')
    args.run_id = args.run_id or time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',args.run_id):
        raise ValueError('invalid run-id')
    if args.backend == 'host' and not args.smoke:
        raise ValueError('host backend requires --smoke')
    adapter = get_agent(args.harness)
    args.network = args.network or ('egress' if adapter.needs_network else 'none')
    if args.backend == 'host' and args.network == 'none':
        # Host smoke cannot enforce a network namespace.
        args.network = 'open'
    endpoint = _resolve_endpoint(args, adapter, oj)
    selected_gpus = []
    hardware_matches = int(oj.get('resources',{}).get('gpus',0)) == 0
    gpu_class_matches = True
    shared_gpu = bool(getattr(args,'allow_shared_gpu',False))
    if shared_gpu and oj['resources']['budget_tier'] == 'pro':
        raise ValueError('Pro runs require exclusive hardware')
    pol = egress_proxy.policy_for_case(oj,args.harness,endpoint) if args.network == 'egress' else None
    if args.backend == 'docker' and int(oj.get('resources',{}).get('gpus',0)) and os.environ.get('ABENCH_CALIBRATION_CPU') != '1':
        selected_gpus, hardware_matches, gpu_class_matches = _select_gpus(args, oj, shared_gpu)
    key = s.validate_hidden_key(args.hidden_key_file) if args.hidden_key_file else None
    mode = 'smoke' if args.smoke else 'benchmark'
    judge_kind = (oj.get('judge') or {}).get('kind')
    if mode == 'benchmark' and judge_kind not in (None,'pending') and key is None and not args.defer_evaluation:
        raise ValueError('benchmark requires --hidden-key-file or --defer-evaluation')
    result = run_dir(args.result_root or s.ROOT / 'runs', args.harness, endpoint['name'] if endpoint else None, args.case, args.run_id)
    if result.exists() and any(result.iterdir()):
        raise ValueError('run record already exists and is immutable: '+str(result))
    from contextlib import nullcontext
    from agents.recovery import run_lock
    with (nullcontext() if args.dry_run else run_lock(result)):
        if not args.dry_run and any(path.name != '.operator.lock' for path in result.iterdir()):
            raise ValueError('run record already exists and is immutable: ' + str(result))
        from agents.lifecycle import Attempt
        attempt = Attempt(result, args.case, args.run_id) if not args.dry_run else None
        pending = None
        original_error = None
        tmp = Path(tempfile.mkdtemp(prefix='abench-agent-'))
        paths = {'root':tmp,'view':tmp/'view','workspace':tmp/'workspace','control':tmp/'control'}
        for k,rel in {'home':'.home','tmp':'.tmp','model':'model_store','checkpoint':'checkpoint','submission':'submission',
                      'retrieval_log':'.abench/retrieval_log.jsonl'}.items():
            paths[k] = paths['workspace']/rel
        for k in ('workspace','home','tmp','model','checkpoint','submission','control'):
            paths[k].mkdir(parents=True,exist_ok=True)
        paths['retrieval_log'].parent.mkdir(parents=True,exist_ok=True)
        environment = None
        old_signals = {}
        try:
            view = s.build_agent_view(args.case,oj,paths['view'],copy_data=args.backend != 'docker')
            paths['public_sources'] = view.get('public_sources',[])
            papers = s.paper_index_info(args.papers_index)
            if papers:
                paths['papers'] = tmp/'papers'; paths['papers'].mkdir()
                exclude = (oj.get('agent_visible') or {}).get('paper_exclude') or []
                if exclude:
                    s.filtered_paper_index(papers['path'],exclude,paths['papers']/'current.sqlite')
                else:
                    shutil.copyfile(papers['path'],paths['papers']/'current.sqlite')
            mdir = args.models_dir or (oj.get('data') or {}).get('models_dir')
            if mdir:
                p = Path(mdir).expanduser()
                paths['models'] = p if p.is_absolute() else s.ROOT/p
                if not paths['models'].is_dir():
                    raise ValueError('frozen model directory missing: '+str(paths['models']))
            control = paths['control']
            for src,name in ((s.TOOLS/'egress_proxy.py','egress_proxy.py'),(s.TOOLS/'runtime/forwarder.py','forwarder.py'),
                             (s.TOOLS/'agents/oracle.py','oracle.py'),(s.ROOT/'sandbox/mock_agent.sh','mock_agent.sh'),
                             (s.TOOLS/'agents/secret_config.py','secret_config.py')):
                shutil.copyfile(src,control/name)
            if pol:
                (control/'policy.json').write_text(json.dumps(pol))
            if args.harness == 'oracle':
                shutil.copytree(s.ROOT/'cases'/args.case/'example_submission',control/'example_submission',
                                ignore=shutil.ignore_patterns('__pycache__','checkpoint'))
                cal = yaml.safe_load((s.ROOT/'sandbox/calibration/cases.yaml').read_text()) or {}
                (control/'oracle.json').write_text(json.dumps({'mode':oj.get('mode'),'calibration':cal.get(args.case)}))
            image = args.image
            attestation = {'locked':False}
            if args.backend == 'docker':
                # Local images can exercise the pipeline, but never establish published-image attestation.
                # Every agent runs in the case's profile image (the task environment); a harness CLI is its bundle.
                image = image or resolve_image(oj['environment'],allow_unlocked=True)
                attestation = image_attestation(image,oj['environment'])
                sidecar_image = resolve_image('egress',allow_unlocked=True) if pol else None
                environment = DockerAgentEnvironment(args,paths,oj,image,sidecar_image,pol,
                                                     harness=args.harness if adapter.protocols else None)
            else:
                environment = HostSoftEnvironment(args,paths,policy=pol)
            runpaths = environment.paths
            env, passed = s.agent_env(args,runpaths)
            env['PATH'] = environment.default_path() if args.backend == 'docker' else os.environ.get('PATH','')
            env['ABENCH_RETRIEVAL_LOG'] = str(runpaths['retrieval_log'])
            if papers:
                env['ABENCH_PAPERS_INDEX'] = '/abench/papers/current.sqlite' if args.backend == 'docker' else str(paths['papers']/'current.sqlite')
            if mdir:
                env['ABENCH_MODELS_DIR'] = '/abench/models' if args.backend == 'docker' else str(paths['models'])
            cutoff = oj.get('knowledge_cutoff')
            if cutoff:
                env['ABENCH_CUTOFF'] = str(cutoff)
            budget = oj.get('agent_budget') or {}
            if budget.get('gpu_hours') is not None:
                env['ABENCH_DEV_BUDGET_GPU_HOURS'] = str(budget['gpu_hours'])
            if endpoint:
                # The alias exists only in subprocess environment, never in argv/record.
                env[endpoint['key_env']] = os.environ.get(endpoint['key_env'],'')
                for name in adapter.credential_env(endpoint):
                    env[name] = env[endpoint['key_env']]
            if pol:
                env.update({k:'http://127.0.0.1:3128' for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy')})
                env.update(NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1')
                env.update(adapter.proxy_env('http://127.0.0.1:3128'))
            # Build prompt using host-readable view, then substitute only runner-owned paths.
            prompt = s.agent_prompt(paths,args.network)
            if args.backend == 'docker':
                for name in ('submission','checkpoint','workspace','view'):
                    prompt = prompt.replace(str(paths[name]),str(runpaths[name]))
            if args.smoke:
                prompt += '\nThis is an unscored smoke run: prioritize a complete baseline submission, self-check, report and references within the deadline before further experiments.\n'
            launch = adapter.launch(args,runpaths,prompt,endpoint)
            if args.backend == 'host':
                launch.argv[:] = [str(control/x.removeprefix('/abench/control/')) if x.startswith('/abench/control/') else x for x in launch.argv]
            # A harness that takes the prompt as an argument (zcode) records its hash, not the text, among the argv.
            recorded_argv = ['<prompt sha256:'+s.sha256_bytes(prompt.encode())+'>' if x == prompt else x for x in launch.argv]
            catalog = snapshot(selected[0])
            run_config = {'case':args.case,'harness':args.harness,'harness_version':adapter.version,
                          'requested_cli_version':adapter.requested_cli_version,'actual_cli_version':None,
                          'model':endpoint['name'] if endpoint else None,
                          'harness_options':{'effort':args.claude_effort or ((endpoint or {}).get('parameters') or {}).get('effort'),
                                             'max_budget_usd':args.claude_max_budget_usd},
                          'attempt':args.attempt,'image_digest':image,'suite':args.suite,
                          'gpus':args.gpus,'allow_shared_gpu':shared_gpu}
            # What this run is judged with (checked again before judging) and what ran the agent.
            run_config['run_lock'] = {'schema':'abench.run_lock/v1','case':locked_case,
                                      'agent':{'harness':args.harness,'harness_version':adapter.version,
                                               'requested_cli_version':adapter.requested_cli_version,'image':image,
                                               'model':endpoint['name'] if endpoint else None,
                                               'base_url':endpoint.get('base_url') if endpoint else None}}
            restrictions = {'disabled_provider_tools':list(adapter.disabled_tools)}
            plan = {'case_id':args.case,'run_id':args.run_id,'run_mode':mode,'argv':recorded_argv,'launch_files':sorted(launch.files),
                    'harness_restrictions':restrictions,
                    'agent_view':view,'model_endpoint':endpoint,'run_config':run_config,'catalog':catalog}
            if args.dry_run:
                print(json.dumps(plan,indent=2)); return 0
            result.mkdir(parents=True, exist_ok=True)
            (result/'run_lock.json').write_text(json.dumps(run_config['run_lock'],ensure_ascii=False,indent=2))
            logs = result/'logs'; logs.mkdir(exist_ok=True)
            def interrupted(sig, frame):
                raise KeyboardInterrupt(f'signal {sig}')
            for sig in (signal.SIGINT,signal.SIGTERM):
                old_signals[sig] = signal.signal(sig,interrupted)
            attempt.phase('environment_setup')
            sandbox = environment.start()
            sandbox['hardware_matches'] = hardware_matches
            sandbox['gpu_class_matches'] = gpu_class_matches
            sandbox['shared_gpu_validation'] = shared_gpu
            sandbox['gpu_inventory'] = selected_gpus
            sandbox['image_attestation'] = attestation
            # Recorded for the operator only; the agent's prompt and environment stay as on main.
            allocation = resource_limits(oj, sandbox, args.timeout_seconds)
            run_config['resource_limits'] = allocation
            (result/'resource_limits.json').write_text(json.dumps(allocation,ensure_ascii=False,indent=2) + '\n')
            net = {'mode':args.network,'policy':pol}
            if pol:
                net.update(policy_version=pol['policy_version'],policy_sha256=pol['policy_sha256'],
                           case_deny=pol['case_deny'],proxy_port=3128,upstream=bool(os.environ.get('ABENCH_EGRESS_UPSTREAM')),
                           firewall={'enforced':bool(sandbox.get('verified')),'method':'container network and Unix socket policy'})
            # Harness configuration files (never secrets) go into the agent's HOME before anything runs.
            environment.write_files(launch.files)
            attempt.phase('harness_identity')
            cli_version = verify_harness_version(adapter,args,environment,env,result)
            if cli_version:
                sandbox['harness_cli_version'] = cli_version
                run_config['actual_cli_version'] = cli_version['actual_version']
            attempt.phase('authentication')
            # Authentication is outside development time and budget.
            if endpoint:
                check = adapter.preflight(endpoint)
                if check:
                    status,code = environment.execute(check,env,60,logs/'preflight_stdout.log',logs/'preflight_stderr.log')
                    if status != 'succeeded':
                        from agents.preflight import preflight_failure
                        failure = preflight_failure((logs/'preflight_stdout.log').read_bytes())
                        # Jobs read this to retry transient provider failures instead of counting a failed run.
                        (result/'preflight.json').write_text(json.dumps(failure,indent=2))
                        raise ValueError(f"model preflight failed ({failure['error_type']}): {failure['message'][:200]}")
            attempt.phase('agent')
            env['ABENCH_DEADLINE'] = str(int(time.time()+args.timeout_seconds))   # integer Unix seconds, as on main
            if hasattr(environment,'begin_meter'):
                environment.begin_meter()
            started = s.now(); t0 = time.perf_counter(); c0 = os.times()
            meter = s.GpuMeter(gpu_ids=args.gpus or []).start()
            try:
                status,code = environment.execute(launch,env,args.timeout_seconds,logs/'agent_stdout.log',logs/'agent_stderr.log')
            finally:
                gpu = meter.stop()
                gpu['shared_allocation'] = shared_gpu
            wall = time.perf_counter()-t0
            if args.backend == 'docker':
                environment.prepare_collection()  # freeze background work before operator diagnostics or fsync
            context = {'args': {name: getattr(args, name) for name in ('case','run_id','backend','harness','smoke','gpus','defer_evaluation')},
                       'oj': oj, 'endpoint': endpoint, 'net': net, 'attestation': attestation, 'view': view,
                       'cutoff': cutoff, 'papers': papers, 'run_config': run_config, 'catalog': catalog,
                       'restrictions': restrictions, 'recorded_argv': recorded_argv, 'passed': passed,
                       'cli_version': cli_version, 'budget': budget, 'gpu': gpu, 'status': status, 'code': code,
                       'wall': wall, 'started': started, 'mode': mode, 'prompt_sha256': s.sha256_bytes(prompt.encode()),
                       'agent_env_names': sorted(env), 'launch_files': sorted(launch.files), 'harness_capabilities': adapter.capabilities(),
                       'host': {'platform': platform.platform(), 'python': platform.python_version()},
                       'host_cpu_seconds': None, 'host_cpu_baseline': c0.children_user+c0.children_system,
                       'credential_sha256': hashlib.sha256(os.environ.get(endpoint['key_env'], '').encode()).hexdigest() if endpoint else None}
            attempt.phase('collection')
            if args.backend == 'docker':
                from agents.recovery import save_execution
                pending = save_execution(result, context, environment, secrets=_recovery_credentials(args, endpoint))
            collected = environment.collect(native=adapter.session_logs)
            if pending:
                from agents.recovery import collected as commit_collection
                commit_collection(result, environment, collected)
            environment.close()
            if pending:
                from agents.recovery import cleanup_state
                cleanup_state(result, environment)
                pending['record'] = sandbox
            attempt.phase('collected')
            return _finalize_run(result, context, collected, sandbox, key, attempt)
        except BaseException as error:
            original_error = error
            raise
        finally:
            # A second Ctrl-C or SIGTERM waits until the cleanup is done; it then reaches the caller's handlers.
            with shielded(restore=old_signals):
                retained = pending is not None and not (result/'agent_run.json').is_file()
                try:
                    if environment and not retained:
                        _close_environment(environment, result)
                    if not retained:
                        if pending:
                            from agents.recovery import finish
                            pending['record'] = environment.record
                            finish(result, pending)
                        s.make_writable(paths['view'])
                        shutil.rmtree(tmp,ignore_errors=True)
                finally:
                    if attempt:
                        cleanup = (getattr(environment, 'record', {}) or {}).get('cleanup') if environment else None
                        attempt.finish(error=original_error or sys.exc_info()[1], pending=retained, cleanup=cleanup)
