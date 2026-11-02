"""Apply the recorded agent's restrictions to an independently produced judge result."""
from __future__ import annotations


def apply_agent_attestation(evaluation: dict, agent_record: dict) -> dict:
    """Only reduce scoreability; judging can never upgrade the agent's recorded isolation or run mode."""
    sandbox = agent_record.get('sandbox') or {}
    network = agent_record.get('network') or {}
    attestation = sandbox.get('image_attestation') or {}
    evaluation['agent_isolation'] = {'kind': agent_record.get('isolation'), 'sandbox': sandbox}
    reasons = []
    if agent_record.get('run_mode') != 'benchmark':
        reasons.append('agent run was not a benchmark')
    if agent_record.get('isolation') != 'container' or sandbox.get('backend') != 'docker':
        reasons.append('agent used host_soft isolation')
    if not sandbox.get('verified') or not attestation.get('locked'):
        reasons.append('agent container or published image digest not verified')
    from agents import HARNESSES
    if agent_record.get('harness') in HARNESSES:
        cli = sandbox.get('harness_cli_version') or {}
        if (cli.get('verified') is not True or not cli.get('requested_version')
                or cli.get('actual_version') != cli.get('requested_version')):
            reasons.append('actual harness CLI version not verified')
        if not (sandbox.get('harness_bundle') or {}).get('locked'):
            reasons.append('harness CLI bundle is not the one locked on this host')
    if sandbox.get('resources_enforced') is not True:
        reasons.append('container resource limits are not enforced')
    remaining = (sandbox.get('cleanup') or {}).get('remaining') or {}
    if any(remaining.values()):
        reasons.append('agent Docker cleanup is incomplete')
    if sandbox.get('shared_gpu_validation'):
        reasons.append('shared GPU validation; exclusive GPU allocation is not guaranteed')
    if sandbox.get('validation_cpu_override'):
        reasons.append('CPU validation override of a GPU case')
    elif sandbox.get('hardware_matches') is not True and not sandbox.get('shared_gpu_validation'):
        reasons.append('agent development resource allocation is unverified or does not match the case requirements')
    # A GPU model other than the case's reference class is reported, not penalised.
    evaluation['agent_hardware'] = {'gpu_class_matches': sandbox.get('gpu_class_matches'),
                                    'gpus': [{k: g.get(k) for k in ('name', 'class', 'memory_mb')}
                                             for g in sandbox.get('gpu_inventory') or []]}
    locks = {'agent': attestation.get('lock'), 'judge': (evaluation.get('judge_dependency') or {}).get('image_lock'),
             'sidecar': ((sandbox.get('sidecar') or {}).get('image_attestation') or {}).get('lock'),
             'harness': (sandbox.get('harness_bundle') or {}).get('lock')}
    evaluation['image_locks'] = locks
    if 'local' in locks.values():
        evaluation['image_lock_note'] = ('scored with host-local locks (spec/env/images.local.lock.yaml): the images '
                                         'and harness bundles are unpublished and cannot be pulled elsewhere')
    if sandbox.get('platform') != 'linux/amd64':
        reasons.append('non-reference platform')
    if network.get('mode') not in ('none', 'egress'):
        reasons.append('open or unverified agent network')
    if network.get('mode') == 'egress':
        if not (network.get('summary') or {}).get('chain_ok'):
            reasons.append('egress log chain not verified')
        if not (sandbox.get('sidecar', {}).get('image_attestation') or {}).get('locked'):
            reasons.append('egress sidecar image has no verified published digest')
    budget = agent_record.get('budget') or {}
    if budget.get('over_budget') is True:
        reasons.append('development budget exceeded')
    elif budget.get('over_budget') is not False:
        reasons.append('development budget not attested (resource use was not metered)')
    if (agent_record.get('catalog') or {}).get('leaderboard') == 'none':
        reasons.append('pipeline calibration case')
    from agents.errors import PROVIDER_ERRORS
    if agent_record.get('error_type') in PROVIDER_ERRORS:
        reasons.append(f"model provider error ({agent_record['error_type']}): the run does not measure the agent")
    elif agent_record.get('status') != 'succeeded':
        reasons.append('agent did not finish successfully')
    if reasons:
        evaluation['scorable'] = False
        previous = evaluation.get('scoreability_reason')
        # The judge may already give the same reason (a CPU override or cap applies to both containers).
        pieces = [piece for x in [previous, *reasons] if x for piece in x.split('; ')]
        evaluation['scoreability_reason'] = '; '.join(dict.fromkeys(pieces))
    return evaluation
