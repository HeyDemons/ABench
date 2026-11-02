"""Convert Claude stream-json to Harbor's text/tool ATIF v1.8 schema.

Schema reference: https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md
The raw stream remains authoritative. Missing timestamps, prompts and tool outputs are never invented.
"""
from __future__ import annotations
import hashlib
import json
import re
from typing import Any


def _text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    return '\n'.join(str(b.get('text', '')) for b in content if isinstance(b, dict) and b.get('type') == 'text')


def _metrics(usage):
    result = {}
    if not isinstance(usage, dict):
        return result
    if 'input_tokens' in usage:
        result['prompt_tokens'] = sum(int(usage.get(k, 0) or 0) for k in
                                      ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'))
    if 'output_tokens' in usage:
        result['completion_tokens'] = int(usage['output_tokens'] or 0)
    if 'cache_read_input_tokens' in usage:
        result['cached_tokens'] = int(usage['cache_read_input_tokens'] or 0)
    return result


def claude_to_atif(raw: bytes, version: str, *, prompt: str | None = None, session_id: str | None = None) -> dict:
    """One assistant message ID becomes one agent step; tool observations attach to their originating step."""
    steps, calls, messages, seen_blocks = [], {}, {}, {}
    models, result_event = set(), {}
    invalid_lines = 0

    def append(source, message, **values):
        step = {'step_id':len(steps) + 1, 'source':source, 'message':message, **values}
        steps.append(step)
        return step

    if prompt is not None:
        append('user', prompt)
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            invalid_lines += 1
            continue
        if not isinstance(event, dict):
            invalid_lines += 1
            continue
        session_id = session_id or event.get('session_id')
        kind = event.get('type')
        if kind == 'result':
            result_event = event
            continue
        message = event.get('message') or {}
        if not isinstance(message, dict):
            continue
        content = message.get('content', [])
        if kind == 'assistant':
            mid = message.get('id')
            step = messages.get(mid) if mid else None
            if step is None:
                step = append('agent', '')
                if mid:
                    messages[mid] = step
                seen_blocks[step['step_id']] = set()
            if message.get('model'):
                step['model_name'] = str(message['model'])
                models.add(step['model_name'])
            if event.get('parent_tool_use_id'):
                step.setdefault('extra', {})['parent_tool_use_id'] = event['parent_tool_use_id']
            if message.get('usage'):
                step['metrics'] = _metrics(message['usage'])
            if isinstance(content, str):
                step['message'] = content
                continue
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                marker = json.dumps(block, sort_keys=True)
                if marker in seen_blocks[step['step_id']]:
                    continue
                seen_blocks[step['step_id']].add(marker)
                if block.get('type') == 'text':
                    step['message'] += ('\n' if step['message'] else '') + str(block.get('text', ''))
                elif block.get('type') == 'thinking':
                    step['reasoning_content'] = step.get('reasoning_content', '') + str(block.get('thinking', ''))
                elif block.get('type') == 'tool_use':
                    cid = str(block.get('id') or f"step-{step['step_id']}-call-{len(calls) + 1}")
                    args = block.get('input', {})
                    call = {'tool_call_id':cid, 'function_name':str(block.get('name', 'unknown')),
                            'arguments':args if isinstance(args, dict) else {'value':args}}
                    step.setdefault('tool_calls', []).append(call)
                    calls[cid] = step
        elif kind == 'user':
            text = _text(content)
            if text:
                append('user', text)
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict) or block.get('type') != 'tool_result':
                    continue
                cid = block.get('tool_use_id')
                target = calls.get(cid)
                observation = {'content':_text(block.get('content', ''))}
                if target:
                    observation['source_call_id'] = cid
                else:
                    target = append('system', 'Tool result without an originating call in this stream.')
                    observation['extra'] = {'original_call_id':cid}
                if block.get('is_error'):
                    observation.setdefault('extra', {})['is_error'] = True
                target.setdefault('observation', {'results':[]})['results'].append(observation)
    if not steps:
        append('system', str(result_event.get('result') or 'No message events were present in the raw stream.'))
    agent = {'name':'claude-code', 'version':version}
    if len(models) == 1:
        agent['model_name'] = next(iter(models))
    metrics = {'total_steps':len(steps)}
    for name, value in _metrics(result_event.get('usage')).items():
        metrics[{'prompt_tokens':'total_prompt_tokens', 'completion_tokens':'total_completion_tokens',
                 'cached_tokens':'total_cached_tokens'}[name]] = value
    if result_event.get('total_cost_usd') is not None:
        metrics['total_cost_usd'] = result_event['total_cost_usd']
    return {'schema_version':'ATIF-v1.8', 'session_id':session_id or 'sha256:' + hashlib.sha256(raw).hexdigest(),
            'agent':agent, 'steps':steps, 'final_metrics':metrics,
            'notes':'Converted from recorded Claude stream-json; unrecorded input and timestamps are omitted.',
            'extra':{'raw_sha256':hashlib.sha256(raw).hexdigest(), 'invalid_json_lines':invalid_lines}}


def summarize_stream(out: bytes) -> dict[str, Any] | None:
    """Summary of a `claude -p --output-format stream-json` trajectory: tool calls by name, Bash commands, result."""
    tools: dict[str, int] = {}
    bash: list[str] = []
    result: dict[str, Any] | None = None
    events = 0
    for line in out.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        events += 1
        if e.get("type") == "assistant":
            for c in (e.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    tools[c.get("name", "?")] = tools.get(c.get("name", "?"), 0) + 1
                    if c.get("name") == "Bash" and len(bash) < 400:
                        bash.append(str((c.get("input") or {}).get("command", ""))[:300])
        elif e.get("type") == "result":
            result = {k: e.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms", "total_cost_usd")}
            result["final_text"] = str(e.get("result", ""))[-2000:]
    if not events:
        return None
    return {"events": events, "tool_calls": tools, "bash_commands": bash, "result": result}


# ---------------------------------------------------------------- other harnesses
# Each summary has the fields of summarize_stream plus usage, actual_models and provider_error. result.is_error is
# true when the harness reported a failed run, which makes the runner record the agent as failed. provider_error is
# the last model-provider failure the harness reported (status, message, error_type); terminal means nothing
# happened after it, i.e. it ended the run or the harness was still retrying it when stopped.

def provider_error(status, message, terminal=False):
    from .errors import classify
    message = str(message or '')[:500]
    return {'status': status, 'message': message, 'error_type': classify(status, message), 'terminal': terminal}


class _Summary:
    def __init__(self):
        self.events, self.tools, self.bash, self.models = 0, {}, [], set()
        self.usage, self.turns, self.error, self.final, self.finished = {}, 0, None, '', False
        self.provider, self.truncated = None, 0

    def provider_failure(self, status, message):
        self.provider = provider_error(status, message, terminal=True)

    def tool(self, name, command=None):
        self.tools[name] = self.tools.get(name, 0) + 1
        if command is not None and len(self.bash) < 400:
            self.bash.append(str(command)[:300])

    def add_usage(self, values):
        for key, value in (values or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.usage[key] = self.usage.get(key, 0) + value

    def rows(self, raw):
        for line in raw.splitlines():
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(event, dict):
                self.events += 1
                self.truncated += event.get('truncated') is True    # the harness cut a long field of this event
                if self.provider:                    # a later event means the provider error did not end the run
                    self.provider['terminal'] = False
                yield event

    def result(self):
        if not self.events:
            return None
        failed = bool(self.error) or not self.finished
        return {'events': self.events, 'truncated_events': self.truncated, 'tool_calls': self.tools,
                'bash_commands': self.bash, 'usage': self.usage, 'actual_models': sorted(self.models),
                'provider_error': self.provider,
                'result': {'subtype': 'error' if failed else 'success', 'is_error': failed, 'num_turns': self.turns,
                           'error': self.error or (None if self.finished else 'no terminal event'),
                           'final_text': self.final[-2000:]}}


def summarize_codex(raw: bytes, stderr: bytes = b'') -> dict[str, Any] | None:
    """`codex exec --json`: thread/turn/item events; usage on turn.completed. Provider failures arrive as `error`
    events ("Reconnecting... (unexpected status 401 ...)") and a final turn.failed."""
    s = _Summary()
    for e in s.rows(raw):
        kind, item = e.get('type'), e.get('item') or {}
        if kind == 'turn.started':
            s.turns += 1
        elif kind == 'turn.completed':
            s.add_usage(e.get('usage'))
            s.finished = True
        elif kind == 'error':                         # "Reconnecting... 1/5 (...)": the turn may still succeed
            s.provider_failure(None, e.get('message'))
        elif kind == 'turn.failed':
            s.error = str((e.get('error') or {}).get('message') or kind)[:500]
            s.provider_failure(None, s.error)
        elif kind == 'item.completed':
            if item.get('type') == 'agent_message':
                s.final = str(item.get('text', ''))
            elif item.get('type') == 'command_execution':
                s.tool('command_execution', item.get('command'))
            elif item.get('type') == 'mcp_tool_call':
                s.tool(f"mcp:{item.get('server')}/{item.get('tool')}")
            elif item.get('type') in ('file_change', 'web_search', 'todo_list'):
                s.tool(item['type'])
    return s.result()


def summarize_gemini(raw: bytes, stderr: bytes = b'') -> dict[str, Any] | None:
    """`gemini --output-format stream-json`: init/message/tool_use/tool_result/error/result events. API failures are
    in the result's error; retries in progress are only on stderr ("Attempt N failed with status 429")."""
    s = _Summary()
    reply = []
    for e in s.rows(raw):
        kind = e.get('type')
        if kind == 'init' and e.get('model'):
            s.models.add(e['model'])
        elif kind == 'message' and e.get('role') == 'assistant':
            reply.append(str(e.get('content', '')))
        elif kind == 'tool_use':
            reply = []
            name = e.get('tool_name', '?')
            s.tool(name, (e.get('parameters') or {}).get('command') if name == 'run_shell_command' else None)
            s.turns += 1
        elif kind == 'result':                         # error events are non-fatal warnings; result decides
            stats = e.get('stats') or {}
            s.add_usage({k: v for k, v in stats.items() if k in ('input_tokens', 'output_tokens', 'total_tokens', 'cached')})
            s.models.update((stats.get('models') or {}).keys())
            s.finished = True
            if e.get('status') != 'success':
                s.error = str((e.get('error') or {}).get('message') or e.get('status'))[:500]
                s.provider_failure(None, s.error)
    s.final = ''.join(reply)
    retries = re.findall(r'^Attempt \d+ failed with status (\d{3})\. Retrying with backoff\.\.\. (.*)$',
                         stderr.decode('utf-8', 'replace'), re.MULTILINE)
    if retries and not s.finished:
        s.provider = provider_error(int(retries[-1][0]), retries[-1][1], terminal=True)
    return s.result()


DSH_CODES = {'AUTH': 401, 'QUOTA': 402, 'RATE_LIMIT': 429, 'SERVER': 500, 'MODEL_NOT_FOUND': 404}


def summarize_dsh(raw: bytes, stderr: bytes = b'') -> dict[str, Any] | None:
    """`dsh --profile headless --json`: session/status/tool_call/tool_result/text/final events. A failed turn ends with
    reason "error"; the cause is DSH's own diagnostic line on stderr ("dsh: QUOTA: Insufficient Balance")."""
    s = _Summary()
    for e in s.rows(raw):
        kind = e.get('type')
        if kind == 'status' and e.get('phase') == 'turn_start':
            s.turns += 1
        elif kind == 'status' and e.get('phase') == 'step_end':
            s.add_usage(e.get('usage'))
        elif kind == 'status' and e.get('phase') == 'turn_end':
            reason = (e.get('reason') or {}).get('kind')
            if reason != 'completed':
                s.error = f'turn ended: {reason}'
        elif kind == 'tool_call':
            name = e.get('tool', '?')
            s.tool(name, (e.get('input') or {}).get('command') if name == 'bash' else None)
        elif kind == 'error':
            s.error = str(e.get('message') or e.get('error'))[:500]
        elif kind == 'final':
            s.final, s.finished = str(e.get('text', '')), True
    diagnostics = re.findall(r'^dsh: ([A-Z_]+): (.*)$', stderr.decode('utf-8', 'replace'), re.MULTILINE)
    if diagnostics and s.error:
        code, message = diagnostics[-1]
        s.provider = provider_error(DSH_CODES.get(code), f'{code}: {message}', terminal=True)
    return s.result()


def summarize_dsh_sessions(directory) -> dict[str, Any] | None:
    """DSH session logs (.dsh/sessions/**/session.v4.jsonl[.zstd]) keep every tool result in full and name the model
    of each response (assistant/message data.message.source.model), which `--json` stdout does not."""
    from pathlib import Path
    from .native_logs import decompressed
    files = sorted(p for p in Path(directory).glob('.dsh/sessions/**/session.v*.jsonl*') if p.is_file())
    if not files:
        return None
    out = {'files': [p.relative_to(directory).as_posix() for p in files], 'responses': 0, 'tool_results': 0,
           'actual_models': [], 'readable': True}
    models = set()
    for path in files:
        try:
            text = decompressed(path).decode('utf-8', 'replace')
        except Exception as exc:              # a record must not fail the run; say why it is incomplete
            out['readable'], out['error'] = False, f'{path.name}: {type(exc).__name__}: {exc}'[:300]
            continue
        for line in text.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get('type') == 'assistant/message':
                out['responses'] += 1
                source = ((event.get('data') or {}).get('message') or {}).get('source') or {}
                if isinstance(source.get('model'), str):
                    models.add(source['model'])
            elif event.get('type') == 'tool/result':
                out['tool_results'] += 1
    out['actual_models'] = sorted(models)
    return out


def summarize_zcode(raw: bytes, stderr: bytes = b'') -> dict[str, Any] | None:
    """`zcode --prompt ... --output-format stream-json`: session/turn/tool/model events and a final result. Provider
    failures carry an attribution with status code in turn.failed, and retries as model_retry_scheduled updates."""
    s = _Summary()
    calls = set()
    for e in s.rows(raw):
        kind, payload = e.get('type'), e.get('payload') or {}
        if kind == 'turn.started':
            s.turns += 1
        elif kind == 'session.updated' and payload.get('type') == 'model_retry_scheduled':
            s.provider_failure(payload.get('statusCode'), f"{payload.get('errorCode')}: {payload.get('message')}")
        elif kind == 'turn.failed':
            error = payload.get('error') or {}
            cause = error.get('attribution') or {}
            s.error = str(error.get('message') or 'turn failed')[:500]
            if cause.get('source') == 'provider':
                s.provider_failure(cause.get('statusCode'), f"{cause.get('providerErrorCode') or cause.get('reason')}: {s.error}")
        elif kind == 'session.updated' and payload.get('modelId'):
            s.models.add(payload['modelId'])
        elif kind == 'tool.updated' and payload.get('kind') == 'scheduled' and payload.get('toolCallId') not in calls:
            calls.add(payload.get('toolCallId'))         # later updates (started/result/batch) repeat the call
            name = payload.get('toolName', '?')
            s.tool(name, (payload.get('input') or {}).get('command') if name == 'Bash' else None)
        elif kind == 'turn.completed' and payload.get('resultType') not in (None, 'success'):
            s.error = f"turn result: {payload.get('resultType')}"
        elif kind == 'result':
            s.final, s.finished = str(e.get('response', '')), True
            s.add_usage({k: v for k, v in (e.get('usage') or {}).items() if k.endswith('Tokens')})
    return s.result()
