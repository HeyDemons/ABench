# Docker sandbox runner (v3)

`./abench` orchestrates view construction, agent execution, safe collection and independent judging. Ordinary
Docker user access is sufficient on Linux (rootful or rootless) and Docker Desktop on macOS. Host execution is
available only for unscored smoke tests. The [OJ contract](oj_contract.md) defines evaluation semantics.

## Start a run

```bash
./abench doctor
./abench images pull --profile cpu-sci
ABENCH_ALLOW_SYNTHETIC=1 python3 tools/prepare/run.py --case DS-PIPE-002 --synthetic 3000 --key-file ~/.abench/hidden.key
./abench calibrate DS-PIPE-002
./abench run --case TC-FT-001 --agent mock --hidden-key-file ~/.abench/hidden.key
./abench run --case TC-FT-001 --agent claude --hidden-key-file ~/.abench/hidden.key
./abench run --case TC-FT-001 --agent codex --model gpt-6-astra --effort high --hidden-key-file ~/.abench/hidden.key
```

Freeze a case's public data and encrypted split in the checkout with `tools/prepare/run.py` before calibration or an
agent run: `--raw-dir` verifies downloaded sources, generated cases escrow their secret seed, and `--synthetic N`
(with `ABENCH_ALLOW_SYNTHETIC=1`) verifies plumbing only. Calibration uses that frozen data and this checkout's code
directly; it copies nothing and receives the key only by path. Images without
published digest locks allow validation but make results unscorable, unless their exact local image IDs are locked
on the host (`./abench images lock --local`), which scores and is recorded as unpublished. Build only the profiles
needed for a run.

Cases share seven versioned profiles; agent images add one harness to the same profile layers. Build only the
combinations a batch needs, rather than all 35 profile × harness combinations. Runtime task files, prompts,
configuration and judge/control scripts are uploaded or mounted and do not enter the image build identity.
Agent tags contain the profile version, harness pin and a `-b<recipe>` suffix. Node, harness source or packaging
recipe changes produce a new agent identity; the profile version changes only when its own environment changes.
Platform-specific published digests still establish the actual image identity for scoring.

Local builds serialize identical requests per Docker daemon/context and build-input identity. The second request
rechecks the image and verified environment under the lock and reuses a successful build when they still match.
Build readiness and locks persist under `ABENCH_IMAGE_BUILD_CACHE` (default `~/.cache/abench/images`, or
`$XDG_CACHE_HOME/abench/images` when set). A local build-input hash does not replace a registry digest lock.

`tools/sandbox_runner.py` remains a direct entrypoint. Arguments include `--backend docker|host`,
`--agent` / `--harness claude|codex|gemini|dsh|zcode|oracle|mock|command`, `--model`, `--effort`, `--base-url`,
`--allow-private-model-endpoint`, `--gpus`, `--allow-nonreference-gpu`, `--image`,
`--network egress|none|open`, `--hidden-key-file`, `--defer-evaluation`, `--device`,
`--timeout-seconds`, `--result-root`, `--run-id`, `--attempt`, `--suite`, `--smoke` and `--dry-run`.
The host backend requires `--smoke`. The command adapter accepts an argv command with `$ABENCH_*` placeholders;
its custom image must be locked for scoring. New harnesses implement the small adapter contract under
`tools/agents/`; real CLI binaries are installed at image build time, never during an agent session.

## Harnesses

Every harness runs headless and auto-approving inside the agent container (the container is the sandbox). Its pin
lives in `spec/env/images.yaml`; the agent image is the unchanged profile image plus Node.js and that one CLI.
Configuration goes into the agent HOME before launch; keys reach the CLI only as environment variables.

Each pinned harness declares an offline version command and its expected output format. After container startup
and configuration, but before authentication or budget metering, the runner executes that command with a
30-second deadline and without model credentials. A nonzero exit, timeout, unrecognized output or version mismatch
stops setup. `harness_version.json` preserves `requested_version`, `actual_version`, `verified`, command status and
log hashes even when no `agent_run.json` is produced. Successful records add the same `harness_cli_version`
evidence to the agent and sandbox records, plus `requested_cli_version` and `actual_cli_version` in `run_config`.
The existing `harness_version` field remains the pinned release/source version. For ZCode, `version` selects its
source release while `cli_version` pins the different version reported by `zcode --version`. Mock, oracle and
command adapters do not require a pinned CLI probe. Formal runs require verified matching CLI evidence; assigning
a new image tag to an older CLI cannot satisfy that check.

| Harness | Invocation | Endpoint configuration | Effort | Trajectory |
|---|---|---|---|---|
| Claude Code | `claude -p --output-format stream-json`, bypass permissions | `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` | `--effort` | stream-json, ATIF export |
| Codex | `codex exec --json --ephemeral`, sandbox/approval bypass | custom Responses provider in `$CODEX_HOME/config.toml`; key in the model's `key_env` | `model_reasoning_effort` | thread/turn/item events |
| Gemini CLI | `gemini --output-format stream-json --yolo`, prompt on stdin | `GOOGLE_GEMINI_BASE_URL`, `GEMINI_API_KEY`; usage statistics and telemetry off | thinking level override | init/message/tool/result events |
| DSH | `dsh --profile headless --json -`, `danger-full-access` | `--patch` route: DeepSeek's own route for DeepSeek models on a Messages endpoint, otherwise a pi-ai route; telemetry disabled | `reasoningEffort` | session/status/tool/final events |
| ZCode | `zcode --prompt ... --mode yolo --output-format stream-json` | personal provider file; ZCode reads keys only from that file, so `secret_config.py` fills a keyless template inside the container (mode 0600) | `reasoningLevel` | session/turn/tool/result events |

A model configuration's `parameters` other than those a harness supports are refused, never ignored. Codex speaks
only the Responses API. ZCode receives the prompt as one argument (at most 120 KB) and ignores `HTTP(S)_PROXY`, so
its adapter also sets `ZCODE_HTTP_PROXY`; the other CLIs use the standard proxy variables.

**Provider-side web tools are disabled.** A search (or fetch) executed by the model provider bypasses the logged
egress proxy and each case's blocklist, so every launch turns those tools off and records them under
`harness_restrictions.disabled_provider_tools`. Fetch tools that run locally go through the proxy like any other
download and stay. Each choice was checked against the real CLI by observing where a fetch of a public URL arrived:

| Harness | Disabled (provider-side) | Kept (local, through the proxy) |
|---|---|---|
| Claude Code | `WebSearch` (`--disallowedTools`, settings deny) | `WebFetch`, with `skipWebFetchPreflight` (its claude.ai domain lookup would fail behind the policy) |
| Codex | `web_search = "disabled"` (OpenAI-hosted) | none (shell `curl`) |
| Gemini CLI | `google_web_search`, `web_fetch` (Google grounding / URL context) | none (shell `curl`) |
| DSH | `web_search` (DeepSeek API search) | `web_fetch` |
| ZCode | `WebSearch` (provider-native search) | `WebFetch` |

**Effort values are checked before launch.** Each adapter lists the effort values its CLI accepts, narrowed per
model where the CLI or provider does: Claude Code `low|medium|high|xhigh|max`; Codex `none|minimal|low|medium|high|
xhigh|max` (Codex itself forwards any value to the API unchecked); Gemini CLI `low|high`, plus `minimal|medium` for
Flash models and none for 2.5 models; DSH `off|low|high|max` on its DeepSeek route; ZCode by model family as in its
built-in rules (GLM-5.3 `low|high|max`, otherwise `disabled|enabled`). An invalid value is refused before any
container starts, and job matrices skip such pairs with the reason.

**Capabilities and session transcripts.** `./abench harnesses` prints each adapter's capability record (pinned
versions, protocols, default model, parameters, effort values, credential and proxy variables, disabled provider
tools, session-log directories); every agent record keeps it as `harness_capabilities`. The CLIs now persist their
own sessions inside the run's HOME (Codex without `--ephemeral`, Claude without `--no-session-persistence`), and the
runner copies the declared directories into `native_logs/` (at most 512 MiB; symlinks or oversize trees are
dropped and noted under `sandbox.native_log_transfers`): `.claude/projects`, `.codex/sessions`, `.gemini/tmp`,
`.dsh/sessions`, `.zcode/cli/rollout`. Any file containing the model credential is withheld and listed in
`native_logs.withheld`. Transcripts are evidence for audit and later ATIF conversion; they never decide a score.

**Failures are classified.** The runner derives `error_type` from the harness's own structured events and
diagnostics (never tool output): `provider_auth`, `provider_quota`, `provider_rate_limit`, `provider_unavailable`,
`model_not_found`, `provider_rejected` (a preflight request refused otherwise), or the agent's own `agent_error`,
`timeout`, `context_length`. A provider error counts only when it ended the run (the last event, or what the CLI was
still retrying at the deadline). Provider errors make a run unscorable with an explicit reason instead of "agent did
not finish". A failed preflight writes `preflight.json` with its kind. Jobs retry `provider_rate_limit` and
`provider_unavailable` like infrastructure failures, and stop other provider errors in state `provider_error`
rather than `failed`. The classifications are checked against each real CLI's recorded output for 401, 402, 429
and 500 responses (`tests/fixtures/harness_errors`).

`tests/test_harness_contract.py` runs each real CLI offline in its image against `tests/fake_model_server.py`
behind a forward proxy: preflight, proxy routing to a host only the proxy reaches, one shell-tool round, absence of
disabled tools, version pin, and a classified provider failure.

## Views and transport

| Container path | Contents | Access |
|---|---|---|
| `/abench/view` | `task.md`, `interface/`, templates and public files | read-only |
| `/abench/models`, `/abench/papers` | frozen models and filtered paper index | read-only |
| `/abench/workspace` | submission, checkpoints, `.home`, `.tmp` | writable |
| `/run/abench-egress` | proxy Unix socket | read-only |

The agent-visible filenames and paths stay compatible with task statements. Host paths and container paths are
managed separately. Named volumes and bounded tar transfers avoid host directory mounts and rootless UID-mapping
problems. Public data, model and paper caches use content hashes; mutable workspaces belong to one run.
Docker views validate the public-file allowlist against the frozen hashes, then upload those source files
directly into the data volume, preserving their agent-visible names without a temporary large-data copy.
The run's Docker client memoizes source hashes while checking filesystem metadata for changes, so later judge
stages can reuse a model cache without rereading every model byte. Content identity remains independent of
the image used to populate the volume.

Ready metadata and per-content locks persist in `ABENCH_CACHE_DIR` (default `~/.abench/cache`), scoped to the
Docker daemon/context and the volume's creation identity. If metadata is missing, the existing owned volume is
verified through a read-only offline helper before it is reused; its name alone is never trusted. A mismatch or
daemon error fails cache preparation. `sandbox.data_caches` and judge stage `data_caches` record the content digest,
byte count, cache/memo hits and hashing, upload and cold-verification durations.
Archive extraction rejects symbolic/hard links, devices, absolute paths, traversal, too many members and oversized
payloads before collection applies the submission allowlist and drops self-reported result data.

Agent containers use uid 1000, `--network none`, dropped capabilities, no new privileges, init and explicit
CPU/memory/PID/shared-memory/GPU limits. `resources` in each case declares the reference hardware and budget class;
`spec/container_policy.yaml` supplies defaults and hard caps. Inspect checks the effective configuration, and
in-container probes verify that direct connections and DNS fail and the view is unwritable. A rootless daemon
that cannot enforce cgroup limits records that fact; enforcement failure never establishes a scorable run.

## Research egress and model endpoints

| Mode | Path | Scorable network configuration |
|---|---|---|
| `egress` | agent loopback forwarder → read-only Unix socket → bridge sidecar | yes, with successful verification |
| `none` | agent has only loopback, no sidecar | yes |
| `open` | ordinary bridge | no |

The sidecar resolves destinations, applies global and case-specific host rules, rejects private addresses,
and writes a hash-chained `egress_log.jsonl`. Downloads over 20 MiB are listed for review. Dataset/model hubs and
each case's source hosts remain blocked. `ABENCH_EGRESS_UPSTREAM` specifies an upstream proxy reachable from the
sidecar; forwarded host loopback addresses may require a relay in rootless Docker's separate network namespace.
The host's proxy configuration and the daemon/build proxy configuration are separate. Docker/buildx's host-side
registry authentication uses the standard `HTTP_PROXY` / `HTTPS_PROXY` environment. `ABENCH_BUILD_PROXY`, when
set, overrides only the build container's HTTP/HTTPS build arguments; leave it unset when both addresses agree.

`spec/models.yaml` defines model protocols, endpoints, names and credential variable names independently of the
harness. The adapter checks protocol/parameter compatibility before launch and makes a short authenticated request
through the same proxy path before the session budget starts. The run allows only the configured model host.
Endpoints must use HTTPS and cannot override case/global denies. Private gateways need the explicit
`--allow-private-model-endpoint` exception, limited to that model host and port. Docker receives secret values
through inherited environment variable names; records contain names and endpoint metadata. Raw trajectories are
retained with a common summary and optional ATIF export.

Decisions are per host; TLS bodies are not inspected. Provider-side web searches are outside the logging proxy.

## Judging, metering and cleanup

Each stage × seed runs in a fresh network-none judge container using the case's profile. The host decrypts the
held-out split and keeps labels and reference values in memory. The child duplicates stdout for R/T/P protocol
bytes, redirects submission output into logs, and reads G from stdin. Pre-inputs arrive before start; held-out
post-inputs arrive only after T. P is followed by bounded output download and container removal. Declared carry
files are downloaded and uploaded into the next fresh stage. Local selfchecks use the same child protocol.
Parent and child R/T timings are recorded to expose transport overhead. Metrics, gates, baselines and bootstrap
intervals run on the host with the original scoring semantics.

GPU metering samples only the assigned devices. CPU usage and peak memory come from the container cgroup.
A budgeted resource that cannot be metered (no nvidia-smi samples covering the session, no readable container
cgroup) leaves the budget unattested and the run unscorable, the same as exceeding it. Pro `h100_hours` is
checked against GPU busy hours summed over the four allocated cards.
The container-side timeout is backed by a host deadline; signals terminate the container and clean up only
objects carrying this run's `abench.run` label. Every agent, sidecar and judge object also carries
`abench.run_id=RUN`, so `./abench gc --run-id RUN` removes the residues of an interrupted run. Plain
`./abench gc` removes stopped, unused leftovers older than `--min-age-minutes` (default 10) and keeps the shared
`abench-data-*` caches unless `--caches` is given.
Archive-transfer deadlines cover the entire streaming operation, including blocked pipe reads and writes.
Cleanup attempts each object independently, retaining failed IDs for another attempt. The existing cleanup
container/volume counts describe successful removals; `removed`, `failed` and `remaining` preserve the detailed
outcome. The runner writes `cleanup.json` independently of `agent_run.json`, including after an authentication or
version-probe failure, without replacing the original error with a cleanup failure. Optional missing artifacts are
distinguished from daemon and transfer errors.
No agent or judge system users, sudo, iptables or host network namespaces are required.
Stopping a run never interrupts its cleanup: Docker removals, the judge stage shutdown and the runner's final cleanup
hold SIGINT/SIGTERM until they are done and deliver the signal afterwards (`runtime/cleanup.shielded`); the Docker CLI
runs in its own session, so a terminal Ctrl-C cannot abort a removal; `abench judge` and `oj_eval.py` turn SIGTERM into
an interrupt so their cleanup runs; and jobs give a stopped child 180 s for it before killing it.

`--defer-evaluation` collects the submission without a final evaluation. Complete it once with
`./abench judge --run runs/CASE/RUN --hidden-key-file KEY`. The command verifies the collected tree hash, refuses
unless the run's suite version still holds the case unchanged and the case lock recorded at the start of the run
equals the case now, and retains the original smoke/benchmark mode, network, isolation, budget and image restrictions.
The run lock (`run_lock.json`, also `run_config.run_lock`) has a `case` part, compared item by item before judging:
spec and judge-code hashes, environment, the hash of the effective `oj.yaml` (judge, seeds, submission rules, resources,
frozen data), the hash of the whole `harness/` directory including the kit copies the judge executes, and the
`images.lock.yaml` entry of the judge image. Its `agent` part records the harness, CLI version, image and model
endpoint. Records from before the lock are checked against their suite version only.
It refuses to overwrite a final evaluation; old `not_evaluated` placeholders are archived as
`evaluation.deferred.json`. A `judge_error` whose `failure` (`phase`, `exception_type`, `kind`, `retryable`) marks an
infrastructure failure (Docker, transfer, connection or timeout errors, in the policy table of
`tools/agents/errors.py`) may be judged again; it is archived as `evaluation.judge-error-N.json`.

```
runs/<harness>-<model>/<category>/<case>/<run-id>/   # runs/oracle/... for calibration, runs/mock/... for the plumbing agent
  agent_run.json
  run_lock.json          # case lock checked before judging, and what ran the agent
  harness_version.json   # pinned CLI probe, including failed setup
  cleanup.json          # final Docker cleanup outcome, including failed setup
  submission/
  submission.sha256
  retrieval_log.jsonl
  egress_log.jsonl       # egress runs
  logs/
  evaluation.json
```

Existing record fields are retained. `sandbox` adds image/configuration/probe/limit/sidecar evidence,
`model_endpoint` records endpoint identity and source, `run_config` records harness/model/attempt/suite identity,
and `catalog` freezes case classification. Judging records `judge_isolation: container` and image attestation.
Scoring requires activation, trusted tools, locked images, verified CLI identity, isolation and limits, reference
`linux/amd64` hardware and compliance with the unchanged development budgets. Other platforms/hardware can
validate a case, and their records remain unscored.

## Catalog, jobs and verification

`tools/catalog.py` combines `index.yaml`, `verify.csv`, areas and case contracts. CLI and website filtering use this
catalog. Suites freeze case membership, spec hashes, judge-code hashes (everything under `harness/` except the
shared kit copies and documentation) and profile versions. A run checks only its own case against the suite, so
changing one case blocks only that case; a suite version is never edited, and `./abench suites freeze
all@VERSION --default` records the changed cases as a new version. The default `all` suite includes all
73 cases; TC-FT-001 and DS-PIPE-002 remain unscored pipeline calibration cases. Difficulty and resources are
independent axes; Pro H100 hours and Budget reference-card GPU hours are reported separately.

```bash
./abench cases list --category data_science
./abench jobs run --category cv --agent mock --dry-run
./abench jobs run --cases TC-FT-001,DS-PIPE-002 --agent mock -n 2 --hidden-key-file ~/.abench/hidden.key
./abench jobs resume JOB_ID
./abench jobs report JOB_ID
python3 -m unittest discover -s tests
ABENCH_TEST_IMAGE=<cpu-profile-image> python3 -m unittest discover -s tests -p 'test_container_*.py'
```

Jobs allocate CPU cores and idle GPUs, reserve GPU file locks across jobs, start independent child processes,
resume completed evaluations and retry infrastructure failures only. Missing hardware or credentials is reported
as skipped work. Normal completion and resume derive the same terminal classification from the recorded result:
`provider_error` and `failed` do not become `completed` just because an evaluation file exists. Deferred submissions
stay `awaiting_judge`, while interrupted attempts retain their partial records and restart under a new run ID.
A retryable judge failure becomes `rejudge`: `abench judge` judges the collected submission again in the same run
directory on newly leased GPUs, never rerunning the agent, until the retries are used up (`judge_failed`).
Judge stage records carry `phase_seconds` (start, held-out input delivery, run, collect) and, in Docker,
`sandbox.seconds` (input upload, data cache, create, upload, verify, attach, post, download, remove); these
infrastructure times are for finding bottlenecks and never enter a metric.
Estimates add the unchanged development budget to stage execution limits, including seeds and
baseline runs. `--max-gpu-hours` prevents starting new work that would exceed the cap.

Explicit `--allow-shared-gpu` is limited to unscored Budget validation: occupied cards need at least
24,576 MiB free and utilization no higher than 10%. A card of another model than the case's reference class
needs `--allow-nonreference-gpu`; that alone does not make a run unscorable, the GPU model is
reported with the result. The default remains idle allocation, and Pro nodes remain exclusive.

## Historical validation

The 2026-10-18 and 2026-10-23 smoke/calibration records predate Docker migration. Their recorded numbers and
legacy `child_user_netns` isolation retain their original meanings in `verify.csv`; current runs use the Docker
backend. Container smoke and calibration do not activate a case or change evaluation metrics.

## Platform validation

Rootless Linux (Ubuntu 24.04, Docker 29.4.1, cgroup v2): a Unix socket in a named volume remains connectable by uid
1000 in a network-none container when mounted read-only. Tar upload preserves uid 1000; stopped-container tar download
succeeds. `cpu.stat` and `memory.peak` are readable. A 10 ms no-op interval measured inside each process differs by
0.019 ms at the median (five local and five container trials). Parent and child R/T measurements are both retained by
the stage protocol; container creation/copy is outside the timed interval. Rootless policy checks record whether cgroup
limits are enforced ([Docker documentation](https://docs.docker.com/engine/security/rootless/tips/)). Native NVIDIA CDI
exposes a selected GPU without a registered NVIDIA runtime
([NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)).
Rootful Linux validation is pending.

macOS Docker Desktop 29.4.1 (`linux/arm64`) passes the same read-only Unix socket, uid-1000 upload, stopped download
and cgroup probes. Median child interval difference is 0.364 ms, below 10 ms. This platform is a valid CPU
verification target and is unscored by policy.

Behind an HTTP proxy, set `ABENCH_EGRESS_UPSTREAM` for the egress sidecar and `ABENCH_BUILD_PROXY` for image builds;
from rootless Docker the host is reachable at `10.0.2.2`.
