# Docker sandbox runner (v3)

`./abench` orchestrates view construction, agent execution, safe collection and independent judging. Ordinary
Docker user access is sufficient on Linux (rootful or rootless) and Docker Desktop on macOS. Host execution is
available only for unscored smoke tests. The [OJ contract](oj_contract.md) defines evaluation semantics.

## Operator setup

Set up the host once, prepare and review each case once, then run agents on that frozen case. The examples use
`DS-GBM-001` (CPU) and DSH; use the profile pinned in `cases/<ID>/case/oj.yaml` for another case.

### Host and credentials

The host needs Python 3.10+, NumPy, PyYAML and Docker user access. Linux GPU cases also need NVIDIA support in
Docker. Training packages live in the profile image. Generate the hidden key on each host, outside the repository;
it must be a regular, non-symlink file with mode 0600 and is passed to the judge only by path.

```bash
python3 -m pip install numpy pyyaml
cp .env.example .env
python3 tools/hidden_store.py keygen ~/.abench/hidden.key
```

Fill in the credential named by the selected model's `key_env` in `spec/models.yaml`. `./abench` reads literal
`NAME=value` lines from `.env`; shell variables take precedence. The file is ignored by Git and excluded from
Docker build contexts. Credential values do not enter command arguments or run records.

### Task images and harness bundles

Build only the task profiles and harnesses needed for a batch. Agents and judges use the same profile image;
each selected harness CLI comes from a separate bundle mounted into the agent container. One bundle works with
all profiles on its platform, so there is no profile × harness image matrix.

```bash
./abench images build --profile cpu-sci
./abench images build --agent dsh
```

`images build --agent NAME` builds or reuses a host-local bundle in `~/.abench/harness/` (`ABENCH_HARNESS_DIR`
overrides the directory). A temporary `node:<pinned-version>-bookworm` container installs the pinned npm CLI or
builds ZCode's pinned source, then exports Node.js and the CLI as regular files. This trusted builder has network
access; the agent session does not download or install its harness.

The `egress` manifest entry names an existing source image, not a build profile. For local validation without a
published egress digest, pull that source image:

```bash
docker pull python:3.11.13-slim-bookworm
```

Published digests in `spec/env/images.lock.yaml` establish portable profile/egress image locks; `images pull`
uses them, including `./abench images pull --profile egress` after an egress digest is published. Until those
digests exist, explicitly lock the reviewed local image IDs. Every model harness also needs a content lock for
its host-built bundle:

```bash
./abench images lock --local --profile cpu-sci
./abench images lock --local --profile egress
./abench images lock --local --agent dsh
./abench images verify --agent dsh
```

Both image IDs and bundle content digests are stored in `spec/env/images.local.lock.yaml`. A published image
digest takes precedence over its local image lock; bundles are built and locked on each host, with no registry
pull/push path. The egress local-lock command resolves the manifest's source image in the unpublished case.
Rebuilding an image requires locking its new ID. If a bundle changes after it was built, locking refuses it;
rebuild and lock the new bundle rather than editing it in place. Changing the harness pin, Node.js, build script or
platform gives the bundle a new identity. Existing completed bundles are reused; builds of one bundle are
serialized across processes, stage in their own directory and publish their metadata last, so an interrupted or
concurrent build never leaves a bundle that looks finished.

The identity names the *source* of a bundle (pin, Node.js image tag, build script, platform). It does not pin the
builder image digest or npm's transitive dependencies, so two hosts can build different bytes for the same
identity. What a run scores with is the *content* digest of the mounted volume, compared with the digest locked
on that host.

Records from before the task-image/bundle layout (agent images such as `agent-dsh-cpu-sci:2026.10-...`, profile
versions named by date) are not resolved by the current code, and their cases now pin profile 1.0.0, so their run
locks no longer match. Their stored evaluations stay unchanged; to judge such a run again, use the checkout it ran
with.

At startup the bundle is uploaded through the content-addressed volume cache and mounted read-only at
`/opt/abench/harness`; its `bin` is prepended to the image's own PATH, preserving CUDA toolkit paths. The runner
records `sandbox.harness_bundle` (harness/version, cache volume, mount, key, digest and lock), and evaluation
adds `image_locks.harness`. A matching CLI version alone does not replace the bundle-content check.
CPU cases can be validated on macOS arm64, while scoring requires `linux/amd64`.

### Paper index

Build the offline arXiv title/abstract index from the Cornell metadata snapshot, following
[the index guide](../tools/paper_search/README.md):

```bash
python3 tools/paper_search/build_paper_index.py --snapshot arxiv-metadata-oai-snapshot.jsonl --cutoff 2026-09-01 \
  --categories --out ~/.abench/papers/arxiv-full-20260831.sqlite
ln -s arxiv-full-20260831.sqlite ~/.abench/papers/current.sqlite
./abench doctor
```

The index used for ABench runs contains all 3,148,792 papers first submitted through 2026-08-31 (6.7 GB). It was
built from the snapshot published on 2026-08-31: Hugging Face mirror `librarian-bots/arxiv-metadata-snapshot`,
revision `47141d6`, JSON-lines SHA256 `ee93401a…`. Keep one copy per server on local disk. The runner uses explicit
`--papers-index` (missing paths are refused), then `ABENCH_PAPERS_INDEX`, then `~/.abench/papers/current.sqlite`.
Without an explicit argument it can start without an index; require the `doctor` paper-index check to pass before
formal runs. A handful of papers is a test index. The index has no full texts; online research uses the proxy.

### Prepare and calibrate

`prepare` downloads available sources into `RAW_ROOT/<case>/`, verifies and freezes data, calibrates, then
activates. It skips frozen data, skips recalibration of active cases, and reports work that needs the operator.
Use the same case filters as `jobs`, or select a subset of steps with `--steps download,freeze`:

```bash
./abench prepare --cases DS-GBM-001 --raw-root ~/abench-raw
./abench prepare --category data_science --raw-root ~/abench-raw --gpus 0,1 --allow-nonreference-gpu --jobs 8
```

`--jobs` cases are downloaded and frozen at a time (default 4). Calibrations then run alongside each other: every
GPU in `--gpus` holds one GPU case at a time, and cases calibrated on the CPU run without a GPU.

Pinned source URLs and sources with complete expected checksums share verified downloads across cases. The
operator cache defaults to `~/.cache/abench/downloads` (`ABENCH_DOWNLOAD_CACHE` overrides it). A per-source lock
allows one fetch at a time; cache hits are hashed again, and each case gets its own writable copy at the original
raw path. URLs without a pinned revision or full checksum retain direct-download behavior. Publisher revision
and checksum verification still runs before freezing.

Operator `SOURCES` entries can declare `raw_path` explicitly. A tree's `files` accepts the existing string list,
or per-file mappings with `path`, `raw_path` and checksum fields; a mapping of remote names to those records is
also accepted. Explicit paths take precedence over descriptive `note` text. Unambiguous existing files are mapped
before the first build; older descriptions keep the missing-path compatibility fallback.

Freezing and activation hold a case lock and stage outputs privately under ignored `sandbox/.prepare/`. Public
and model hashes and hidden-set integrity are checked before publication. A journal backs up the previous data,
models, lock, OJ and seed escrow, then installs the new artifacts and writes the lock after the data. The old lock
and OJ are unavailable during publication so readers cannot use an old active declaration with partial inputs.
An exception restores the previous generation; after a killed process, the next preparation, activation or frozen
state check for that case performs recovery first. This is a recoverable transaction across several directories,
not one atomic rename. Run preparation separately from evaluation, and keep the journal with its backups until
recovery finishes.

Hugging Face files/trees, GitHub blobs and plain-file URLs can be downloaded automatically; a web page, a source
without a file list, an untuned baseline or missing calibration hardware may need operator input. The lower-level
commands expose each step for manual preparation:

```bash
python3 tools/prepare/run.py --case DS-GBM-001 --list-sources
python3 tools/prepare/run.py --case DS-GBM-001 --raw-dir RAW --key-file ~/.abench/hidden.key
./abench calibrate DS-GBM-001
python3 tools/prepare/run.py --case DS-GBM-001 --activate
```

`calibrate` runs the case's reference submission through the oracle adapter and fresh judge stages using the data
already frozen in this checkout. It neither prepares data nor copies the checkout. A GPU case needs `--gpus`;
a different GPU model additionally needs `--allow-nonreference-gpu` and is recorded. Pro calibration follows the
audit workflow on an exclusive four-GPU node; the ordinary shortcut refuses it.

Review the calibration's correctness coverage, gates, finite primary metric and repeat-run stability before formal
runs. Require `verdict: evaluated` and `judge_isolation: container`; `evaluated` alone does not prove those other
conditions. Require a confidence interval only when the case declares resampling. Expect a ratio near one only
for self-comparisons against the same implementation: kernel references are candidate implementations, compared
with the faster eager/compiled PyTorch or SDPA baseline. Synthetic data validates plumbing and never scores.

Generated held-out sets use the secret seed variable named by the case module; the seed is escrowed beside the
key. The batch command generates one when absent. Manual preparation uses that variable and an empty `--raw-dir`.
`--allow-unverified` records sources that could not be verified. For a synthetic plumbing check, use
`ABENCH_ALLOW_SYNTHETIC=1` and `--synthetic N` instead of `--raw-dir`.

IC-SPD-001 uses `tools/prepare_ic_spd.py`: its raw directory must contain `cifar-10-python.tar.gz`,
`cifar10.1_v6_data.npy` and `cifar10.1_v6_labels.npy` with the pinned hashes. Freeze with
`python3 tools/prepare_ic_spd.py --raw-dir RAW --key-file ~/.abench/hidden.key`, then calibrate with `--gpus 0`.
Use `--synthetic 3000` with `ABENCH_ALLOW_SYNTHETIC=1` for its unscored plumbing check.

### Verify both sides

Before agent runs, `doctor` must report working `network_none` and `paper_index`; `runtime_probe` must not fail.
The Docker tests exercise real containers with local profile images. The harness contract needs the named
bundles built on this host; it uses a fake model service and needs no provider key or paid requests:

```bash
ABENCH_TEST_IMAGE=<cpu-profile-image> ABENCH_TEST_HARNESSES=dsh python3 -m unittest \
  tests.test_container_agent tests.test_container_judge tests.test_egress tests.test_containers \
  tests.test_transfer_cleanup tests.test_volume_cache tests.test_harness_contract
```

Calibration covers the judge. Run the mock agent once per case, and after changes to its image, code or host, to
check the agent side: view/data access, offline paper search, baseline training when needed, self-check, report and
reference collection. Cases with no runnable baseline in the view (the kernel cases) stop at development. A failed
mock run also blocks the corresponding real-agent workflow.

```bash
export ABENCH_JUDGE_TOOLS_SHA256="$(python3 -c 'import sys; sys.path.insert(0,"tools"); import oj_eval; print(oj_eval.tree_hash(oj_eval.TOOLS))')"
./abench run --case DS-GBM-001 --agent mock
./abench run --case DS-GBM-001 --agent dsh
./abench summarize runs/dsh-deepseek-flash-anthropic/data_science/DS-GBM-001/RUN_ID
```

For a host-only process/format check without Docker, use
`./abench run --case TC-FT-001 --agent mock --backend host --smoke`. Host smoke never decrypts the hidden set and is
unscored. TC-FT-001 and DS-PIPE-002 are pipeline calibration cases and remain unscored in all modes.

## Images and entrypoints

The seven task profiles define the scientific software environment independently of harness selection.
[Harness bundles](#task-images-and-harness-bundles) supply the selected CLI at run time. Task files, prompts,
configuration and judge/control scripts are uploaded or mounted and do not require rebuilding an image or bundle.

Profiles and egress use semantic versions, pinned in `spec/env/profiles.yaml` and `spec/env/images.yaml`.
Common command-line tools (`system_packages` in `spec/env/profiles.yaml`: procps, curl, editors, archivers and
pdftotext) live in root task images, so agents and judges share the same tools. GPU profiles add gcc/g++ for
`torch.compile` and Triton. [Tagged publication](../.github/workflows/images.yml) builds profile/egress images and
records validated platform digests; harness bundles are built and locked locally. Changing a profile's packages
requires a new version; published versions and their older lock entries are retained.

Profile-image builds serialize identical requests per Docker daemon/context and build-input identity. The second request
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
`tools/agents/`; real CLIs come from their prebuilt bundles, never from a download during an agent session.

## Harnesses

Every harness runs headless and auto-approving inside the agent container (the container is the sandbox). Its pin
lives in `spec/env/images.yaml`; its bundle supplies Node.js and that one CLI inside the case's task image.
Configuration goes into the agent HOME before launch; keys reach the CLI only as environment variables.

Each pinned harness declares an offline version command and its expected output format. After container startup
and configuration, but before authentication or budget metering, the runner executes that command with a
30-second deadline and without model credentials. A nonzero exit, timeout, unrecognized output or version mismatch
stops setup. `harness_version.json` preserves `requested_version`, `actual_version`, `verified`, command status and
log hashes even when no `agent_run.json` is produced. Successful records add the same `harness_cli_version`
evidence to the agent and sandbox records, plus `requested_cli_version` and `actual_cli_version` in `run_config`.
The existing `harness_version` field remains the pinned release/source version. For ZCode, `version` selects its
source release while `cli_version` pins the different version reported by `zcode --version`. Mock, oracle and
command adapters do not require a pinned CLI probe. Formal runs require verified matching CLI evidence;
labelling a bundle as a newer CLI does not satisfy that check.

| Harness | Invocation | Endpoint configuration | Effort | Trajectory |
|---|---|---|---|---|
| Claude Code | `claude -p --output-format stream-json`, bypass permissions | `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` | `--effort` | stream-json, ATIF export |
| Codex | `codex exec --json`, sandbox/approval bypass | custom Responses provider in `$CODEX_HOME/config.toml`; key in the model's `key_env` | `model_reasoning_effort` | thread/turn/item events |
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
`native_logs.withheld`. Transcripts are evidence for audit and later ATIF conversion; they never decide a score. For DSH, the stdout
summary counts `truncated:true` events and supplements actual model identities from the complete native session;
use the native log to inspect long tool results. Compressed sessions are checked for credentials after decoding;
unreadable compressed files are withheld.

**Failures are classified.** The runner derives `error_type` from the harness's own structured events and
diagnostics (never tool output): `provider_auth`, `provider_quota`, `provider_rate_limit`, `provider_unavailable`,
`model_not_found`, `provider_rejected` (a preflight request refused otherwise), or the agent's own `agent_error`,
`timeout`, `context_length`. A provider error counts only when it ended the run (the last event, or what the CLI was
still retrying at the deadline). Provider errors make a run unscorable with an explicit reason instead of "agent did
not finish". A failed preflight writes `preflight.json` with its kind. Jobs retry `provider_rate_limit` and
`provider_unavailable` like infrastructure failures, and stop other provider errors in state `provider_error`
rather than `failed`. The classifications are checked against each real CLI's recorded output for 401, 402, 429
and 500 responses (`tests/fixtures/harness_errors`).

`tests/test_harness_contract.py` runs each selected bundle's CLI offline in a task image against `tests/fake_model_server.py`
behind a forward proxy: preflight, proxy routing to a host only the proxy reaches, one shell-tool round, absence of
disabled tools, version pin, and a classified provider failure.

## Views and transport

| Container path | Contents | Access |
|---|---|---|
| `/abench/view` | `task.md`, `interface/`, templates and public files | read-only |
| `/abench/models`, `/abench/papers` | frozen models and filtered paper index | read-only |
| `/opt/abench/harness` | selected CLI bundle and its Node.js runtime | read-only |
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
CPU/memory/PID/shared-memory/GPU limits. As on the original host runner, agents receive all host CPUs and nearly
all host memory (`host_scale` in `spec/container_policy.yaml`: 90% of memory); GPUs follow the case allocation.
`resource_limits.json` records the verified allocation, startup memory headroom, session deadline and declared
compute allowance for the operator; the agent prompt stays unchanged. `resources` in each case declares the reference hardware and budget class;
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
intervals run on the host with the original scoring semantics. Task and judge containers must be confirmed
stopped before artifacts are collected; a Docker client exiting is not sufficient. Mount checks verify volume
identity as well as destination and access mode. Format-only rejection, smoke and pending cases record
`judge_isolation: not_started` plus `judge_backend_requested`; they have not executed a judge stage.

GPU metering samples only the assigned devices. CPU usage and peak memory come from the container cgroup.
A budgeted resource that cannot be metered (no nvidia-smi samples covering the session, no readable container
cgroup) leaves the budget unattested and the run unscorable, the same as exceeding it. Pro `h100_hours` is
checked against GPU busy hours summed over the four allocated cards.
The container-side timeout is backed by a host deadline; signals terminate the container and clean up only
objects carrying this run's `abench.run` label. Every agent, sidecar and judge object also carries
`abench.run_id=RUN`, so `./abench gc --run-id RUN` removes the residues of an interrupted run. Plain
`./abench gc` removes stopped, unused leftovers older than `--min-age-minutes` (default 10) and keeps the shared
`abench-data-*` caches unless `--caches` is given.
Stopped containers and private staging retained for a pending collection are protected from both forms of GC.
Use `./abench recover --run PATH` to finish that execution, or explicitly discard it with
`./abench gc --run-id RUN --discard-recovery`. Discard verifies daemon identity, ownership labels and stopped
state before deleting registered objects and private staging; it also handles a process interrupted between
write-ahead registration and committing its pending context. A mismatched daemon or uncertain state fails closed.
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

`attempt.json` records operator-only phases and structured exception types, including setup failures. After
execution completes, the runner freezes the agent and sidecar before committing `collection_pending.json` and
retaining a private temporary directory. The context contains metering results and configuration names, not model
credentials, pass-env values, authentication input or hidden keys. Container/volume ownership is bound to the
original daemon. Collected archives are committed atomically with content-hash receipts; a partial directory or
unchanged file size/mtime is insufficient evidence for reuse. The original runner and recovery use the same run
lock, and immutable agent records are never replaced.

`./abench recover --run PATH` resumes only collection and finalization, preserving the original execution status,
wall time, CPU/GPU accounting, prompt hash and run lock. It never executes, authenticates or meters the agent
again. `--collect-only` finalizes its agent record for a later `abench judge`; jobs recognize this as judge-only
work. Native logs are withheld if the original model credential is unavailable for credential checks. Successful
cleanup removes retained staging and its registration; incomplete recovery remains available for another explicit
resume and cannot establish scoreability from partial artifacts.

`--defer-evaluation` collects the submission without a final evaluation. Complete it once with
`./abench judge --run runs/LABEL/CATEGORY/CASE/RUN --hidden-key-file KEY`. The command verifies the collected tree hash, refuses
unless the run's suite version still holds the case unchanged and the case lock recorded at the start of the run
equals the case now, and retains the original smoke/benchmark mode, network, isolation, budget and image restrictions.
An explicit `--rejudge REASON` can supersede a final evaluation: the old record is kept as
`evaluation.superseded-N.json`, and the new one names it and the reason. Report format checks count
`ltx_bibitem` bibliography entries; their enclosing section need not carry the `ltx_bibliography` class.
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
  attempt.json          # operator phases and typed failures, never agent-visible
  collection_pending.json # only while stopped collection/finalization needs recovery
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
Scoring requires activation, trusted tools, locked images and harness content, verified CLI identity, isolation and limits, reference
`linux/amd64` hardware and compliance with the unchanged development budgets. Other platforms/hardware can
validate a case, and their records remain unscored.

## Catalog, jobs and verification

`tools/catalog.py` combines `index.yaml`, `verify.csv`, areas and case contracts. CLI and website filtering use this
catalog. Suites freeze case membership, spec hashes, judge-code hashes (everything under `harness/` except the
shared kit copies and documentation) and profile versions. A run checks only its own case against the suite, so
changing one case blocks only that case; a suite version is never edited, and `./abench suites freeze
all@VERSION --default` records the changed cases as a new version. The default `all` suite includes all
73 cases; `--suite all` resolves the default version pinned in `spec/suites.yaml`, and that version is retained in
the run record. TC-FT-001 and DS-PIPE-002 remain unscored pipeline calibration cases. Difficulty and resources are
independent axes; Pro H100 hours and Budget reference-card GPU hours are reported
separately. Suite names use `MAJOR.MINOR.PATCH`: substantive task, scoring, timing, budget or agent-input changes
need a major version; engineering/documentation changes that preserve evaluation semantics use a patch. A
Docker-only migration qualifies for a patch only after protocol equivalence is verified. Legacy date-shaped names
remain readable, and `created_at` records snapshot dates independently.

```bash
./abench cases list --category data_science
./abench jobs run --category cv --agent mock --dry-run
./abench jobs run --cases TC-FT-001,DS-PIPE-002 --agent mock -n 2 --hidden-key-file ~/.abench/hidden.key
./abench jobs resume JOB_ID
./abench jobs report JOB_ID
python3 -m unittest discover -s tests
ABENCH_TEST_IMAGE=<cpu-profile-image> python3 -m unittest discover -s tests -p 'test_container_*.py'
```

Jobs allocate CPU cores and idle GPUs, reserve resource file locks across jobs, start independent child processes,
resume completed evaluations and retry infrastructure failures only. Missing hardware or credentials is reported
as skipped work. Actual Docker jobs use the daemon's CPU capacity; a dry run can work without Docker and uses the
client's CPU count for its preliminary plan. With the current `host_scale` policy, every agent and judge container
keeps all daemon CPUs and 90% of its memory. Jobs therefore lease the entire daemon node for one trial at a time,
including GPU trials, rather than shrinking a trial's CPU or memory allowance to increase concurrency. `-n` remains
an upper bound; it does not override this node lease. Independent nodes can execute trials in parallel.
The daemon node lock and GPU locks coordinate jobs on the same client with a shared lock directory
(`~/.abench/gpu-locks`). Manual `abench run` processes and separate clients without that shared directory do not
participate; the operator must avoid overlapping them on the same node. Pro also requires an exclusive node.
Normal completion and resume derive the same terminal classification from the recorded result:
`provider_error` and `failed` do not become `completed` just because an evaluation file exists. Deferred submissions
stay `awaiting_judge`, while interrupted attempts retain their partial records and restart under a new run ID.
A retryable judge failure becomes `rejudge`: `abench judge` judges the collected submission again in the same run
directory on newly leased GPUs, never rerunning the agent, until the retries are used up (`judge_failed`).
An interrupted stopped collection becomes `recollect` and invokes `abench recover` in the original run directory;
it consumes no new development-budget estimate. If that recovery fails, `collection_pending` waits for another
explicit resume instead of looping or repeating paid agent work. Only earlier interruptions without committed
execution evidence use a new agent attempt.

Each job has a checksummed, fsynced `job.journal.jsonl` of entry transitions alongside its compatible `job.json`
checkpoint. Status/report/resume read the checkpoint and replay only its matching journal; a truncated final event
is ignored, and complete corruption is rejected. The executor re-reads the latest state after acquiring `job.lock`,
commits launch intent before spawning a child, and checkpoints on startup and exit. This avoids rewriting a large
matrix for every entry while preserving retry counts and immutable run paths after a crash.

`--provider-concurrency N` (default 1, on jobs run/resume) caps active agent sessions per endpoint and credential
identity within that job, separately from trial/node leases. The permit ends when agent execution ends, so collection
and judging do not consume it. HTTP 429/5xx infrastructure retries keep the existing retry limit and classification
but wait 5 seconds initially, doubling up to 300 seconds between outer attempts; cooldown eligibility survives job
restarts. No provider key value is persisted, and no agent-side retry behavior or session budget is changed.
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
