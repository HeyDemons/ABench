# ABench

ABench asks a research agent to start from an anchor paper, research related work, and improve a method within a
fixed compute budget. The agent sees `task.md`, `interface/`, templates and public training data. After the session,
the judge evaluates the collected code and checkpoint on a held-out split. Labels and reference values stay in the
host parent process; the agent never receives them.

There are 73 cases in 20 tracks. `verify.csv` records data, reference runs, activation and blockers. Cases awaiting
activation, synthetic plumbing runs, and the TC-FT-001 / DS-PIPE-002 pipeline-calibration cases are unscored.
The default versioned suite is `all@2.0.0`, including all 73 cases. This protocol version includes the changed GPU
scoring eligibility and leaderboard membership relative to the original main branch. Reports group difficulty and resource classes
separately and exclude the two pipeline-calibration cases from scoring.

## Quick start (no sudo)

Install Docker Engine with user access or rootless Docker on Linux, or Docker Desktop on macOS. The host needs
Python 3.10+, NumPy, PyYAML and the Docker CLI. Training dependencies and agent CLIs live in profile images.
GPU cases additionally need a working NVIDIA Docker runtime on Linux.

```bash
python3 -m pip install numpy pyyaml
./abench doctor
./abench images pull --profile cpu-sci
python3 tools/hidden_store.py keygen ~/.abench/hidden.key      # once per host
ABENCH_ALLOW_SYNTHETIC=1 python3 tools/prepare/run.py --case DS-PIPE-002 --synthetic 3000 --key-file ~/.abench/hidden.key
./abench calibrate DS-PIPE-002
```

Image publication and locking require registry access. Until the published digests are available, build a local
image to validate the pipeline; local images produce **unscorable** records unless you lock them on this host:

```bash
./abench images build --profile cpu-sci
./abench images build --profile cpu-sci --agent claude  # one agent image per harness you run
./abench images lock --local --profile cpu-sci           # likewise --agent claude, and --profile egress for egress runs
```

`images lock --local` records the local image ID in `spec/env/images.local.lock.yaml`. Runs on exactly those images
score; their records list `image_locks` (`local`) and an `image_lock_note` that the images are unpublished and cannot
be pulled elsewhere. Rebuilding changes the ID, so the run is unscorable until you lock again. A published digest of
the same version always takes precedence, and a published version cannot be locked locally.

Cases share seven versioned environment profiles. Build only the profile and harness combinations you use;
there is no need to build all 35 combinations. Agent images inherit the profile's layers, including Torch/CUDA,
and add the usual command-line tools the original host runner offered (`agent_tools` in `spec/env/images.yaml`:
procps, curl, git, editors, archivers and so on). GPU profiles (2026.11) include gcc/g++ for `torch.compile` and
Triton, so agent and judge compile alike, as on the original host.
Task files, prompts, configuration and judge scripts are injected at runtime and do not require a rebuild.
CLI, Node or packaging changes give the agent image a new `-b<recipe>` suffix; profile versions stay unchanged.
Identical builds share a process lock and reuse a verified successful build. `ABENCH_IMAGE_BUILD_CACHE` overrides
the persistent build metadata directory (normally `~/.cache/abench/images`). Published digest locks remain the
authority for scoring.

When a proxy has different addresses on the host and inside Docker, keep `HTTP_PROXY` / `HTTPS_PROXY` set to
the host-reachable address and set `ABENCH_BUILD_PROXY` to the build-container address. BuildKit fetches registry
authentication from its host-side client; build arguments alone do not configure that connection.

`calibrate` is an optional shortcut. With this checkout's code and the data already frozen in it, the oracle
adapter runs the reference submission in Docker and the judge evaluates it in fresh stage containers. It neither
copies the repository nor prepares data, and it passes the hidden key only by path (`--hidden-key-file`, default
`~/.abench/hidden.key`). Keys are generated on each host with mode 0600 and never copied with the code. Results
appear under `runs/oracle/<case>/<run-id>/`. Data frozen with `--synthetic` is only a plumbing check.

For a real calibration, download the pinned sources and freeze them first:

```bash
python3 tools/prepare/run.py --case FN-LTSF-005 --list-sources       # download these files into RAW
python3 tools/prepare/run.py --case FN-LTSF-005 --raw-dir RAW --key-file ~/.abench/hidden.key
./abench calibrate FN-LTSF-005
./abench calibrate A-LIN-002 --gpus 0                                 # after freezing its data the same way
```

For IC-SPD-001, the raw directory must contain `cifar-10-python.tar.gz`, `cifar10.1_v6_data.npy` and
`cifar10.1_v6_labels.npy` from the pinned sources in `tools/prepare_ic_spd.py`; preparation checks their hashes.
Freeze it with `python3 tools/prepare_ic_spd.py --raw-dir RAW --key-file ~/.abench/hidden.key`, then run
`./abench calibrate IC-SPD-001 --gpus 0`; `ABENCH_ALLOW_SYNTHETIC=1` with `--synthetic 3000` instead of `--raw-dir`
gives an unscored plumbing check. Missing data is a preparation requirement: keep `awaiting_data` until
real-data calibration succeeds.

Generated held-out sets are frozen by `tools/prepare/run.py` with the case's secret seed variable set (it names the
variable) and an empty `--raw-dir`; the seed is escrowed next to the key. If source verification cannot run,
`--allow-unverified` on `tools/prepare/run.py` records that limitation. A GPU of another model than
the case's reference class requires `--allow-nonreference-gpu`; the result is still scored and reports the GPU
model. Pro cases require an exclusive 4-GPU node (4 × H100 is the reference; other models likewise need the flag and
are scored and reported).

## Run a case end to end

Do the host steps once per machine, the case steps once per case, then run. Each step must pass before the next.
The example is the CPU case DS-GBM-001 with the `dsh` harness; substitute the case, its profile (`profile` in
`cases/<ID>/case/oj.yaml`) and the harness you use.

```bash
# Host (once): configuration, images, isolation checks, paper index
python3 -m pip install numpy pyyaml
cp .env.example .env          # model keys (key_env in spec/models.yaml); ABENCH_EGRESS_UPSTREAM if you need a proxy
python3 tools/hidden_store.py keygen ~/.abench/hidden.key      # held-out encryption key, generated on this host
./abench images build --profile cpu-sci
./abench images build --profile cpu-sci --agent dsh
ABENCH_TEST_IMAGE=<cpu-sci image> ABENCH_TEST_AGENT_IMAGES=dsh=<agent image> python3 -m unittest \
  tests.test_container_agent tests.test_container_judge tests.test_egress tests.test_containers \
  tests.test_transfer_cleanup tests.test_volume_cache tests.test_harness_contract
python3 tools/paper_search/build_paper_index.py --snapshot arxiv-metadata-oai-snapshot.jsonl --cutoff 2026-09-01 \
  --categories --out ~/.abench/papers/arxiv-full-20260831.sqlite
ln -s arxiv-full-20260831.sqlite ~/.abench/papers/current.sqlite
./abench doctor               # network_none and paper_index must pass; runtime_probe must not fail

# Case (once): download, freeze, calibrate the judge on the frozen data, and only then activate
python3 tools/prepare/run.py --case DS-GBM-001 --list-sources       # download these files into RAW
python3 tools/prepare/run.py --case DS-GBM-001 --raw-dir RAW --key-file ~/.abench/hidden.key
./abench calibrate DS-GBM-001
python3 tools/prepare/run.py --case DS-GBM-001 --activate

# Runs: the plumbing agent once per case (again after image, code or host changes), then any number of agent runs
export ABENCH_JUDGE_TOOLS_SHA256="$(python3 -c 'import sys; sys.path.insert(0,"tools"); import oj_eval; print(oj_eval.tree_hash(oj_eval.TOOLS))')"
./abench run --case DS-GBM-001 --agent mock
./abench run --case DS-GBM-001 --agent dsh
./abench summarize runs/dsh-deepseek-flash-anthropic/DS-GBM-001/RUN_ID
```

- **Images.** Agent images name the harness; build one per profile and harness you run. Behind a proxy whose
  address differs inside Docker, see the proxy notes above.
- **Isolation checks.** These run real containers: no network or DNS for the agent, read-only view, the hub
  blocked by the egress proxy, timeout kills, refused symlinks, the judge's stage protocol (held-out inputs only after
  the timed phase), transfers and cleanup. The harness contract runs each CLI against a fake model service, so it
  needs no key and spends no tokens.
- **Paper index.** Task statements offer an offline arXiv search, so a run needs an index. Build it from the Cornell
  arXiv metadata snapshot (one JSON object per paper; CC0 descriptive metadata), see
  [tools/paper_search/README.md](tools/paper_search/README.md). The index used for ABench runs holds every arXiv
  paper first submitted through 2026-08-31 (`--cutoff 2026-09-01`, all categories: 3,148,792 papers, 6.7 GB), built
  from the snapshot published on 2026-08-31 (Hugging Face mirror `librarian-bots/arxiv-metadata-snapshot`, revision
  `47141d6`; JSON lines sha256 `ee93401a…`). Build it once per server, on local disk. The runner uses `--papers-index` (refused
  if missing), else `ABENCH_PAPERS_INDEX`, else `~/.abench/papers/current.sqlite`; without an explicit
  `--papers-index` it starts even when no index exists, so check `paper_index` in `doctor` (a hand-picked handful of
  papers counts as a test index).
- **Calibration.** It judges the case's reference submission exactly like an agent's. On real data, require
  `verdict: evaluated` and `judge_isolation: container`, then check the case's correctness coverage, gates, finite
  primary metric and repeat-run stability. `evaluated` alone does not mean those checks passed. Require a confidence
  interval only when the case declares resampling support. Expect a ratio near 1 only for a self-comparison against
  the same implementation. Kernel references are candidate implementations: KernelBench compares them with the
  faster of PyTorch eager and compile, and attention cases compare with SDPA, so their speed ratios need not be 1.
  Activate only after the case-specific calibration and formal scoring prerequisites are satisfied; synthetic runs
  validate plumbing only.
  Generated held-out sets are frozen with the case's secret seed variable set (e.g. `ABENCH_PH_SYM_SEED`).
- **Mock run.** Calibration covers the judge side only (the oracle hands the reference submission straight to the
  judge). The plumbing agent covers the agent side without spending tokens: it reads the case's view and data, searches
  papers for the anchor's title, submits the interface baseline that provides the required code (trained into a
  checkpoint when the case requires one), runs the self-check, writes the report and references from the
  templates, and its submission is collected and judged. Cases whose view has no such baseline (the kernel cases)
  stop at the develop step. If it fails, a real agent fails for the same reason. Run it
  once per case and again after image, code or host changes, not before every agent run.
- **Keys.** `--hidden-key-file` defaults to `~/.abench/hidden.key`. `./abench` reads `NAME=value` lines from `.env`
  in the checkout at startup (variables set in the shell take precedence). `.env` is ignored by git and excluded from
  Docker build contexts; only `.env.example` is committed.

Five harnesses are supported, each pinned in `spec/env/images.yaml` and installed in its own agent image:

| `--agent` | Harness | Default model (`spec/models.yaml`) | Key variable | Protocols it accepts |
|---|---|---|---|---|
| `claude` | Claude Code | `claude-opus-5-5`, effort high | `ANTHROPIC_API_KEY` | anthropic |
| `codex` | OpenAI Codex CLI | `gpt-6-astra`, effort high | `OPENAI_API_KEY` | openai-responses |
| `gemini` | Gemini CLI | `gemini-4-argon`, effort high | `GEMINI_API_KEY` | gemini |
| `dsh` | DeepSeek Harness | `deepseek-flash-anthropic`, effort high | `DEEPSEEK_API_KEY` | anthropic, openai-chat, openai-responses, gemini |
| `zcode` | Z.ai ZCode | `glm-5.3`, effort high | `ZAI_API_KEY` | openai-chat, openai-responses, anthropic |

Before authentication or the development budget starts, the runner probes the installed CLI version without
model credentials. A missing, unrecognized or mismatched version stops the run and leaves `harness_version.json`
and probe logs for diagnosis. Records retain `harness_version` and add requested/actual CLI versions and verified
evidence. Retagging an old CLI as a newer release does not satisfy the check; build the requested agent image.
ZCode's source-release version and its executable's reported version are distinct pins.

`./abench harnesses` lists each harness's capabilities (protocols, accepted effort values, disabled provider tools,
session transcript locations). Provider-side web search tools are disabled for every harness (they would bypass the
logged egress proxy); local fetch tools stay. Effort values are validated before launch, and each CLI's own
session transcript is kept under `native_logs/` in the run record. Failed runs record an `error_type`; model provider failures (bad key, quota, rate limit, outage)
make a run unscorable rather than counting as an agent failure, and jobs retry the transient ones. Details:
[spec/sandbox_runner.md](spec/sandbox_runner.md#harnesses).

Any model configuration whose protocol a harness accepts can be combined with it, e.g.
`--agent claude --model deepseek-flash-anthropic` or a new entry for an OpenAI-compatible gateway with `--agent dsh`.
`--effort` overrides the configured effort for one run. Model names, protocols, URLs and credential variable names
are defined in `spec/models.yaml`. `--base-url` overrides
the endpoint; only its model host is added to the run's egress policy. Endpoints must use HTTPS, and private model
gateways require `--allow-private-model-endpoint`. Credential values are passed through Docker environment names,
never written in records or command arguments. Research egress uses a logging proxy; dataset/model hubs and each
case's data sources remain blocked. `ABENCH_EGRESS_UPSTREAM` can specify an upstream HTTP proxy reachable **from the
sidecar's Docker network**. Rootless Docker's network namespace needs its own reachable proxy configuration.

A host smoke run is available when Docker is absent:

```bash
./abench run --case TC-FT-001 --agent mock --backend host --smoke
```

Smoke runs check process and submission format without decrypting the hidden set. macOS Docker can evaluate CPU
cases using arm64 images, but scoring requires the reference platform `linux/amd64`. The GPU model does not affect scoring; it is
recorded (`agent_hardware`, `judge_gpu_inventory`) and shown by `summarize` and `jobs report`.

For scoring, pin the reviewed host evaluator as well as published images. Before a batch, set
`ABENCH_JUDGE_TOOLS_SHA256` to the reviewed tools tree hash (the judge records and checks it):

```bash
export ABENCH_JUDGE_TOOLS_SHA256="$(python3 -c 'import sys; sys.path.insert(0,"tools"); import oj_eval; print(oj_eval.tree_hash(oj_eval.TOOLS))')"
```

The default session deadline comes from each case's unchanged `resources.wall_clock_limit_h` declaration;
`--timeout-seconds` can shorten it.

## Select cases and run jobs

```bash
./abench cases list --difficulty low --resources low
./abench jobs run --agent mock --dry-run
./abench jobs run --cases TC-FT-001,DS-PIPE-002 --agent mock -n 2 --hidden-key-file ~/.abench/hidden.key
./abench jobs resume JOB_ID -n 2
./abench jobs status JOB_ID
./abench jobs report JOB_ID
```

`--suite`, `--difficulty`, `--resources`, `--area`, `--type`, `--subset` and `--cases` share the website's catalog.
Repeat `--agent` and `--model` (or give comma-separated values) for a matrix; `--attempts` repeats each combination.
Jobs allocate CPU cores and idle GPUs, use per-GPU file locks across jobs, skip hardware this machine does not have
(a compatible GPU that is busy is waited for, or left pending for `jobs resume`), resume completed `evaluation.json`
records, and retry only infrastructure failures. When only judging failed on the infrastructure (a Docker or transfer
error), the entry becomes `rejudge` and the collected submission is judged again in the same run directory, without
rerunning the agent; it ends as `judge_failed` when the retries are used up. An interrupted attempt keeps its partial
record and is restarted in a new run directory; runs started with `--defer-evaluation` wait as `awaiting_judge` for
`./abench judge --run`.
Resuming preserves the same terminal failure classifications, including `provider_error`; the presence of an
evaluation file alone does not turn a failed run into a successful job entry.
`--max-gpu-hours` prevents starting work whose estimate would exceed the cap. Reference-card GPU hours and H100 hours
are reported separately.

A changed case (spec or judge code under `harness/`) no longer matches the frozen suite; other cases still run.
`./abench suites freeze all@VERSION --default` records the current cases as a new, immutable suite version.

New suite names use `MAJOR.MINOR.PATCH`, for example `all@2.0.1`. Increment the major version for substantive changes
to tasks, scoring, timing, budgets or agent-visible information. Use a patch for engineering and documentation
changes that preserve evaluation semantics. Each new snapshot records its creation date separately as `created_at`;
spec and judge hashes identify its case contents. Legacy date-shaped names remain available for existing run
records. A Docker-only migration may be a patch only after equivalence with the previous protocol is verified.

For unscored Budget validation, `--allow-shared-gpu` permits an occupied GPU with at least 24 GiB free and
utilization at most 10%. Cards of another model also require `--allow-nonreference-gpu` (scored, model reported). Formal runs retain the idle
GPU requirement, and Pro runs remain exclusive. Calibration accepts the same explicit sharing flag.

`./abench case check ID` validates generated files, environment and resources, then calibrates the oracle on the
case's frozen data; add `--no-calibrate` for static checks. `./abench judge --run runs/LABEL/ID/RUN` completes deferred judging once (or judges again
after an infrastructure failure), only against the case lock in the run's `run_lock.json` (spec, judge code, the
effective `oj.yaml`, the whole harness directory with its kit copies, and the judge image lock), and
`./abench gc --run-id RUN` removes every Docker object of that run (agent, sidecar and judge). Plain `./abench gc`
removes only stopped leftovers of crashed runs older than ten minutes; `--caches` also drops unused shared data
caches. Final evaluation records are immutable;
legacy deferred placeholders are preserved as `evaluation.deferred.json`, and infrastructure judge failures as
`evaluation.judge-error-N.json`, when judging completes. `--rejudge REASON` judges a final evaluation again on
purpose (for example after a judge rule changed); the old record is kept as `evaluation.superseded-N.json` and the new
one names it and the reason. The report check counts the bibliography by its `ltx_bibitem` entries and does not
require the `ltx_bibliography` class on its section (a relaxation of `main`'s rule).

## Isolation and records

Agent containers run as uid 1000 with dropped capabilities, no new privileges, explicit resource limits and
`--network none`. As on the original host runner, which set no CPU, memory, `/dev/shm` or process limit, containers
get all CPUs and nearly all memory of the Docker host (`host_scale` in `spec/container_policy.yaml`: 90% of its
memory); GPUs are assigned as each case declares. Read-only inputs and a Unix proxy socket arrive through named volumes and tar transfers. Docker
inspect and runtime probes check the effective configuration. Every stage × seed uses a new judge container.
The verified container RAM, CPU quota, shared memory, PID limits, startup memory headroom and the run's session
deadline/compute allowance are saved as `resource_limits.json` with the run; the agent prompt is unchanged.
Run records are filed by harness and model configuration: `runs/<harness>-<model>/<case>/<run-id>/`
(`runs/oracle/...` for calibration, `runs/mock/...` for the plumbing agent).
R/T/G/P stdio handshakes delay held-out inputs until the timed phase ends. Safe archive extraction rejects links,
devices, traversal and oversized outputs. Signals and deadlines clean up run-labeled objects.
Mount verification checks the named volume identity as well as its destination and access mode. Task and judge
containers must be confirmed stopped before collecting artifacts, even when a Docker client has already exited.
Format-only rejection, smoke and pending cases report `judge_isolation: not_started` with
`judge_backend_requested`; they have not executed submission code in a judge stage.

Public data is validated against the frozen hashes and uploaded directly into shared read-only caches, without
first copying large files into a temporary agent view. Source hashes are memoized within the run and invalidated
when source metadata changes. `ABENCH_CACHE_DIR` overrides the persistent cache metadata directory
(`~/.abench/cache`); a missing ready record triggers read-only verification of the existing volume before reuse.
Cache records include hashing, upload and verification time and byte counts. Transfer deadlines cover streaming
reads/writes as well as waiting for Docker. `cleanup.json` records objects actually removed, failures and remaining
objects, including cleanup after a failed setup.

`agent_run.json` retains the existing fields and adds `sandbox`, `model_endpoint`, `run_config` and a `catalog`
snapshot. `evaluation.json` records `judge_isolation: container`, image and dependency attestation, metrics, gates
and intervals. Scoring requires activation, verified container isolation and CLI identity, published image locks, trusted tools,
reference platform/hardware and compliance with the existing development budget. See `spec/oj_contract.md`,
and `spec/sandbox_runner.md`.

The former sudo wrappers, per-user firewall and legacy launch/calibration scripts have been removed after the
Linux rootless container milestones passed. Use the Docker entrypoint or an unscored host smoke run.

## Repository and checks

`index.yaml` classifies cases; `spec/cases/` and `spec/standard_cases/` define tasks and resource declarations;
`spec/handcrafted/` keeps stable contracts for manually authored tasks; `cases/<ID>/` holds the generated agent
contract, case harness and reference submission. `tools/agents/` contains
harness adapters; `tools/runtime/` contains Docker transport, policy and environments; `tools/harness_kit/` is copied
to case interfaces. `tools/catalog.py` feeds CLI selection, scheduling and `site/`. `TESTING.md` is the generated
work list by research area. Historical B-OPT-001 demo scores in `sim/` and `outputs/` remain simulated fixtures.

```bash
python3 -m unittest discover -s tests
python3 tools/validate_task_statement.py --strict
python3 tools/build_standard_cases.py --check
python3 tools/verify_registry.py --check
python3 tools/test_split.py --check
python3 tools/images.py generate --check
```

Run the full command list: unittest covers behavior and regressions, while the separate static checks verify
generated case files (including shared harness copies), the registry, testing split and Dockerfiles. CI runs each
static check once; unittest does not repeat those commands.

Docker integration tests are opt-in (`ABENCH_TEST_IMAGE=<available-cpu-profile-image>`). CI builds the
CPU profile for those tests. Tagged image publication is configured in `.github/workflows/images.yml`; registry
permissions and validated platform-specific locks are required before formal scoring. Image versions are immutable:
to change a profile's packages, give it a new `version` in `spec/env/profiles.yaml`; publication then builds only
the versions not yet published and adds their digests to `spec/env/images.lock.yaml` next to the older ones.
Agent packaging identities include the CLI/source pin, Node version and packaging recipe, so changing those
inputs publishes a new agent version without altering a case's pinned profile version.
