# ABench

ABench evaluates research agents: start from an anchor paper, study related work, and improve a method within a
fixed compute budget. The agent receives the task, interface, report templates and public training data. After
its session ends, an independent judge evaluates the submitted code and checkpoint on a held-out split. Labels,
reference values and the hidden-set key stay outside the agent environment.

There are **73 cases in 20 tracks**, grouped into five [categories](spec/categories.yaml): `llm_opt`, `llm_post`,
`cv`, `data_science` and `pro`. The default selection is `--suite all`; [suite definitions](spec/suites.yaml) pin
its version, which is retained in each run record. [verify.csv](verify.csv) records data readiness, calibration,
activation and blockers.

## Set up a host

Use Python 3.10+ and Docker Engine (rootful or rootless) on Linux, or Docker Desktop on macOS. GPU cases also need
NVIDIA support in Docker. Task dependencies live in versioned profile images. Harness CLIs are pinned in
[spec/env/images.yaml](spec/env/images.yaml), built once per host, and mounted into the task image as read-only
bundles with their own Node.js runtime.

```bash
python3 -m pip install numpy pyyaml
cp .env.example .env                         # fill in the model credential for your harness
python3 tools/hidden_store.py keygen ~/.abench/hidden.key
./abench images build --profile cpu-sci
./abench images build --agent dsh
docker pull python:3.11.13-slim-bookworm       # egress fallback image from spec/env/images.yaml
```

Build only the task profiles and harness bundles you use; the same bundle works across profiles on its platform.
Egress uses the source image named in the image manifest. Scoring requires matching task/egress image locks and
a host-local content lock for the selected bundle. Keep the hidden key outside the repository. See
[operator setup](spec/sandbox_runner.md#operator-setup) for image locks, proxies and Docker verification.

Install the [offline paper index](tools/paper_search/README.md) from the arXiv metadata snapshot, then run:

```bash
./abench doctor
```

Before a formal run, require working isolation and paper-index checks. The index provides titles and abstracts;
agents read full papers through the logged research proxy. Dataset/model hubs and each case's source hosts stay
blocked. A missing default index does not itself prevent `run`, so check the `paper_index` result in `doctor`.

## Prepare and run a case

The example uses the CPU case `DS-GBM-001` and the `dsh` harness. Once the host is ready, download, verify, freeze,
calibrate and activate the case:

```bash
./abench prepare --cases DS-GBM-001 --raw-root ~/abench-raw
```

`prepare` skips finished work and reports cases that need operator input. GPU calibration needs `--gpus`; add
`--allow-nonreference-gpu` for a different GPU model. Review calibration correctness, gates and finite metrics
before formal runs. [Preparation details](spec/sandbox_runner.md#prepare-and-calibrate) cover manual sources,
synthetic validation and the lower-level commands.

Pin the reviewed evaluator, run the plumbing agent once after preparing a case or changing the host, then run the
research agent:

```bash
export ABENCH_JUDGE_TOOLS_SHA256="$(python3 -c 'import sys; sys.path.insert(0,"tools"); import oj_eval; print(oj_eval.tree_hash(oj_eval.TOOLS))')"
./abench run --case DS-GBM-001 --agent mock
./abench run --case DS-GBM-001 --agent dsh
./abench summarize runs/dsh-deepseek-flash-anthropic/data_science/DS-GBM-001/RUN_ID
```

Runs are stored under `runs/<harness>-<model>/<category>/<case>/<run-id>/`; `mock` and `oracle` use their own labels.
Each record retains the agent trajectory, submission, resource use, image/CLI identity, cleanup and evaluation.
`--defer-evaluation` collects a submission for later `./abench judge --run PATH`.

Raw run records and host-local image locks contain private deployment information. To share selected structured
metadata, create a separate publication copy:

```bash
python3 tools/redact_records.py PATH/evaluation.json --out exports/evaluation.public.json
```

The filter supports JSON, YAML and JSONL; it removes deployment identities and recognized credentials, including
known secret values from the record, process environment and local `.env`. Measurements, software/hardware models,
image digests and fixed `/abench` paths are retained. Each output object is marked `sanitized_public_copy` and
unscorable; it cannot replace original attestation evidence. Existing files and symlinks are refused. This tool
handles structured operator metadata, not raw logs or native session transcripts, which remain private.

To publish source without the private identities retained in older Git commits, export the reviewed working tree
as a separate archive:

```bash
python3 tools/export_public.py --out exports/abench-public.tar.gz --include-untracked
```

The archive contains current source bytes and a per-file manifest. It excludes Git history, run records, frozen
data, local image locks and private configuration. `--include-untracked` includes new source files before they are added to Git;
ignored files remain excluded. Source symlinks, recognized credentials and deployment identities stop the export.
The existing development history and private files stay in place.

**Scoring and validation are distinct.** Scoring requires real frozen data, activation, trusted tools, matching
image, harness-bundle and CLI locks, verified Docker isolation, `linux/amd64`, and attested resource budgets. Synthetic runs,
host smoke, inactive cases and the two pipeline-calibration cases (`TC-FT-001`, `DS-PIPE-002`) are unscored. A GPU
model different from the reference is recorded and can score with the explicit flag; shared GPU validation is
unscored and Pro runs require an exclusive four-GPU node. See the [evaluation contract](spec/oj_contract.md).

Without Docker, use a process-and-format smoke check:

```bash
./abench run --case TC-FT-001 --agent mock --backend host --smoke
```

## Choose a harness and model

| `--agent` | Harness | Default model | Credential variable | Protocols |
|---|---|---|---|---|
| `claude` | Claude Code | `claude-opus-5-5` | `ANTHROPIC_API_KEY` | anthropic |
| `codex` | Codex CLI | `gpt-6-astra` | `OPENAI_API_KEY` | openai-responses |
| `gemini` | Gemini CLI | `gemini-4-argon` | `GEMINI_API_KEY` | gemini |
| `dsh` | DeepSeek Harness | `deepseek-flash-anthropic` | `DEEPSEEK_API_KEY` | anthropic, openai-chat, openai-responses, gemini |
| `zcode` | ZCode | `glm-5.3` | `ZAI_API_KEY` | openai-chat, openai-responses, anthropic |

`./abench harnesses` lists pinned versions and capabilities. Each harness uses its own bundle; no per-harness
image is built. [spec/models.yaml](spec/models.yaml) defines endpoints
and parameters independently of harnesses; use `--model`, `--effort` and `--base-url` to choose a compatible route.
The runner verifies the CLI before authentication and metering. Provider-side web tools are disabled, native
sessions are retained, and provider failures are classified separately from agent failures.
[Harness details](spec/sandbox_runner.md#harnesses) describe these controls.

## Run batches

```bash
./abench cases list --category cv
./abench jobs run --category llm_opt --agent dsh --dry-run
./abench jobs run --cases TC-FT-001,DS-PIPE-002 --agent mock -n 2
./abench jobs resume JOB_ID -n 2
./abench jobs status JOB_ID
./abench jobs report JOB_ID
```

Selection shares the catalog with the website: `--suite`, `--category`, `--difficulty`, `--resources`, `--area`,
`--type`, `--subset` and `--cases`. Repeat `--agent`/`--model` for a matrix and use `--attempts` for repeats. Jobs
lease resources, preserve interrupted attempts and retry infrastructure failures; a judge-only failure reuses the
collected submission. [Scheduling and recovery](spec/sandbox_runner.md#catalog-jobs-and-verification) has the details.

If collection fails after the agent ends, `jobs resume` collects that same stopped execution again. For a single
run, use `./abench recover --run PATH` (or add `--collect-only`, then judge separately). This never starts the agent
or its budget meter again. Pending collection is protected from ordinary garbage collection; explicitly discard it
with `./abench gc --run-id RUN_ID --discard-recovery` when it is no longer needed.

`--provider-concurrency` limits simultaneous agent sessions per endpoint/account within a job (default 1).
Transient provider failures use an outer cooldown before another attempt. `jobs status` and `jobs resume` replay
the durable per-entry journal; `job.json` remains a compatible checkpoint written at startup and exit.

## Develop and verify

```bash
python3 -m unittest discover -s tests
python3 tools/validate_task_statement.py --strict
python3 tools/build_standard_cases.py --check
python3 tools/verify_registry.py --check
python3 tools/test_split.py --check
python3 tools/images.py generate --check
```

Run the full list: unit tests cover behavior; the other commands check generated contracts, registry, testing
split and Dockerfiles. Docker integration checks are opt-in; see [operator setup](spec/sandbox_runner.md#verify-both-sides).

| Reference | Contents |
|---|---|
| [Sandbox runner](spec/sandbox_runner.md) | Host setup, images, isolation, networking, records and recovery |
| [OJ contract](spec/oj_contract.md) | Views, stage protocol, scoring and provenance |
| [Report contract](spec/report_contract.md) | Reports, bibliography and reference integrity |
| [Task statements](spec/task_statement_standard.md) | Task authoring and agent-visible contracts |
| [Offline paper search](tools/paper_search/README.md) | Index preparation and search behavior |
| [Testing plan](TESTING.md) | Case work list by research area |
| [Website](site/README.md) | Static catalog and local viewing |

`cases/` contains generated task contracts, case harnesses and reference submissions; `spec/` defines their
versioned inputs. `tools/agents/` implements adapters, `tools/harness_bundles.py` packages their CLIs,
`tools/runtime/` implements container execution, and
`tools/catalog.py` supplies the CLI, jobs and website. Historical B-OPT-001 demo scores in `sim/` and `outputs/`
are simulated fixtures. Changes to tasks, budgets, scoring or agent-visible inputs require a new suite version;
published image versions and historical records stay immutable.
