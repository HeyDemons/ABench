# sandbox/

Operator files used by `./abench`; see [the runner contract](../spec/sandbox_runner.md),
[the OJ contract](../spec/oj_contract.md) and [the egress policy](../spec/egress_policy.yaml).

| Path | Contents |
|---|---|
| `templates/` | report and reference templates copied into the agent view |
| `data/<case>/` | frozen public data, not in git |
| `hidden/<case>/` | encrypted held-out splits, not in git; host-local key under `~/.abench/`, mode 0600 |
| `models/<case>/`, `judge_models/<case>/` | frozen model files, not in git |
| `calibration/cases.yaml` | oracle training flags and small synthetic data sizes |
| `calibration/summarize.py` | summarizes evaluation records; also available through `./abench summarize` |
| `mock_agent.sh` | deterministic agent for pipeline checks, invoked by the mock adapter |

Freeze a case's data with `tools/prepare/run.py` (`--raw-dir` for real data, `--synthetic N` with
`ABENCH_ALLOW_SYNTHETIC=1` for a plumbing check), then `./abench calibrate ID` runs the oracle and the judge in
Docker on that data, from this checkout. Run records are filed under `runs/<harness>-<model>/CATEGORY/ID/RUN/` (`runs/oracle/CATEGORY/ID/RUN/` for calibration).
The former sudo calibration and local launch scripts have been replaced by `./abench calibrate` and
`./abench run`. The Docker CLI and ordinary user permissions are sufficient for current commands.
