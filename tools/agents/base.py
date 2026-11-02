"""Small, explicit harness contract; adapters never receive operator secrets."""
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def pinned(name: str) -> dict:
    """The harness CLI pin in spec/env/images.yaml, the single source for adapter and agent image versions."""
    harnesses = yaml.safe_load((ROOT / 'spec/env/images.yaml').read_text())['harnesses']
    return dict(harnesses[name])


@dataclass(frozen=True)
class Launch:
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    stdin: bytes = b""
    # HOME-relative path -> content, written before the launch (harness configuration, never a secret)
    files: dict[str, str] = field(default_factory=dict)


class Agent:
    name = ""
    version = "1"
    protocols: tuple[str, ...] = ()
    default_model = None
    needs_network = False
    extra_files: tuple[str, ...] = ()
    # model configuration parameters this harness can apply; anything else is refused, never ignored
    parameters: tuple[str, ...] = ()
    # Built-in tools that would run at the model provider (web search, provider-side fetch) and so bypass the
    # logged egress proxy and each case's blocklist; every launch turns them off. Local fetch tools stay.
    disabled_tools: tuple[str, ...] = ()
    # Reasoning effort values the CLI accepts (None: not validated here); effort_values() narrows them per model.
    efforts: tuple[str, ...] | None = None
    # HOME-relative directories holding the CLI's own session transcripts, kept as raw evidence after a run.
    session_logs: tuple[str, ...] = ()
    # An offline CLI identity probe, independent of model credentials and configuration.
    cli_version_argv: tuple[str, ...] = ()
    cli_version_prefix = ""
    cli_version_suffix = ""

    @property
    def requested_cli_version(self) -> str | None:
        if not self.cli_version_argv:
            return None
        pin = pinned(self.name)
        return str(pin.get("cli_version", pin["version"]))

    def version_probe(self, args) -> Launch | None:
        return Launch(list(self.cli_version_argv)) if self.cli_version_argv else None

    def parse_cli_version(self, stdout: bytes) -> str | None:
        # Exact output formats keep a warning or an arbitrary number from becoming identity evidence.
        semver = r"(?P<version>\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)"
        pattern = re.escape(self.cli_version_prefix) + semver + re.escape(self.cli_version_suffix)
        match = re.fullmatch(pattern, stdout.decode("utf-8", errors="replace").strip())
        return match["version"] if match else None

    def launch(self, args, paths, prompt, endpoint=None) -> Launch:
        raise NotImplementedError

    def proxy_env(self, proxy_url) -> dict[str, str]:
        """Extra variables for a harness that ignores HTTP(S)_PROXY (the runner sets those for every harness)."""
        return {}

    def credential_env(self, endpoint) -> tuple[str, ...]:
        """Further environment names the CLI reads the endpoint's key from (besides endpoint['key_env'])."""
        return ()

    def preflight(self, endpoint) -> Launch | None:
        if not endpoint:
            return None
        from .preflight import protocol_preflight
        return protocol_preflight(endpoint)

    def summarize(self, raw: bytes, stderr: bytes = b"") -> dict[str, Any] | None:
        """Trajectory summary from the harness's stdout events and its own stderr diagnostics."""
        return None

    def native_summary(self, directory) -> dict[str, Any] | None:
        """Summary of the harness's own session logs as collected under native_logs/ (complete where stdout is
        truncated): responses, tool results and the models that answered. None when the harness keeps none."""
        return None

    def effort_values(self, endpoint) -> tuple[str, ...] | None:
        """Effort values accepted for the endpoint's model; None when this harness does not restrict them."""
        return self.efforts

    def check_effort(self, endpoint, effort) -> None:
        """Refuse an effort value before any container starts, instead of failing inside the CLI later."""
        allowed = self.effort_values(endpoint or {})
        if effort is not None and allowed is not None and str(effort) not in allowed:
            model = (endpoint or {}).get('model')
            raise ValueError(f'{self.name} does not accept effort {effort!r} for model {model}; '
                             + (f'choose one of {", ".join(allowed)}' if allowed else 'this model takes no effort setting'))

    def options(self, endpoint) -> dict:
        """The endpoint's parameters, after refusing any this harness cannot apply (unknown names, effort values)."""
        options = dict((endpoint or {}).get('parameters') or {})
        unsupported = sorted(set(options) - set(self.parameters))
        if unsupported:
            raise ValueError(f'{self.name} does not support model parameters: {unsupported}')
        self.check_effort(endpoint, options.get('effort'))
        return options

    def capabilities(self) -> dict[str, Any]:
        """What this harness supports and how it is restricted, in one record (see `./abench harnesses`)."""
        return {'harness': self.name, 'image_version': self.version, 'cli_version': self.requested_cli_version,
                'needs_network': self.needs_network, 'protocols': list(self.protocols),
                'default_model': self.default_model, 'parameters': list(self.parameters),
                'efforts': None if self.efforts is None else list(self.efforts),
                'credential_variables': list(self.credential_env({})),
                'own_proxy_variables': sorted(self.proxy_env('PROXY')),
                'disabled_provider_tools': list(self.disabled_tools), 'session_logs': list(self.session_logs)}
