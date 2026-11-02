"""Resolve endpoints independently of harness; returned records contain names, never keys."""
import ipaddress
import os
import re
import socket
from pathlib import Path
from urllib.parse import urlsplit
import yaml

ROOT = Path(__file__).resolve().parents[2]

def resolve(agent, name=None, base_url=None, allow_private=False, oj=None, path=None):
    explicit = name is not None
    name = name or agent.default_model
    if name is None:
        if base_url:
            raise ValueError("this agent has no model protocol")
        return None
    entries = yaml.safe_load(Path(path or ROOT / "spec/models.yaml").read_text())["models"]
    if name not in entries:
        raise ValueError(f"unknown model configuration {name!r}")
    cfg = dict(entries[name])
    if cfg.get("api") not in agent.protocols:
        raise ValueError(f"{agent.name} does not support {cfg.get('api')}")
    if set(cfg) - {"api", "base_url", "model", "key_env", "parameters", "auxiliary_models"}:
        raise ValueError("unknown model configuration fields")
    source = "models.yaml"
    environment = os.environ.get("ANTHROPIC_BASE_URL") if cfg["api"] == "anthropic" else None
    if base_url:
        cfg["base_url"], source = base_url, "command_line"
    elif environment and environment.rstrip("/") != cfg["base_url"].rstrip("/"):
        if explicit:
            # Records and reports name the configuration; silently sending it elsewhere would mislabel results.
            raise ValueError(f"ANTHROPIC_BASE_URL contradicts model configuration {name!r}; "
                             "unset it, choose the matching --model, or pass --base-url")
        cfg["base_url"], source = environment, "environment"
    u = urlsplit(cfg["base_url"])
    if u.scheme != "https" or not u.hostname or u.username or u.password or u.fragment or u.query:
        raise ValueError("model endpoint must be an HTTPS URL without credentials, query or fragment")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cfg.get("key_env", "")):
        raise ValueError("invalid model key_env")
    host, port = u.hostname.lower().rstrip("."), u.port or 443
    private = host == "localhost" or host.endswith(".localhost")
    try:
        private |= not ipaddress.ip_address(host).is_global
    except ValueError:
        try:
            private |= any(not ipaddress.ip_address(x[4][0]).is_global
                           and ipaddress.ip_address(x[4][0]) not in ipaddress.ip_network("198.18.0.0/15")
                           for x in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except OSError:
            pass  # proxy resolves names when host DNS is unavailable
    if private and not allow_private:
        raise ValueError("private model endpoint requires --allow-private-model-endpoint")
    import egress_proxy as e
    pol = e.policy_for_case(oj or {}, agent.name)
    for pattern in pol["deny"] + pol["case_deny"]:
        if e._match(host, pattern):
            raise ValueError(f"model endpoint is blocked by case/global policy: {host}")
    cfg.update(name=name, host=host, port=port, source=source, allow_private=bool(allow_private), actual_models=[])
    return cfg
