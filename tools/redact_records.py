#!/usr/bin/env python3
"""Make explicitly unscorable publication copies of selected structured operator records.

This is a publication filter, not an attestation writer or a raw-log exporter. Original
records are never changed. Only JSON, YAML and JSONL objects (or arrays of objects)
are supported. It deliberately retains measurements, software/hardware models and
image/content digests while removing deployment identities and known credentials.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

import yaml

MARKER = "_abench_publication"
PUBLIC_MODEL_HOSTS = frozenset((
    "api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
    "api.deepseek.com", "api.z.ai", "open.bigmodel.cn", "openrouter.ai",
))
SENSITIVE_NAME = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?key|secret|token|password|passwd|credential|"
    r"authorization|auth|signature|sig|key)(?:$|[_-])", re.I)
URL = re.compile(r"(?:https?|socks5?h?|ssh|tcp|unix|file)://[^\s<>\"'`\])}]+", re.I)
GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)
UUID = re.compile(r"(?<![\w-])[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}(?![\w-])", re.I)
KEY_SHAPE = re.compile(r"(?<![\w-])(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})(?![\w-])")
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----.*?(?:-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----|$)", re.S)
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
LOCAL_PATH = re.compile(
    r"/(?:Users|home|root|mnt|srv|Volumes|data|private/(?:var/folders|tmp)|"
    r"var/folders|run/user|var/lib/docker)(?:/[^\s<>\"'`\])},;]*)?")
SSH_TARGET = re.compile(r"(?<![\w/@])(?:[\w.-]+)@(?:\[[0-9a-f:]+\]|[\w.-]+)(?::\d+)?", re.I)
SSH_COMMAND = re.compile(r"(\bssh\s+(?:(?:-[A-Za-z]+)\s+)*)([\w.-]+@)?([\w.-]+)")
ENV_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)
HOST_FIELDS = frozenset(("host", "hostname", "server", "server_name", "remote_host", "nodename", "computer_name"))
ENDPOINT_FIELDS = frozenset(("base_url", "baseurl", "endpoint", "model_url", "model_endpoint", "docker_host"))
PROXY_FIELDS = frozenset(("proxy", "http_proxy", "https_proxy", "all_proxy", "upstream", "egress_upstream"))
IDENTITY_FIELDS = frozenset(("daemon", "daemon_id", "docker_daemon_id", "serial", "serial_number", "serialnumber"))
ADDRESS_FIELDS = frozenset(("ip", "ipaddress", "ip_address", "address", "gateway", "ipv6gateway", "globalipv6address"))
PATH_FIELDS = frozenset(("source", "source_record", "raw_path", "frozen_work", "dropin", "user_cdi_spec",
                         "job", "run", "log", "cwd", "home", "root", "dockerrootdir", "docker_root_dir",
                         "path", "destination", "mountpoint", "src", "source_dir", "workdir", "workingdir",
                         "directory", "models_dir", "papers_index", "docker_config"))


class RedactionError(ValueError):
    """Only fixed, non-sensitive messages may be exposed by the CLI."""


def _sensitive(name):
    normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", str(name))
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
    return bool(SENSITIVE_NAME.search(normalized))


def _proxy_name(name):
    name = name.lower()
    return name in PROXY_FIELDS or name.endswith(("_proxy", "_upstream")) and name != "no_proxy"


def _loopback(host):
    if host.lower().rstrip(".") in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class Redactor:
    def __init__(self, *, environ=None, secrets=()):
        values = os.environ if environ is None else environ
        self.secrets = {str(value) for key, value in values.items() if value and _sensitive(key)}
        self.secrets.update(str(value) for value in secrets if value)
        self.identities = {}
        self.hosts = set()

    def token(self, category, value):
        identities = self.identities.setdefault(category, {})
        if value not in identities:
            identities[value] = len(identities) + 1
        return f"<redacted:{category}-{identities[value]}>"

    def _credential(self, value):
        if isinstance(value, str) and value:
            self.secrets.add(value)

    def _endpoint_host(self, value, *, proxy=False):
        for match in URL.finditer(value):
            try:
                host = urlsplit(match[0]).hostname or ""
            except ValueError:
                continue
            if host and not _loopback(host) and (proxy or host not in PUBLIC_MODEL_HOSTS):
                self.hosts.add(host)

    def collect(self, value):
        """Collect declared credentials first, so an earlier error/tail cannot echo a later Env secret."""
        if isinstance(value, dict):
            docker_info = any(key in value for key in ("DockerRootDir", "ServerVersion", "NCPU", "MemTotal"))
            for key, item in value.items():
                name = str(key).lower()
                if _sensitive(key) and name not in ("key_env", "credential_variables"):
                    self._credential(item)
                if name in HOST_FIELDS and isinstance(item, str) and not _loopback(item) and item not in PUBLIC_MODEL_HOSTS:
                    self.hosts.add(item)
                    self._endpoint_host(item)
                if isinstance(item, str) and name in ENDPOINT_FIELDS | PROXY_FIELDS:
                    self._endpoint_host(item, proxy=name in PROXY_FIELDS)
                if isinstance(item, str) and name in IDENTITY_FIELDS:
                    self._credential(item)
                if isinstance(item, str) and docker_info and name == "name":
                    self.hosts.add(item)
                self.collect(item)
        elif isinstance(value, list):
            for item in value:
                self.collect(item)
        elif isinstance(value, str):
            assignment = ENV_ASSIGNMENT.fullmatch(value)
            if assignment and _sensitive(assignment[1]):
                self._credential(assignment[2])
            if assignment and (assignment[1].upper().endswith(("_BASE_URL", "_ENDPOINT")) or
                               assignment[1].upper() == "DOCKER_HOST" or _proxy_name(assignment[1])):
                self._endpoint_host(assignment[2], proxy=_proxy_name(assignment[1]))
            for match in URL.finditer(value):
                try:
                    parsed = urlsplit(match[0])
                    self._credential(unquote(parsed.password or ""))
                    for name, content in parse_qsl(parsed.query, keep_blank_values=True):
                        if _sensitive(name):
                            self._credential(content)
                except ValueError:
                    pass

    def url(self, value, *, endpoint=False, proxy=False):
        try:
            parsed = urlsplit(value)
            host = parsed.hostname or ""
            private = False
            try:
                private = not ipaddress.ip_address(host).is_global and not _loopback(host)
            except ValueError:
                private = host.endswith((".internal", ".local", ".lan"))
            if parsed.scheme in ("file", "unix"):
                return self.token("host-path", value)
            if (proxy or endpoint and host not in PUBLIC_MODEL_HOSTS or private or host in self.hosts) and not _loopback(host):
                return self.token("proxy" if proxy else "endpoint", value)
            # Public URLs retain their resource path and harmless query parameters. Strip all userinfo.
            netloc = parsed.netloc.rsplit("@", 1)[-1]
            query = [(key, "<redacted:secret>" if _sensitive(key) else content)
                     for key, content in parse_qsl(parsed.query, keep_blank_values=True)]
            fragment = parsed.fragment
            fragment_sensitive = any(_sensitive(key) for key, _ in parse_qsl(fragment)) if "=" in fragment else False
            if "@" not in parsed.netloc and not any(_sensitive(key) for key, _ in query) and not fragment_sensitive:
                return value
            if "=" in fragment:
                fragment = urlencode([(key, "<redacted:secret>" if _sensitive(key) else content)
                                      for key, content in parse_qsl(fragment, keep_blank_values=True)])
            return urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(query), fragment))
        except ValueError:
            return "<redacted:url>"

    def finish(self, value):
        result = PRIVATE_KEY.sub("<redacted:private-key>", value)
        result = KEY_SHAPE.sub("<redacted:credential>", result)
        for secret in sorted(self.secrets, key=len, reverse=True):
            if len(secret) >= 4:
                result = result.replace(secret, "<redacted:secret>")
            else:
                result = re.sub(r"(?<!\w)" + re.escape(secret) + r"(?!\w)", "<redacted:secret>", result)
        return result

    def text(self, value, *, field="", path=()):
        assignment = ENV_ASSIGNMENT.fullmatch(value)
        if assignment:
            name, content = assignment.groups()
            if _sensitive(name):
                return name + "=<redacted:secret>"
            if _proxy_name(name):
                return self.finish(name + "=" + self.url(content, proxy=True))
            if name.upper().endswith(("_BASE_URL", "_ENDPOINT")) or name.upper() == "DOCKER_HOST":
                return self.finish(name + "=" + self.url(content, endpoint=True))
        if field in HOST_FIELDS and value in self.hosts:
            return self.token("host", value)
        if field in IDENTITY_FIELDS:
            return self.token("identity", value)
        if field in ADDRESS_FIELDS and value and not _loopback(value) and value not in ("0.0.0.0", "::"):
            return self.token("address", value)
        if field in ("context", "docker_context"):
            return self.token("docker-context", value)
        if field in PATH_FIELDS and value.startswith("/") and not (value == "/abench" or value.startswith("/abench/")):
            return self.token("host-path", value)
        if field in ENDPOINT_FIELDS and "://" not in value and value:
            return self.finish(value) if _loopback(value) or value in PUBLIC_MODEL_HOSTS else self.token("endpoint", value)
        # Protect public resource URLs from the local-path and SSH-target patterns below.
        urls = []
        def protect(match):
            urls.append(self.url(match[0], endpoint=field in ENDPOINT_FIELDS, proxy=field in PROXY_FIELDS))
            return f"\x00URL{len(urls) - 1}\x00"
        result = URL.sub(protect, value)
        result = GPU_UUID.sub(lambda match: self.token("gpu", match[0]), result)
        result = UUID.sub(lambda match: self.token("uuid", match[0]), result)
        result = LOCAL_PATH.sub(lambda match: self.token("host-path", match[0]), result)
        result = SSH_COMMAND.sub(lambda match: match[1] + self.token("ssh-target", match[0][len(match[1]):]), result)
        result = SSH_TARGET.sub(lambda match: self.token("ssh-target", match[0]), result)
        # Version/model strings are not network addresses. Numerical measurements never enter this method.
        if "version" not in field and not any(part in ("software", "hardware", "distributions", "packages") for part in path):
            def address(match):
                try:
                    ip = ipaddress.ip_address(match[0])
                    return match[0] if ip.is_loopback or ip.is_unspecified else self.token("address", match[0])
                except ValueError:
                    return match[0]
            result = IPV4.sub(address, result)
        for host in sorted(self.hosts, key=len, reverse=True):
            if host:
                result = re.sub(r"(?<![\w.-])" + re.escape(host) + r"(?![\w.-])", self.token("host", host), result)
        for index, url in enumerate(urls):
            result = result.replace(f"\x00URL{index}\x00", url)
        return self.finish(result)

    def value(self, value, *, path=()):
        if isinstance(value, dict):
            result = {}
            docker_container = "Config" in value and any(key in value for key in ("HostConfig", "NetworkSettings", "State"))
            docker_info = any(key in value for key in ("DockerRootDir", "ServerVersion", "NCPU", "MemTotal"))
            for key, item in value.items():
                name = str(key).lower()
                safe_key = self.text(key, path=path) if isinstance(key, str) else key
                if safe_key in result:
                    raise RedactionError("redaction would merge distinct mapping keys")
                if name in ("scorable", "scoreable") and isinstance(item, bool):
                    result[safe_key] = False
                elif isinstance(item, str) and _sensitive(key) and name not in ("key_env", "credential_variables"):
                    result[safe_key] = "<redacted:secret>"
                elif isinstance(item, str) and (name == "id" and (docker_container or docker_info)):
                    result[safe_key] = self.token("docker-id", item)
                elif isinstance(item, str) and name == "name" and docker_info:
                    result[safe_key] = self.token("host", item)
                elif isinstance(item, str):
                    result[safe_key] = self.text(item, field=name, path=path)
                else:
                    result[safe_key] = self.value(item, path=(*path, name))
            return result
        if isinstance(value, list):
            return [self.value(item, path=path) for item in value]
        if isinstance(value, str):
            return self.text(value, path=path)
        return value

    def publication(self, record):
        if isinstance(record, list):
            return [self.publication(item) for item in record]
        if not isinstance(record, dict):
            raise RedactionError("input must contain objects or arrays of objects")
        if MARKER in record:
            raise RedactionError("input is already a publication copy")
        result = self.value(record)
        result[MARKER] = {"schema": "abench.public_copy/v1", "kind": "sanitized_public_copy",
                          "scoring_attestation": False}
        if "provenance" in result:
            result[MARKER]["original_provenance"] = result["provenance"]
        result.update(provenance="sanitized_public_copy", scorable=False)
        return result


def _dotenv_secrets():
    """Read only local credential values, without exposing .env paths or contents in output."""
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return []
    try:
        values = []
        for line in path.read_text().splitlines():
            assignment = ENV_ASSIGNMENT.fullmatch(line.strip().removeprefix("export ").strip())
            if assignment and _sensitive(assignment[1]) and assignment[2]:
                values.extend((assignment[2], assignment[2].strip("\"'")))
        return values
    except (OSError, UnicodeError):
        raise RedactionError("local credential configuration could not be read") from None


def publish(source, destination, *, environ=None, secrets=()):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or destination.is_symlink():
        raise RedactionError("symlink inputs and outputs are not supported")
    if source.resolve() == destination.resolve():
        raise RedactionError("publication output must be separate from the original")
    if destination.exists():
        raise RedactionError("publication output already exists")
    kind = source.suffix.lower()
    if kind not in (".json", ".jsonl", ".yaml", ".yml"):
        raise RedactionError("only structured JSON, YAML and JSONL files are supported")
    try:
        fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise RedactionError("input must be a regular file")
            raw = stream.read()
        if kind == ".jsonl":
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if not records:
                raise RedactionError("input contains no records")
        else:
            records = [json.loads(raw) if kind == ".json" else yaml.safe_load(raw)]
        redactor = Redactor(environ=environ, secrets=secrets)
        for record in records:
            redactor.collect(record)
        outputs = [redactor.publication(record) for record in records]
        if kind in (".yaml", ".yml"):
            content = yaml.safe_dump(outputs[0], sort_keys=False, allow_unicode=True)
        elif kind == ".json":
            content = json.dumps(outputs[0], ensure_ascii=False, indent=2) + "\n"
        else:
            content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in outputs)
    except RedactionError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, yaml.YAMLError, RecursionError):
        raise RedactionError("input could not be read as supported structured records") from None
    temporary = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, name = tempfile.mkstemp(prefix=".abench-public-", dir=destination.parent)
        temporary = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # A complete file appears atomically. link() refuses existing files, including symlinks;
        # replace() would allow a concurrent invocation to overwrite a publication or original.
        os.link(temporary, destination)
    except OSError:
        raise RedactionError("publication output could not be created without overwriting a file") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "redact_records: invalid arguments; use --help\n")


def main(argv=None):
    parser = _Parser(prog="redact_records", description=__doc__)
    parser.add_argument("input", help="one structured operator record file (not raw logs)")
    parser.add_argument("--out", required=True, help="new publication file; original and existing files are refused")
    args = parser.parse_args(argv)
    try:
        publish(args.input, args.out, secrets=_dotenv_secrets())
    except RedactionError as error:
        print("redact_records: " + str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError, UnicodeError, RuntimeError):
        print("redact_records: publication copy could not be created", file=sys.stderr)
        return 1
    print("wrote sanitized_public_copy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
