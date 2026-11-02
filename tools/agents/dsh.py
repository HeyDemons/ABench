import yaml
from .base import Agent, Launch, pinned

# Model configuration protocols -> pi-ai wire protocols of DSH's multi-provider route
PI_AI = {"anthropic": "anthropic-messages", "openai-chat": "openai-completions",
         "openai-responses": "openai-responses", "gemini": "google-generative-ai"}


class Dsh(Agent):
    """DeepSeek Harness (`dsh --profile headless --json`). A DeepSeek model on a Messages endpoint uses DSH's own
    `deepseek-official` route, as the shipped harness does; any other endpoint becomes a pi-ai route. Both read the
    key from the endpoint's environment variable. Telemetry is disabled; the container replaces DSH's sandbox."""
    name = "dsh"
    version = pinned("dsh")["version"]
    protocols = tuple(PI_AI)
    default_model = "deepseek-flash-anthropic"
    needs_network = True
    parameters = ("effort",)
    disabled_tools = ("web_search",)        # searches through the DeepSeek API; web_fetch is a local HTTP fetch
    efforts = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
    session_logs = (".dsh/sessions",)

    def effort_values(self, endpoint):
        if self._native(endpoint):
            return ("off", "low", "high", "max")      # DSH's DeepSeek route
        return self.efforts                           # pi-ai passes the value to the provider

    @staticmethod
    def _native(endpoint):
        return endpoint.get("api") == "anthropic" and str(endpoint.get("model", "")).startswith("deepseek")
    cli_version_argv = ("dsh", "--version")

    def version_probe(self, args):
        return Launch(list(self.cli_version_argv), {"DSH_TELEMETRY_MODE": "DISABLED"})

    def launch(self, args, paths, prompt, endpoint=None):
        options = self.options(endpoint)
        effort = options.get("effort")
        selection = {"model": endpoint["model"], **({"reasoningEffort": effort} if effort else {})}
        if self._native(endpoint):
            route = {"baseURL": endpoint["base_url"], "apiKeyEnv": endpoint["key_env"],
                     **({"reasoningEffort": effort} if effort else {})}
            patch = [{"id": "llm-deepseek", "config": route},
                     {"id": "agent-default-model", "config": {"provider": "deepseek-official", **selection}}]
        else:
            model = {"id": endpoint["model"], **({"reasoning": True, "reasoningEfforts": {effort: effort}} if effort else {})}
            route = {"displayName": "ABench model endpoint", "apiKeyEnv": endpoint["key_env"], "api": PI_AI[endpoint["api"]],
                     "baseURL": endpoint["base_url"], "models": [model]}
            patch = [{"id": "llm-pi-ai", "config": {"providers": {"abench": route}}},
                     {"id": "agent-default-model", "config": {"provider": "abench", **selection}}]
        patch.append({"id": "tool-web", "config": {"search": False, "fetch": True}})
        argv = ["dsh", "--profile", "headless", "--patch", str(paths["home"] / ".dsh-abench/patch.yml"), "--json", "-"]
        env = {"DSH_TELEMETRY_MODE": "DISABLED", "DSH_PERMISSION_MODE": "danger-full-access"}
        return Launch(argv, env, prompt.encode(), {".dsh-abench/patch.yml": yaml.safe_dump(patch, sort_keys=False)})

    def summarize(self, raw, stderr=b""):
        from .trajectory import summarize_dsh
        return summarize_dsh(raw, stderr)

    def native_summary(self, directory):
        from .trajectory import summarize_dsh_sessions
        return summarize_dsh_sessions(directory)
