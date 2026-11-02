from .claude import Claude
from .codex import Codex
from .dsh import Dsh
from .gemini import Gemini
from .offline import Command, Mock, Oracle
from .zcode import ZCode

REGISTRY = {a.name: a for a in (Claude(), Codex(), Gemini(), Dsh(), ZCode(), Oracle(), Mock(), Command())}
# Harnesses with a model and a pinned CLI in their own agent image
HARNESSES = tuple(name for name, agent in REGISTRY.items() if agent.protocols)

def run_label(harness, model=None):
    """Top-level runs/ directory of a harness and its model configuration (the harness default when none is given):
    runs/<label>/<category>/<case>/<run_id>. Harnesses without a model (oracle, mock, command) use their own name."""
    model = model or get_agent(harness).default_model
    return f'{harness}-{model}' if model else harness

def run_dir(root, harness, model, case, run_id):
    """A run's record directory: <root>/<harness>-<model>/<category>/<case>/<run_id> (spec/categories.yaml)."""
    from pathlib import Path
    from catalog import category_of
    return Path(root) / run_label(harness, model) / category_of(case) / case / run_id

def get_agent(name):
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; choose {', '.join(REGISTRY)}") from None
