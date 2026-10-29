"""
One-way sync of agent prompts from git to Langfuse prompt management.

Git (agents/prompts/vN.json) stays the source of truth: the app never reads
prompts from Langfuse, so nothing here can change production behaviour. This
only mirrors the *effective* instruction of each agent (tone guardrail
included) into Langfuse so prompts can be opened in the playground and
compared there.

Each agent becomes a text prompt named ``storyland-ai/<agent>`` labelled with
the prompt version (e.g. ``v3``) — never ``production``, which would suggest
Langfuse serves it. Python ``{placeholder}`` fields are rewritten as Langfuse
``{{placeholder}}`` variables so they are fillable in the playground. A prompt
whose content already matches the version-labelled copy is skipped, so re-runs
create no new Langfuse versions.

Usage:
    python -m evaluation.tools.sync_prompts_to_langfuse            # sync current version
    python -m evaluation.tools.sync_prompts_to_langfuse --dry-run  # show what would change
"""

import argparse
import dataclasses
import os
import re
import subprocess
import sys
from typing import Any

from dotenv import load_dotenv

from agents.prompts import CURRENT_PROMPT_VERSION, load_prompts, prompt_content_hash
from common.logging import get_logger

logger = get_logger("storyland.sync_prompts")

PROMPT_NAME_PREFIX = "storyland-ai/"

# A Python str.format field ({book_title}). The prompts contain no literal
# braces ("{{"), so every match is a real placeholder.
_PLACEHOLDER = re.compile(r"(?<!\{)\{([a-z_]+)\}(?!\})")


def to_langfuse_template(instruction: str) -> tuple[str, list[str]]:
    """Rewrite ``{name}`` fields as Langfuse ``{{name}}`` variables."""
    variables = sorted(set(_PLACEHOLDER.findall(instruction)))
    return _PLACEHOLDER.sub(r"{{\1}}", instruction), variables


def _git_sha() -> str:
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha[:12]
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def build_payloads(version: str = CURRENT_PROMPT_VERSION) -> list[dict[str, Any]]:
    """One create_prompt() payload per agent for ``version``. No network."""
    prompt_hash = prompt_content_hash(version)
    sha = _git_sha()
    payloads = []
    for agent, instruction in dataclasses.asdict(load_prompts(version)).items():
        template, variables = to_langfuse_template(instruction)
        payloads.append({
            "name": f"{PROMPT_NAME_PREFIX}{agent}",
            "prompt": template,
            "type": "text",
            "labels": [version],
            "tags": ["git-synced"],
            "config": {
                "source": "git",
                "prompt_version": version,
                "prompt_hash": prompt_hash,
                "git_sha": sha,
                "variables": variables,
            },
            "commit_message": f"sync from git {sha} ({version}, {prompt_hash})",
        })
    return payloads


def _current_text(client, name: str, label: str) -> str | None:
    """The prompt text under ``label``, or None if that prompt/label doesn't exist."""
    from langfuse.api import NotFoundError

    try:
        return client.get_prompt(name, label=label, cache_ttl_seconds=0).prompt
    except NotFoundError:
        return None


def sync(client, version: str = CURRENT_PROMPT_VERSION, dry_run: bool = False) -> dict[str, list[str]]:
    """Create a Langfuse prompt version for every agent whose text changed."""
    result: dict[str, list[str]] = {"created": [], "unchanged": []}
    for payload in build_payloads(version):
        name = payload["name"]
        if _current_text(client, name, version) == payload["prompt"]:
            result["unchanged"].append(name)
            continue
        if not dry_run:
            client.create_prompt(**payload)
        result["created"].append(name)
        logger.info("prompt_synced", name=name, version=version, dry_run=dry_run)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--version", default=CURRENT_PROMPT_VERSION, help="prompt version to sync")
    parser.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = parser.parse_args(argv)

    load_dotenv()
    if not (os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY")):
        print("LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY not set", file=sys.stderr)
        return 2

    from langfuse import Langfuse

    client = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    result = sync(client, args.version, dry_run=args.dry_run)
    verb = "would create" if args.dry_run else "created"
    print(f"{args.version}: {verb} {len(result['created'])}, unchanged {len(result['unchanged'])}")
    for name in result["created"]:
        print(f"  + {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
