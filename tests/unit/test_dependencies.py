"""Dependency-pinning guards.

The agent chain is built and tested against the google-adk 1.x line. The
pin in pyproject.toml is ``google-adk[eval]>=1.33.0,<2`` so a clean
install/CI/deploy resolves a 1.x release deterministically and can never
silently float to a 2.x major (which is an untested, breaking API jump).

These tests fail loudly if the *installed* ADK ever drifts off the
intended 1.x line, turning a latent supply-chain risk into a red test.
"""

from importlib import metadata

import pytest
from packaging.version import Version


def _adk_version() -> Version:
    return Version(metadata.version("google-adk"))


def test_google_adk_major_is_one():
    """A 2.x (or higher) resolve is an untested major jump — fail the build."""
    assert _adk_version().major == 1, (
        f"google-adk resolved to {_adk_version()}; the project is pinned to the "
        "1.x line (>=1.33.0,<2). A 2.x major is an untested, breaking jump — "
        "see the ADK-pin opportunity before raising the ceiling."
    )


def test_google_adk_at_or_above_floor():
    """Stay at/above the floor the agent chain was validated on."""
    assert _adk_version() >= Version("1.33.0"), (
        f"google-adk resolved to {_adk_version()}, below the tested 1.33.0 floor."
    )
