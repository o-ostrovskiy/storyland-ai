"""Dependency-pinning guards.

The agent chain is built and tested against the google-adk 2.x line. The
pin in pyproject.toml is ``google-adk[db]>=2.4.0,<3`` so a clean
install/CI/deploy resolves a 2.x release deterministically and can never
silently float to a 3.x major (which would be an untested, breaking API
jump — exactly what the old ``<2`` cap prevented for a year).

These tests fail loudly if the *installed* ADK ever drifts off the
intended 2.x line, turning a latent supply-chain risk into a red test.
"""

from importlib import metadata

import pytest
from packaging.version import Version


def _adk_version() -> Version:
    return Version(metadata.version("google-adk"))


def test_google_adk_major_is_two():
    """A 3.x (or a 1.x downgrade) resolve is an untested jump — fail the build."""
    assert _adk_version().major == 2, (
        f"google-adk resolved to {_adk_version()}; the project is pinned to the "
        "2.x line (>=2.4.0,<3). Any other major is an untested, breaking jump — "
        "re-run the migration spike before moving the pin."
    )


def test_google_adk_at_or_above_floor():
    """Stay at/above the floor the agent chain was validated on."""
    assert _adk_version() >= Version("2.4.0"), (
        f"google-adk resolved to {_adk_version()}, below the tested 2.4.0 floor."
    )
