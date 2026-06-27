"""Fail-closed identity guards for ``get_gateway_user_id``.

Regression coverage for the unverified-signature-JWT / shared-``dev_user``
foot-gun: ``api/dependencies.get_gateway_user_id`` previously fell back to an
*unverified* JWT claim (or a shared ``dev_user``) whenever ``INTERNAL_API_SECRET``
was unset, so a forged token — or a single misconfigured deploy — could read or
write another user's sessions/itineraries (everything is scoped by ``user_id``).

The hardened contract these tests pin:
  * trust ONLY the gateway-set ``X-User-ID`` header,
  * never parse/trust the raw Authorization JWT,
  * fall back to ``dev_user`` ONLY behind the explicit local ``ALLOW_DEV_USER`` flag,
  * otherwise fail closed with HTTP 403 (never a shared id).
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import api.dependencies as deps


@pytest.fixture
def install_state(monkeypatch):
    """Install a stub AppState whose config exposes the auth-relevant fields."""

    def _install(internal_api_secret: str = "", allow_dev_user: bool = False):
        config = SimpleNamespace(
            internal_api_secret=internal_api_secret,
            allow_dev_user=allow_dev_user,
        )
        monkeypatch.setattr(deps, "get_app_state", lambda: SimpleNamespace(config=config))

    return _install


def test_returns_trusted_x_user_id(install_state):
    """Present X-User-ID is the identity (the only trusted source)."""
    install_state(internal_api_secret="s3cret")
    assert deps.get_gateway_user_id(x_user_id="alice") == "alice"


def test_missing_identity_fails_closed_when_secret_set(install_state):
    """No X-User-ID with the secret set -> 403 (unchanged prod behaviour)."""
    install_state(internal_api_secret="s3cret")
    with pytest.raises(HTTPException) as exc:
        deps.get_gateway_user_id(x_user_id=None)
    assert exc.value.status_code == 403


def test_missing_identity_fails_closed_when_secret_unset(install_state):
    """The core fix: secret unset must NOT collapse to a shared id — fail closed."""
    install_state(internal_api_secret="", allow_dev_user=False)
    with pytest.raises(HTTPException) as exc:
        deps.get_gateway_user_id(x_user_id=None)
    assert exc.value.status_code == 403


def test_dev_user_only_behind_explicit_flag(install_state):
    """``dev_user`` is reachable only when ALLOW_DEV_USER is explicitly true."""
    install_state(internal_api_secret="", allow_dev_user=True)
    assert deps.get_gateway_user_id(x_user_id=None) == "dev_user"


def test_x_user_id_wins_even_with_dev_flag(install_state):
    """A real trusted id always beats the dev fallback."""
    install_state(internal_api_secret="", allow_dev_user=True)
    assert deps.get_gateway_user_id(x_user_id="bob") == "bob"


def test_unverified_jwt_helper_is_removed():
    """The unverified-signature JWT path must be gone entirely."""
    assert not hasattr(deps, "_user_from_jwt")
