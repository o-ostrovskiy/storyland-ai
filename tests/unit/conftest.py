"""
Unit test configuration.

Patches create_app() so every app instance skips gateway auth checks.
Unit tests exercise the API directly without going through the gateway,
so no X-Internal-Secret or X-User-ID headers are present.
"""

import pytest


@pytest.fixture(autouse=True)
def _patch_app_dependency(monkeypatch):
    """Override gateway dependencies to no-ops for unit tests."""
    import api.app as app_module
    from api.dependencies import get_gateway_user_id, verify_gateway_secret

    original_create_app = app_module.create_app

    def patched_create_app():
        app = original_create_app()
        app.dependency_overrides[verify_gateway_secret] = lambda: None
        app.dependency_overrides[get_gateway_user_id] = lambda: "test_user"
        return app

    monkeypatch.setattr(app_module, "create_app", patched_create_app)
