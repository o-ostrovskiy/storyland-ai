"""
Unit test configuration.

Patches create_app() so every app instance skips the gateway secret check.
Unit tests exercise the API directly without going through the gateway,
so no X-Internal-Secret header is present.
"""

import pytest


@pytest.fixture(autouse=True)
def _patch_app_dependency(monkeypatch):
    """Override verify_gateway_secret to a no-op on every app instance."""
    import api.app as app_module
    from api.dependencies import verify_gateway_secret

    original_create_app = app_module.create_app

    def patched_create_app():
        app = original_create_app()
        app.dependency_overrides[verify_gateway_secret] = lambda: None
        return app

    monkeypatch.setattr(app_module, "create_app", patched_create_app)
