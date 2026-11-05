"""The test suite must not be able to export to Langfuse.

common.config runs load_dotenv() at import, which walks up to the real .env. On
2026-11-04 that sent 16 traces of fixture spans from a unit test run to the
production project. The guard is the block at the top of tests/conftest.py.
"""

import http.server
import json
import os
import runpy
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from tests.conftest import LANGFUSE_CONNECTION_VARS, LANGFUSE_EXPORT_OPT_IN

REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs beside a throwaway .env, so a bare load_dotenv() finds that file the way
# common.config finds the real one. Loading it before common.config matters for
# the control run: every variable is already set, so the real .env cannot apply.
PROBE = """
import json, os
import tests.conftest
from dotenv import load_dotenv
assert load_dotenv(), "the throwaway .env was not found"

from common.config import load_config
from langfuse import Langfuse
from plugins.langfuse_plugin import LangfusePlugin

config = load_config()
plugin = LangfusePlugin(
    secret_key=config.langfuse_secret_key,
    public_key=config.langfuse_public_key,
    host=config.langfuse_host,
)

# Explicit keys are the case blanking the environment cannot stop.
client = Langfuse(
    public_key="pk-lf-fake",
    secret_key="sk-lf-fake",
    host=os.environ["PROBE_LISTENER_URL"],
)
with client.start_as_current_observation(name="guard-probe"):
    pass
client.flush()
client.shutdown()
print("PROBE_RESULT=" + json.dumps({
    "config_public_key": config.langfuse_public_key,
    "plugin_enabled": plugin.enabled,
}))
"""


@pytest.fixture
def listener():
    """A loopback stand-in for Langfuse that records each POST path."""
    paths = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            paths.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", paths
    server.shutdown()
    server.server_close()
    thread.join()


def _run_probe(tmp_path, url, *, opt_in):
    """Run PROBE in a clean process and return what it reported."""
    (tmp_path / ".env").write_text(
        "LANGFUSE_SECRET_KEY=sk-lf-from-dotenv\n"
        "LANGFUSE_PUBLIC_KEY=pk-lf-from-dotenv\n"
        f"LANGFUSE_HOST={url}\n"
        f"LANGFUSE_BASE_URL={url}\n"
    )
    probe = tmp_path / "probe.py"
    probe.write_text(PROBE)

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LANGFUSE_") and key != LANGFUSE_EXPORT_OPT_IN
    }
    env.update(
        PYTHONPATH=str(REPO_ROOT),
        PROBE_LISTENER_URL=url,
        NO_PROXY="127.0.0.1",
        no_proxy="127.0.0.1",
        GOOGLE_API_KEY="test-key",
    )
    if opt_in:
        env[LANGFUSE_EXPORT_OPT_IN] = "1"

    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(
        line for line in result.stdout.splitlines() if line.startswith("PROBE_RESULT=")
    )
    return json.loads(line.removeprefix("PROBE_RESULT="))


def test_this_run_has_no_langfuse_connection_settings():
    for var in LANGFUSE_CONNECTION_VARS:
        assert os.environ[var] == ""
    assert os.environ["LANGFUSE_TRACING_ENABLED"] == "false"


def test_guard_holds_with_a_dotenv_present(tmp_path, listener):
    url, posts = listener

    result = _run_probe(tmp_path, url, opt_in=False)

    assert result == {"config_public_key": "", "plugin_enabled": False}
    assert posts == []


def test_opt_in_exports_so_the_probe_can_detect_a_broken_guard(tmp_path, listener):
    url, posts = listener

    result = _run_probe(tmp_path, url, opt_in=True)

    assert result == {"config_public_key": "pk-lf-from-dotenv", "plugin_enabled": True}
    assert any(path.endswith("/otel/v1/traces") for path in posts), posts


def test_unit_suite_refuses_the_opt_in(monkeypatch):
    monkeypatch.setenv(LANGFUSE_EXPORT_OPT_IN, "1")

    with pytest.raises(pytest.UsageError, match="tests/integration only"):
        runpy.run_path(str(REPO_ROOT / "tests" / "unit" / "conftest.py"))
