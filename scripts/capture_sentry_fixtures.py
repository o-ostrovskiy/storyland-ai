"""Capture the raw Sentry events this service emits, for the scrub backstop test.

ai-private#74: the backstop test (tests/unit/test_sentry_scrub_backstop.py)
feeds the scrubber REAL events, not hand-built dicts, because a hand-built dict
can only contain fields we already knew about. This script makes those events:

- the real Sentry SDK (default integrations: FastAPI/Starlette, HTTPX, logging),
  tracing on, and a transport that records envelopes instead of sending them
  (the DSN is a dummy; nothing leaves the machine);
- a request transaction: one synthetic GET to the real app, with a query
  string, through the SDK's ASGI instrumentation;
- a fetch transaction + an error event: the real google-genai client makes a
  streaming Gemini call (the outbound HTTP this service makes) against an
  in-process httpx MockTransport, so no Gemini spend and no network; then the
  service's own structlog bridge logs an error, which becomes the event that
  carries the HTTP breadcrumbs;
- events are written exactly as the SDK serializes them, BEFORE any scrubber
  runs, to tests/fixtures/sentry/.

Synthetic only: no reader data, the API key is the literal "fake-key", the
text is made up. Re-run after a sentry-sdk or google-genai bump:

    .venv/bin/python scripts/capture_sentry_fixtures.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.pop("SENTRY_DSN", None)  # create_app must not init the SDK itself
os.environ.setdefault("INTERNAL_API_SECRET", "fixture-secret")

import httpx  # noqa: E402
import sentry_sdk  # noqa: E402
from sentry_sdk.transport import Transport  # noqa: E402

OUT = REPO / "tests" / "fixtures" / "sentry"

captured: list[dict] = []


class _Recorder(Transport):
    def capture_envelope(self, envelope):
        for item in envelope.items:
            if item.type in ("event", "transaction"):
                captured.append(json.loads(item.get_bytes()))

    def flush(self, timeout, callback=None):
        return None

    def kill(self):
        return None


def _gemini_stub(request: httpx.Request) -> httpx.Response:
    chunk = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "A synthetic answer."}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 3, "totalTokenCount": 6},
    }
    body = f"data: {json.dumps(chunk)}\r\n\r\n".encode()
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


async def _request_visit():
    from api.app import create_app

    app = create_app()
    # The executor is not initialized here, so the route 500s: the request
    # transaction is still recorded, as in prod.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8080") as client:
        resp = await client.get(
            "/api/v1/itinerary/synthetic-fixture-job/status?book_title=Synthetic+Fixture+Novel",
            headers={"X-Internal-Secret": "fixture-secret", "X-User-ID": "fixture-user"},
        )
        print("status route:", resp.status_code)


async def _gemini_call():
    from google import genai
    from google.genai import types

    from common.logging import get_logger

    stub = httpx.AsyncClient(transport=httpx.MockTransport(_gemini_stub))
    client = genai.Client(
        api_key="fake-key",
        http_options=types.HttpOptions(httpx_async_client=stub),
    )
    with sentry_sdk.start_transaction(op="task", name="fixture.gemini_stream"):
        stream = await client.aio.models.generate_content_stream(
            model="gemini-2.5-flash", contents="A synthetic fixture prompt."
        )
        async for _ in stream:
            pass
    # A failure after the call: the service's structlog bridge turns an error
    # log into a Sentry event, and that event ships the HTTP breadcrumbs.
    get_logger("fixture").error("fixture_capture_error", phase="discovery")


def main() -> None:
    from common.logging import configure_logging

    sentry_sdk.init(
        dsn="https://public@o0.ingest.sentry.io/0",
        transport=_Recorder,
        traces_sample_rate=1.0,
        send_default_pii=False,
        max_request_body_size="never",
        include_local_variables=False,
        environment="fixture-capture",
        release="fixture",
        server_name="fixture-host",
    )
    configure_logging(level="INFO")
    asyncio.run(_request_visit())
    asyncio.run(_gemini_call())
    sentry_sdk.flush()

    for e in captured:
        print(e.get("type", "event"), e.get("transaction"))
    errors = [e for e in captured if e.get("type", "event") == "event"]
    request_tx = [e for e in captured if e.get("type") == "transaction" and e.get("request")]
    fetch_tx = [e for e in captured if e.get("transaction") == "fixture.gemini_stream"]
    assert errors and request_tx and fetch_tx, [e.get("type") for e in captured]

    OUT.mkdir(parents=True, exist_ok=True)
    # The one edit to the captured JSON: this checkout's absolute path (it holds
    # the local username) becomes "<repo>". Nothing else is touched.
    for name, event in (
        ("transaction_status_request.json", request_tx[0]),
        ("transaction_gemini_fetch.json", fetch_tx[0]),
        ("error_after_gemini_call.json", errors[-1]),
    ):
        text = json.dumps(event, indent=2, sort_keys=True).replace(str(REPO), "<repo>")
        (OUT / name).write_text(text + "\n")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
