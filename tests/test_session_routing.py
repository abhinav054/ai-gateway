from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import HTTPException
from starlette.datastructures import Headers
from starlette.requests import Request

from ai_gateway import app as gateway
from ai_gateway.storage import GatewayKey, SQLiteKeyStore


class SessionRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SQLiteKeyStore(Path(self.directory.name) / "keys.db")
        self.store.initialize()
        for name in ("a", "b", "c"):
            self.store.upsert_route(name, "custom", f"key-{name}", "https://upstream.example")
        self.key = GatewayKey("client-hash", None, None, None, None)
        self.request = SimpleNamespace(
            headers=Headers(),
            app=SimpleNamespace(state=SimpleNamespace(key_store=self.store)),
        )

    def test_sessions_are_sticky_across_store_instances_and_spread_across_routes(self) -> None:
        assignments = {
            str(index): gateway.resolve_upstream(self.request, self.key, str(index)).name
            for index in range(300)
        }
        self.assertEqual(set(assignments.values()), {"a", "b", "c"})
        self.request.app.state.key_store = SQLiteKeyStore(self.store.path)
        for session, name in assignments.items():
            self.assertEqual(gateway.resolve_upstream(self.request, self.key, session).name, name)

    def test_route_pool_is_order_independent(self) -> None:
        self.request.headers = Headers({"x-llm-route": "a,b"})
        first = gateway.resolve_upstream(self.request, self.key, "session-1")
        self.request.headers = Headers({"x-llm-route": "b,a"})
        self.assertEqual(first, gateway.resolve_upstream(self.request, self.key, "session-1"))
        self.assertIn(first.name, {"a", "b"})

    def test_unknown_route_is_rejected(self) -> None:
        self.request.headers = Headers({"x-llm-route": "a,missing"})
        with self.assertRaises(HTTPException):
            gateway.resolve_upstream(self.request, self.key, "session-1")

    def test_no_session_uses_random_selection(self) -> None:
        with patch.object(gateway.secrets, "choice", side_effect=lambda routes: routes[1]):
            self.assertEqual(gateway.resolve_upstream(self.request, self.key).name, "b")

    def test_bound_key_and_empty_pool_preserve_existing_routing(self) -> None:
        bound = GatewayKey("hash", None, "custom", "bound-key", "https://bound.example")
        self.assertEqual(gateway.resolve_upstream(self.request, bound, "session"), bound)
        with patch.object(self.store, "list_routes", return_value=[]):
            self.assertEqual(gateway.resolve_upstream(self.request, self.key, "session"), self.key)

    def test_session_header_is_validated_and_not_forwarded(self) -> None:
        self.request.headers = Headers({"Gateway-Session": "session-1", "content-type": "application/json"})
        self.assertEqual(gateway.session_parameter(self.request), "session-1")
        route = self.store.get_route("a")
        self.assertNotIn("gateway-session", gateway.upstream_headers(self.request, route))
        self.request.headers = Headers({"gateway-session": " "})
        with self.assertRaises(HTTPException):
            gateway.session_parameter(self.request)

    def test_rate_limits_are_scoped_to_credentials(self) -> None:
        first = gateway.model_rate_limit_key({"model": "model"}, "https://upstream.example", "key-a")
        second = gateway.model_rate_limit_key({"model": "model"}, "https://upstream.example", "key-b")
        self.assertNotEqual(first, second)
        self.assertNotIn("key-a", first)


class ProxySessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_consumes_header_and_preserves_body_and_query(self) -> None:
        calls = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"ok": True})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with tempfile.TemporaryDirectory() as directory:
                store = SQLiteKeyStore(Path(directory) / "keys.db")
                store.initialize()
                store.upsert_route("a", "custom", "key-a", "https://upstream.example")
                key = GatewayKey("hash", None, None, None, None)
                body = b'{"model":"test","messages":[]}'

                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}

                state = SimpleNamespace(http=client, key_store=store, rate_limited_until={})
                request = Request({
                    "type": "http", "method": "POST", "path": "/v1/chat/completions",
                    "query_string": b"extra=1&extra=2",
                    "headers": [(b"gateway-session", b"conversation-1"), (b"content-type", b"application/json")],
                    "app": SimpleNamespace(state=state),
                }, receive)
                with patch.object(gateway, "require_gateway_key", return_value=key), \
                     patch.object(gateway, "GUARDRAILS_ENABLED", False), \
                     patch.object(gateway, "track_usage"):
                    response = await gateway.proxy_openai("chat/completions", request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls[0].content, body)
        self.assertEqual(calls[0].url.query, b"extra=1&extra=2")
        self.assertNotIn("gateway-session", calls[0].headers)
        self.assertEqual(calls[0].headers["authorization"], "Bearer key-a")
