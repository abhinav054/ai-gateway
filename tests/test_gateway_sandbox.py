from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.datastructures import Headers

from ai_gateway import app as gateway
from ai_gateway.config import Gateway
from ai_gateway.keys import create_and_store_key, hash_key
from ai_gateway.storage import SQLiteKeyStore, SQLiteUsageStore


def definition(gateway_id="team-a", **config):
    return Gateway.model_validate({
        "id": gateway_id,
        "routes": [
            {"name": "one", "api_key": f"{gateway_id}-secret-1", "base_url": "https://upstream.example"},
            {"name": "two", "api_key": f"{gateway_id}-secret-2", "base_url": "https://upstream.example"},
        ],
        "guardrails": [{"name": "budget", "config": {"max_total_tokens": 100}}],
        "config": config,
    })


class GatewaySandboxTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "gateway.db"
        self.store = SQLiteKeyStore(self.path)
        self.store.initialize()
        self.sandbox = definition()
        self.store.upsert_gateway(self.sandbox)
        self.token = create_and_store_key(store=self.store, gateway_id=self.sandbox.id)
        self.key = self.store.get_key(hash_key(self.token))
        self.usage = SQLiteUsageStore(self.path)
        self.usage.initialize()
        self.request = SimpleNamespace(
            headers=Headers(), state=SimpleNamespace(gateway=self.sandbox),
            app=SimpleNamespace(state=SimpleNamespace(
                key_store=self.store, usage_store=self.usage, rate_limited_until={},
            )),
        )

    def test_gateway_and_key_membership_persist(self):
        reopened = SQLiteKeyStore(self.path)
        self.assertEqual(reopened.get_gateway("team-a"), self.sandbox)
        self.assertEqual(reopened.get_key(hash_key(self.token)).gateway_id, "team-a")
        self.assertNotIn("secret", json.dumps(self.sandbox.public_dict()))

    def test_keys_cannot_join_missing_gateway_or_bypass_routes(self):
        with self.assertRaisesRegex(ValueError, "Unknown gateway"):
            create_and_store_key(store=self.store, gateway_id="missing")
        with self.assertRaisesRegex(ValueError, "omit upstream credentials"):
            create_and_store_key(store=self.store, gateway_id="team-a", api_key="bypass")

    def test_same_route_names_are_isolated_between_gateways(self):
        second = definition("team-b")
        self.store.upsert_gateway(second)
        second_token = create_and_store_key(store=self.store, gateway_id="team-b")
        second_key = self.store.get_key(hash_key(second_token))
        self.request.headers = Headers({"x-llm-route": "one"})
        first_route = gateway.resolve_upstream(self.request, self.key, "session")
        self.request.state.gateway = second
        second_route = gateway.resolve_upstream(self.request, second_key, "session")
        self.assertEqual(first_route.api_key, "team-a-secret-1")
        self.assertEqual(second_route.api_key, "team-b-secret-1")

    def test_global_routes_are_inaccessible(self):
        self.store.upsert_route("global", "custom", "global-secret", "https://global.example")
        self.request.headers = Headers({"x-llm-route": "global"})
        with self.assertRaises(HTTPException) as exc:
            gateway.resolve_upstream(self.request, self.key, "session")
        self.assertEqual(exc.exception.status_code, 400)

    def test_empty_gateway_does_not_fall_back_to_global_credentials(self):
        empty = Gateway(id="team-a")
        self.store.upsert_gateway(empty)
        self.request.state.gateway = empty
        with self.assertRaises(HTTPException) as exc:
            gateway.resolve_upstream(self.request, self.key)
        self.assertEqual(exc.exception.status_code, 503)

    def test_load_balancing_can_be_disabled(self):
        self.request.state.gateway = definition(load_balancing_enabled=False, default_route="two")
        for session in (None, "first", "second"):
            self.assertEqual(gateway.resolve_upstream(self.request, self.key, session).name, "two")
        self.request.headers = Headers({"x-llm-route": "one"})
        self.assertEqual(gateway.resolve_upstream(self.request, self.key).name, "one")
        self.request.headers = Headers({"x-llm-route": "one,two"})
        with self.assertRaises(HTTPException):
            gateway.resolve_upstream(self.request, self.key)

    def test_session_affinity_can_be_disabled(self):
        self.request.state.gateway = definition(session_aware=False)
        with patch.object(gateway.secrets, "choice", side_effect=lambda routes: routes[1]) as choice:
            self.assertEqual(gateway.resolve_upstream(self.request, self.key, "session").name, "two")
            choice.assert_called_once()

    def test_only_gateway_guardrails_are_evaluated_and_can_be_disabled(self):
        self.usage.record_usage(
            key_hash=self.key.key_hash, method="POST", path="/v1/chat/completions",
            model="test", status_code=200, input_tokens=100, output_tokens=0,
            total_tokens=100, request_id=None,
        )
        with patch.object(gateway, "load_guardrails", side_effect=AssertionError("Global guardrails leaked")):
            response = gateway.enforce_guardrails(self.request, self.key.key_hash, b"{}")
            self.assertEqual(response.status_code, 429)
            self.request.state.gateway = definition(guardrails_enabled=False)
            self.assertIsNone(gateway.enforce_guardrails(self.request, self.key.key_hash, b"{}"))
            self.request.state.gateway = definition("team-b")
            self.assertIsNone(gateway.enforce_guardrails(self.request, "other-key", b"{}"))

    def test_rate_limit_settings_and_fallback_are_gateway_specific(self):
        self.request.state.gateway = definition(rate_limiting={"default_retry_after_seconds": 120})
        gateway.remember_upstream_rate_limit(self.request, "team-a:rate", 429, httpx.Headers())
        until = self.request.app.state.rate_limited_until["team-a:rate"]
        self.assertGreater(until, gateway.utc_now() + timedelta(seconds=119))
        self.assertEqual(gateway.enforce_upstream_rate_limit(self.request, "team-a:rate").status_code, 429)
        self.assertIsNone(gateway.enforce_upstream_rate_limit(self.request, "team-b:rate"))
        self.request.state.gateway = definition(rate_limiting={"enabled": False})
        self.assertIsNone(gateway.enforce_upstream_rate_limit(self.request, "team-a:rate"))
        gateway.remember_upstream_rate_limit(self.request, "disabled", 429, httpx.Headers())
        self.assertNotIn("disabled", self.request.app.state.rate_limited_until)

    def test_invalid_gateway_definitions_are_rejected(self):
        for config in (
            {"load_balancing_enabled": "false"}, {"default_route": "missing"},
            {"rate_limiting": {"default_retry_after_seconds": 0}}, {"unknown_setting": True},
        ):
            with self.subTest(config=config), self.assertRaises(ValidationError):
                definition(**config)
        invalid = self.sandbox.model_dump()
        invalid["routes"].append(invalid["routes"][0])
        with self.assertRaises(ValidationError):
            Gateway.model_validate(invalid)

    def test_existing_database_migrates_without_losing_keys(self):
        legacy_path = Path(self.directory.name) / "legacy.db"
        with sqlite3.connect(legacy_path) as conn:
            conn.execute("CREATE TABLE gateway_keys (key_hash TEXT UNIQUE, name TEXT)")
            conn.execute("INSERT INTO gateway_keys VALUES ('legacy-hash', 'legacy')")
        store = SQLiteKeyStore(legacy_path)
        store.initialize()
        store.initialize()
        self.assertEqual(store.get_key("legacy-hash").name, "legacy")
        self.assertIsNone(store.get_key("legacy-hash").gateway_id)


class GatewayAdminTests(unittest.IsolatedAsyncioTestCase):
    async def test_admin_creation_key_binding_and_proxy_isolation(self):
        calls = []

        async def upstream(request):
            calls.append(request)
            return httpx.Response(200, json={"ok": True})

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteKeyStore(Path(directory) / "gateway.db")
            store.initialize()
            usage = SQLiteUsageStore(store.path)
            usage.initialize()
            with patch.object(gateway.app.state, "key_store", store, create=True), \
                 patch.object(gateway.app.state, "usage_store", usage, create=True), \
                 patch.dict("os.environ", {"GATEWAY_ADMIN_TOKEN": "admin"}):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://gateway") as client:
                    payload = definition().model_dump()
                    response = await client.post("/admin/gateways", json=payload)
                    self.assertEqual(response.status_code, 401)
                    response = await client.post("/admin/gateways", json=payload, headers={"x-admin-token": "admin"})
                    self.assertEqual(response.status_code, 200)
                    self.assertNotIn("api_key", response.json()["routes"][0])
                    response = await client.get("/admin/gateways/team-a", headers={"x-admin-token": "admin"})
                    self.assertEqual(response.json(), definition().public_dict())
                    response = await client.post("/admin/keys?gateway_id=team-a", headers={"x-admin-token": "admin"})
                    self.assertEqual(response.status_code, 200)
                    token = response.json()["key"]
                    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
                        with patch.object(gateway.app.state, "http", upstream_client, create=True), \
                             patch.object(gateway.app.state, "rate_limited_until", {}, create=True):
                            headers = {"authorization": f"Bearer {token}", "gateway-session": "session"}
                            for _ in range(2):
                                response = await client.post("/v1/chat/completions", headers=headers, json={"model": "test"})
                                self.assertEqual(response.status_code, 200)
                            response = await client.post(
                                "/v1/chat/completions", headers=headers, json={"model": "test", "stream": True},
                            )
                            self.assertEqual(response.status_code, 200)
                            headers["x-llm-route"] = "foreign"
                            response = await client.post("/v1/chat/completions", headers=headers, json={"model": "test"})
                            self.assertEqual(response.status_code, 400)
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertEqual(calls[0].headers["authorization"], call.headers["authorization"])
            self.assertNotIn("gateway-session", call.headers)

    async def test_proxy_cooldowns_do_not_cross_gateway_boundaries(self):
        calls = []

        async def upstream(request):
            calls.append(request)
            return httpx.Response(429, json={"error": "rate limited"}, headers={"retry-after": "60"})

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteKeyStore(Path(directory) / "gateway.db")
            store.initialize()
            tokens = []
            for gateway_id in ("team-a", "team-b"):
                sandbox = Gateway.model_validate({
                    "id": gateway_id,
                    "routes": [{"name": "shared", "api_key": "same-upstream-key", "base_url": "https://upstream.example"}],
                })
                store.upsert_gateway(sandbox)
                tokens.append(create_and_store_key(store=store, gateway_id=gateway_id))
            usage = SQLiteUsageStore(store.path)
            usage.initialize()
            async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as upstream_client:
                with patch.object(gateway.app.state, "key_store", store, create=True), \
                     patch.object(gateway.app.state, "usage_store", usage, create=True), \
                     patch.object(gateway.app.state, "http", upstream_client, create=True), \
                     patch.object(gateway.app.state, "rate_limited_until", {}, create=True):
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://gateway") as client:
                        for token in (tokens[0], tokens[0], tokens[1]):
                            response = await client.post(
                                "/v1/chat/completions", headers={"authorization": f"Bearer {token}"}, json={"model": "test"},
                            )
                            self.assertEqual(response.status_code, 429)
        self.assertEqual(len(calls), 2)
