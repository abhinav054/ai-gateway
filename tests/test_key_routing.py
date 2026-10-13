from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from starlette.datastructures import Headers

from ai_gateway import app as gateway
from ai_gateway.keys import create_and_store_key, hash_key, upsert_routing_config
from ai_gateway.storage import SQLiteKeyStore


class KeyRoutingTests(unittest.TestCase):
    def test_create_key_stores_provider_api_key_and_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteKeyStore(Path(directory) / "keys.db")

            gateway_key = create_and_store_key(
                name="groq-dev",
                store=store,
                provider="groq",
                api_key="gsk-test",
            )

            stored = store.get_key(hash_key(gateway_key))

        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored.name, "groq-dev")
        self.assertEqual(stored.provider, "groq")
        self.assertEqual(stored.api_key, "gsk-test")
        self.assertEqual(stored.base_url, "https://api.groq.com/openai")

    def test_provider_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteKeyStore(Path(directory) / "keys.db")

            with self.assertRaisesRegex(ValueError, "api_key is required"):
                create_and_store_key(
                    store=store,
                    provider="groq",
                )

    def test_custom_provider_requires_base_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteKeyStore(Path(directory) / "keys.db")

            with self.assertRaisesRegex(ValueError, "base_url is required"):
                create_and_store_key(
                    store=store,
                    provider="custom",
                    api_key="local-key",
                )

    def test_upsert_route_stores_routing_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteKeyStore(Path(directory) / "keys.db")

            upsert_routing_config(
                name="groq",
                provider="groq",
                api_key="gsk-test",
                store=store,
            )

            route = store.get_route("groq")

        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.name, "groq")
        self.assertEqual(route.provider, "groq")
        self.assertEqual(route.api_key, "gsk-test")
        self.assertEqual(route.base_url, "https://api.groq.com/openai")

    def test_route_header_selects_named_routing_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteKeyStore(Path(directory) / "keys.db")
            store.initialize()
            store.upsert_route(
                name="groq",
                provider="groq",
                api_key="gsk-test",
                base_url="https://api.groq.com/openai",
            )
            request = SimpleNamespace(
                headers=Headers({"x-llm-route": "groq"}),
                app=SimpleNamespace(state=SimpleNamespace(key_store=store)),
            )
            gateway_key = store.get_route("missing")

            route = gateway.resolve_upstream(
                request,
                gateway.GatewayKey(
                    key_hash="hash",
                    name=None,
                    provider=None,
                    api_key=None,
                    base_url=None,
                ),
            )

        self.assertIsNone(gateway_key)
        self.assertEqual(route.name, "groq")
        self.assertEqual(gateway.upstream_api_key(route), "gsk-test")
        self.assertEqual(gateway.upstream_base_url(route), "https://api.groq.com/openai")


if __name__ == "__main__":
    unittest.main()
