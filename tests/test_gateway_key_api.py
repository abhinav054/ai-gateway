import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from ai_gateway import app as gateway
from ai_gateway.config import Gateway
from ai_gateway.keys import create_and_store_key, hash_key
from ai_gateway.storage import PostgresKeyStore, SQLiteKeyStore


class GatewayKeyAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_enable_issue_scope_revoke_and_disable(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteKeyStore(Path(folder) / "gateway.db")
            store.initialize()
            sandbox = Gateway.model_validate({"id": "one", "routes": [{"name": "primary", "provider": "openai", "base_url": "https://api.openai.com", "api_key": "test-provider"}]})
            store.upsert_gateway(sandbox)
            other = sandbox.model_copy(update={"id": "two"})
            other.config = other.config.model_copy(update={"key_api_enabled": True})
            store.upsert_gateway(other)
            with patch.object(gateway.app.state, "key_store", store, create=True), patch.dict(os.environ, {"GATEWAY_ADMIN_TOKEN": "admin"}):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://gateway", headers={"X-Admin-Token": "admin"}) as client:
                    path = "/admin/gateways/one/keys"
                    denied = await client.post(path, json={"name": "client"}, headers={"X-Admin-Token": "wrong"})
                    self.assertEqual(denied.status_code, 401)
                    disabled = await client.post(path, json={"name": "client"})
                    self.assertEqual(disabled.status_code, 403)
                    self.assertEqual(store.list_keys("one"), [])

                    sandbox.config.key_api_enabled = True
                    store.upsert_gateway(sandbox)
                    invalid = await client.post(path, json={"name": " "})
                    self.assertEqual(invalid.status_code, 400)
                    response = await client.post(path, json={"name": "client"})
                    self.assertEqual(response.status_code, 201)
                    self.assertEqual(response.headers["cache-control"], "no-store")
                    issued = response.json()
                    self.assertEqual(issued["id"], hash_key(issued["key"]))
                    self.assertIsNotNone(store.get_key(issued["id"]))
                    wrong_scope = await client.delete(f"/admin/gateways/two/keys/{issued['id']}")
                    self.assertEqual(wrong_scope.status_code, 404)
                    self.assertIsNotNone(store.get_key(issued["id"]))
                    sandbox.config.key_api_enabled = False
                    store.upsert_gateway(sandbox)
                    blocked = await client.delete(f"{path}/{issued['id']}")
                    self.assertEqual(blocked.status_code, 403)
                    sandbox.config.key_api_enabled = True
                    store.upsert_gateway(sandbox)
                    revoked = await client.delete(f"{path}/{issued['id']}")
                    self.assertEqual(revoked.status_code, 204)
                    self.assertIsNone(store.get_key(issued["id"]))
                    rejected = await client.post("/v1/chat/completions", json={}, headers={"Authorization": f"Bearer {issued['key']}"})
                    self.assertEqual(rejected.status_code, 401)
                    self.assertEqual(store.list_keys("one"), [])

    async def test_admin_cleanup_works_with_key_api_disabled(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteKeyStore(Path(folder) / "gateway.db")
            store.initialize()
            store.upsert_gateway(Gateway(id="one"))
            key = create_and_store_key(name="session", store=store, gateway_id="one")
            with patch.object(gateway.app.state, "key_store", store, create=True), patch.dict(os.environ, {"GATEWAY_ADMIN_TOKEN": "admin"}):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://gateway") as client:
                    path = f"/admin/keys/{hash_key(key)}?gateway_id=one"
                    denied = await client.delete(path)
                    self.assertEqual(denied.status_code, 401)
                    response = await client.delete(path, headers={"X-Admin-Token": "admin"})
                    self.assertEqual(response.status_code, 204)
                    self.assertIsNone(store.get_key(hash_key(key)))

    def test_postgres_revocation_invalidates_redis(self):
        store = PostgresKeyStore("test-dsn")
        store._redis = MagicMock()
        with patch("ai_gateway.storage.psycopg.connect") as connect:
            connection = connect.return_value.__enter__.return_value
            connection.execute.return_value.rowcount = 1
            self.assertTrue(store.revoke_key("one", "key-hash"))
            connection.execute.assert_called_once_with("DELETE FROM gateway_keys WHERE gateway_id = %s AND key_hash = %s", ("one", "key-hash"))
            store._redis.delete.assert_called_once_with(store.cache_key("key-hash"))
