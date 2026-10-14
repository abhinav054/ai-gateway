import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from ai_gateway import app as gateway
from ai_gateway.storage import SQLiteKeyStore


class GatewayManagementTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_assigns_id_and_preserves_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteKeyStore(Path(folder) / "gateway.db")
            store.initialize()
            with patch.object(gateway.app.state, "key_store", store, create=True), patch.dict(os.environ, {"GATEWAY_ADMIN_TOKEN": "admin"}):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://gateway", headers={"X-Admin-Token": "admin"}) as client:
                    created = await client.post("/admin/gateways", json={"name": "test", "routes": [{"name": "primary", "provider": "openai", "base_url": "https://api.openai.com", "api_key": "provider-secret"}]})
                    self.assertEqual(created.status_code, 200)
                    definition = created.json()
                    gateway_id = definition["id"]
                    self.assertTrue(gateway_id.startswith("gw-"))
                    self.assertNotIn("provider-secret", created.text)
                    definition["guardrails"] = [{"name": "budget", "type": "token_usage_limit", "enabled": True, "config": {"max_total_tokens": 100}}]
                    updated = await client.put(f"/admin/gateways/{gateway_id}", json=definition)
                    self.assertEqual(updated.status_code, 200)
                    self.assertNotIn("provider-secret", updated.text)
                    self.assertEqual(store.get_gateway(gateway_id).routes[0].api_key, "provider-secret")
                    fetched = await client.get(f"/admin/gateways/{gateway_id}")
                    self.assertEqual(fetched.json()["guardrails"][0]["config"]["max_total_tokens"], 100)
                    definition["routes"][0]["base_url"] = "https://changed.example"
                    rejected = await client.put(f"/admin/gateways/{gateway_id}", json=definition)
                    self.assertEqual(rejected.status_code, 400)
                    self.assertEqual(store.get_gateway(gateway_id).routes[0].base_url, "https://api.openai.com")
                    denied = await client.delete(f"/admin/gateways/{gateway_id}", headers={"X-Admin-Token": "wrong"})
                    self.assertEqual(denied.status_code, 401)
                    deleted = await client.delete(f"/admin/gateways/{gateway_id}")
                    self.assertEqual(deleted.status_code, 204)
                    self.assertEqual(store.get_gateway(gateway_id).routes, [])
