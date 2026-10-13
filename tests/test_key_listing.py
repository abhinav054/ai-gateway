import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from ai_gateway import app as gateway
from ai_gateway.config import Gateway
from ai_gateway.storage import SQLiteKeyStore


class KeyListingTests(unittest.IsolatedAsyncioTestCase):
    async def test_listing_is_scoped_and_does_not_expose_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteKeyStore(Path(folder) / "keys.db")
            store.initialize()
            store.upsert_gateway(Gateway(id="one"))
            store.upsert_gateway(Gateway(id="two"))
            store.add_key_hash("hash-one", name="First", api_key="upstream-secret", gateway_id="one")
            store.add_key_hash("hash-two", name="Second", gateway_id="two")
            with patch.object(gateway.app.state, "key_store", store, create=True), patch.dict(os.environ, {"GATEWAY_ADMIN_TOKEN": "admin-secret"}):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://gateway") as client:
                    denied = await client.get("/admin/keys", params={"gateway_id": "one"})
                    self.assertEqual(denied.status_code, 401)
                    response = await client.get("/admin/keys", params={"gateway_id": "one"}, headers={"X-Admin-Token": "admin-secret"})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["cache-control"], "no-store")
                    self.assertEqual(len(response.json()), 1)
                    self.assertEqual(response.json()[0]["name"], "First")
                    self.assertEqual(set(response.json()[0]), {"id", "name", "createdAt"})
                    self.assertNotIn("upstream-secret", response.text)
                    self.assertNotIn("hash-two", response.text)
                    empty = await client.get("/admin/keys", params={"gateway_id": "empty"}, headers={"X-Admin-Token": "admin-secret"})
                    self.assertEqual(empty.json(), [])
