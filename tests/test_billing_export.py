import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import ai_gateway.app as gateway
from ai_gateway.storage import SQLiteUsageStore


class BillingExportTests(unittest.IsolatedAsyncioTestCase):
    async def test_usage_export_requires_admin_and_contains_only_accounting_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            usage = SQLiteUsageStore(Path(directory) / 'usage.db')
            usage.initialize()
            usage.record_usage(key_hash='never-export-this-hash', method='POST', path='/v1/chat/completions',
                               model='test-model', status_code=200, input_tokens=12, output_tokens=8,
                               total_tokens=20, request_id='private-request', gateway_id='gateway-7', key_name='agent-7')
            # Initializing an existing database preserves both data and new columns.
            usage.initialize()
            with patch.object(gateway.app.state, 'usage_store', usage, create=True), patch.dict('os.environ', {'GATEWAY_ADMIN_TOKEN': 'test-admin'}):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url='http://test') as client:
                    self.assertEqual((await client.get('/admin/billing/usage')).status_code, 401)
                    headers = {'x-admin-token': 'test-admin'}
                    response = await client.get('/admin/billing/usage', headers=headers)
                    self.assertEqual(response.status_code, 200)
                    item = response.json()['items'][0]
                    self.assertEqual(item['gateway_id'], 'gateway-7')
                    self.assertEqual(item['total_tokens'], 20)
                    self.assertNotIn('key_hash', item)
                    self.assertNotIn('request_id', item)
                    next_page = await client.get(f"/admin/billing/usage?after_id={item['id']}", headers=headers)
                    self.assertEqual(next_page.json()['items'], [])
                    self.assertEqual((await client.get('/admin/billing/usage?limit=501', headers=headers)).status_code, 400)
