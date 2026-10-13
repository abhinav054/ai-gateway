from __future__ import annotations

import unittest
from datetime import timedelta
from types import SimpleNamespace

import httpx

from ai_gateway import app as gateway


class RateLimitTests(unittest.TestCase):
    def test_uses_retry_after_header(self) -> None:
        headers = httpx.Headers({"retry-after": "2"})

        self.assertEqual(gateway.rate_limit_retry_after_seconds(headers), 2.0)

    def test_parses_groq_reset_duration_headers(self) -> None:
        headers = httpx.Headers(
            {
                "x-ratelimit-reset-requests": "2m59.56s",
                "x-ratelimit-reset-tokens": "7.66s",
            }
        )

        self.assertEqual(gateway.rate_limit_retry_after_seconds(headers), 7.66)

    def test_model_rate_limit_key_prefers_request_model(self) -> None:
        self.assertEqual(
            gateway.model_rate_limit_key({"model": "openai/gpt-oss-120b"}),
            "upstream:https://api.openai.com:model:openai/gpt-oss-120b",
        )

    def test_enforces_active_cached_rate_limit(self) -> None:
        limited_until = gateway.utc_now() + timedelta(seconds=30)
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(rate_limited_until={"model:test": limited_until}))
        )

        response = gateway.enforce_upstream_rate_limit(request, "model:test")

        self.assertIsNotNone(response)
        assert response is not None
        self.assertEqual(response.status_code, 429)
        self.assertIn("retry-after", response.headers)

    def test_expired_rate_limit_is_removed(self) -> None:
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(rate_limited_until={"model:test": gateway.utc_now() - timedelta(seconds=1)})
            )
        )

        response = gateway.enforce_upstream_rate_limit(request, "model:test")

        self.assertIsNone(response)
        self.assertEqual(request.app.state.rate_limited_until, {})


if __name__ == "__main__":
    unittest.main()
