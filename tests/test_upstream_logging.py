import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient

from ai_gateway import app as gateway
from ai_gateway.storage import GatewayKey


class UpstreamTests(unittest.TestCase):
    def setUp(self):
        self.key = GatewayKey(key_hash="hash", name="test", provider="custom",
                              api_key="private-upstream-secret", base_url="https://bedrock.example/v1")
        gateway.app.state.key_store = SimpleNamespace(get_key=lambda _: self.key)
        gateway.app.state.rate_limited_until = {}
        gateway.app.state.usage_store = Mock()
        self.client = TestClient(gateway.app)
        self.headers = {"Authorization": "Bearer client-secret"}
        self.guardrail = patch.object(gateway, "enforce_guardrails", return_value=None)
        self.guardrail.start()
        self.addCleanup(self.guardrail.stop)
        self.addCleanup(self.client.close)

    def upstream(self, handler):
        gateway.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def test_url_conventions(self):
        for base in ["https://example.com", "https://example.com/v1", "https://example.com/v1/"]:
            self.assertEqual(gateway.upstream_url(base,"chat/completions"),"https://example.com/v1/chat/completions")
        self.assertEqual(gateway.upstream_url("https://example.com/openai/v1", "chat/completions"),
                         "https://example.com/openai/v1/chat/completions")

    def test_mantle_aliases_are_scoped_to_mantle(self):
        for model in ["openai.gpt-oss-20b-1:0", "openai/gpt-oss-20b"]:
            self.assertEqual(gateway.mantle_model_id("https://bedrock-mantle.ap-south-1.api.aws/v1", model), "openai.gpt-oss-20b")
            self.assertEqual(gateway.mantle_model_id("https://api.groq.com/openai", model), model)
        self.assertEqual(gateway.mantle_model_id("https://bedrock-mantle.fake.api.aws.attacker.test", "openai.gpt-oss-20b-1:0"), "openai.gpt-oss-20b-1:0")

    def test_running_session_runtime_id_is_translated(self):
        from dataclasses import replace
        self.key = replace(self.key, base_url="https://bedrock-mantle.ap-south-1.api.aws/v1")
        def handler(req):
            self.assertEqual(json.loads(req.content)["model"], "openai.gpt-oss-20b")
            return httpx.Response(200, json={"choices": [], "usage": {"total_tokens": 1}})
        self.upstream(handler)
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            response = self.client.post("/v1/chat/completions", headers=self.headers, json={"model": "openai.gpt-oss-20b-1:0"})
        self.assertEqual(response.status_code, 200)
        record = json.loads(logs.records[-1].getMessage())
        self.assertEqual(record["model"], "openai.gpt-oss-20b")
        self.assertEqual(record["requested_model"], "openai.gpt-oss-20b-1:0")

    def test_bedrock_harmony_suffix_requires_exact_advertised_tool(self):
        request = {"model": "openai.gpt-oss-20b", "tools": [
            {"type": "function", "function": {"name": "read_file"}}]}
        def body(name):
            return json.dumps({"choices": [{"message": {"tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": name, "arguments": '{"path":"/workspace/file"}'}}]}}]}).encode()
        for channel in ["commentary", "analysis", "final"]:
            result = gateway.normalize_bedrock_tool_names(body("read_file<|channel|>" + channel), request,
                                                        "https://bedrock-mantle.ap-south-1.api.aws/v1")
            call = json.loads(result)["choices"][0]["message"]["tool_calls"][0]
            self.assertEqual(call["function"]["name"], "read_file")
            self.assertEqual(call["id"], "call-1")
            self.assertEqual(call["function"]["arguments"], '{"path":"/workspace/file"}')
        for name in ["delete_file<|channel|>commentary", "read_file<|channel|>unknown", "read_file<|channel|>commentary extra", "read_file"]:
            original = body(name)
            self.assertEqual(gateway.normalize_bedrock_tool_names(original, request, "https://bedrock-mantle.ap-south-1.api.aws/v1"), original)
        original = body("read_file<|channel|>commentary")
        for base in ["https://api.openai.com/v1", "https://bedrock-mantle.fake.api.aws.attacker.test"]:
            self.assertEqual(gateway.normalize_bedrock_tool_names(original, request, base), original)
        self.assertEqual(gateway.normalize_bedrock_tool_names(original, {**request, "model": "other-model"}, "https://bedrock-mantle.ap-south-1.api.aws"), original)

    def test_normalized_tool_response_preserves_usage_and_content_length(self):
        from dataclasses import replace
        self.key = replace(self.key, base_url="https://bedrock-mantle.ap-south-1.api.aws/v1")
        self.upstream(lambda _: httpx.Response(200, headers={"etag": "stale"}, json={
            "choices": [{"message": {"tool_calls": [{"id": "call-1", "type": "function", "function": {
                "name": "read_file<|channel|>commentary", "arguments": '{"path":"/workspace/file"}'}}]}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}}))
        response = self.client.post("/v1/chat/completions", headers=self.headers, json={
            "model": "openai.gpt-oss-20b", "tools": [{"type": "function", "function": {"name": "read_file"}}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(response.json()["usage"]["total_tokens"], 9)
        self.assertEqual(int(response.headers["content-length"]), len(response.content))
        self.assertNotIn("etag", response.headers)

    def test_nonstream_status_latency_and_tokens(self):
        def handler(req):
            self.assertEqual(req.url.path, "/v1/chat/completions")
            return httpx.Response(200,json={"model":"test","choices":[{"message":{"content":"private reply"}}],
                "usage":{"prompt_tokens":0,"completion_tokens":12,"total_tokens":12}})
        self.upstream(handler)
        with self.assertLogs("uvicorn.error",level="INFO") as logs:
            response=self.client.post("/v1/chat/completions",headers=self.headers,json={"model":"test","messages":[{"role":"user","content":"private prompt"}]})
        self.assertEqual(response.status_code,200)
        record=json.loads(logs.records[-1].getMessage())
        self.assertEqual(record["input_tokens"],0)
        self.assertEqual(record["output_tokens"],12)
        self.assertEqual(record["status"],200)
        self.assertGreaterEqual(record["duration_ms"],0)
        self.assertNotIn("private",json.dumps(record))
        self.assertNotIn("client-secret",json.dumps(record))

    def test_explicit_stream_false_reaches_upstream_including_model_alias(self):
        from dataclasses import replace
        self.key = replace(self.key, base_url="https://bedrock-mantle.ap-south-1.api.aws/v1")
        for model in ["deepseek.v3.1", "deepseek.v3-v1:0"]:
            with self.subTest(model=model):
                expected = {"model": "deepseek.v3.1", "stream": False,
                            "messages": [{"role": "user", "content": "test"}]}
                def handler(req):
                    self.assertEqual(json.loads(req.content), expected)
                    self.assertNotIn("stream_options", json.loads(req.content))
                    return httpx.Response(200, json={"choices": [{
                        "finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [{
                            "id": "complete", "type": "function", "function": {
                                "name": "run_command", "arguments": '{"command":"echo ok"}'}}]}}]})
                self.upstream(handler)
                with self.assertLogs("uvicorn.error", level="INFO") as logs:
                    response = self.client.post("/v1/chat/completions",
                        headers={**self.headers, "Accept": "application/json"},
                        json={**expected, "model": model})
                self.assertEqual(response.status_code, 200)
                self.assertIn("application/json", response.headers["content-type"])
                self.assertEqual(response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
                                 '{"command":"echo ok"}')
                self.assertFalse(json.loads(logs.records[-1].getMessage())["stream"])

    def test_stream_error_retains_real_status(self):
        self.upstream(lambda _:httpx.Response(404,json={"error":"missing"}))
        with self.assertLogs("uvicorn.error",level="INFO") as logs:
            response=self.client.post("/v1/chat/completions",headers=self.headers,json={"model":"test","stream":True})
        self.assertEqual(response.status_code,404)
        record=json.loads(logs.records[-1].getMessage())
        self.assertEqual(record["status"],404)
        self.assertIsNone(record["total_tokens"])
        self.assertEqual(record["error_body"], '"missing"')
        self.assertEqual(logs.records[-1].levelname, "ERROR")

    def test_provider_errors_include_diagnostics_without_credentials_or_request(self):
        body = {"error": {"message": "Model failed private-upstream-secret client-secret",
                          "type": "server_error", "code": "internal_error",
                          "api_key": "other-secret"}, "messages": ["private prompt"]}
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                self.upstream(lambda _: httpx.Response(500, json=body,
                    headers={"x-amzn-requestid": "bedrock-request-1"}))
                with self.assertLogs("uvicorn.error", level="ERROR") as logs:
                    response = self.client.post("/v1/chat/completions", headers=self.headers,
                        json={"model": "test", "stream": streaming,
                              "messages": [{"role": "user", "content": "private prompt"}]})
                self.assertEqual(response.json(), body)
                record = json.loads(logs.records[-1].getMessage())
                self.assertEqual(record["request_id"], "bedrock-request-1")
                self.assertEqual(record["error"], "upstream_http_error")
                self.assertIn("internal_error", record["error_body"])
                self.assertFalse(record["error_body_truncated"])
                for secret in ("private-upstream-secret", "client-secret", "other-secret", "private prompt"):
                    self.assertNotIn(secret, json.dumps(record))

    def test_plaintext_error_is_bounded_and_bearer_token_redacted(self):
        self.upstream(lambda _: httpx.Response(502, text="Bearer unknown-secret " + "x" * 5000))
        with self.assertLogs("uvicorn.error", level="ERROR") as logs:
            response = self.client.post("/v1/chat/completions", headers=self.headers, json={"model": "test"})
        record = json.loads(logs.records[-1].getMessage())
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(record["error_body"]), 4096)
        self.assertTrue(record["error_body_truncated"])
        self.assertNotIn("unknown-secret", record["error_body"])

    def test_stream_usage_across_chunk_boundaries(self):
        class Chunks(httpx.AsyncByteStream):
            async def __aiter__(self):
                for chunk in [b'data: {"choices":[]}\n\ndata: {"mo',
                              b'del":"test","usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}\n\n',
                              b'data: [DONE]\n\n']:
                    yield chunk
        def handler(req):
            self.assertTrue(json.loads(req.content)["stream_options"]["include_usage"])
            return httpx.Response(200,headers={"content-type":"text/event-stream"},stream=Chunks())
        self.upstream(handler)
        with self.assertLogs("uvicorn.error",level="INFO") as logs:
            response=self.client.post("/v1/chat/completions",headers=self.headers,json={"model":"test","stream":True})
        self.assertEqual(response.status_code,200)
        self.assertIn(b"[DONE]",response.content)
        record=json.loads(logs.records[-1].getMessage())
        self.assertEqual(record["total_tokens"],10)
        gateway.app.state.usage_store.record_usage.assert_called_once()

    def test_transport_error_is_logged_without_secrets(self):
        def handler(req):
            raise httpx.ConnectError("private-upstream-secret",request=req)
        self.upstream(handler)
        with self.assertLogs("uvicorn.error",level="INFO") as logs:
            response=self.client.post("/v1/chat/completions",headers=self.headers,json={"model":"test"})
        self.assertEqual(response.status_code,502)
        record=json.loads(logs.records[-1].getMessage())
        self.assertEqual(record["error"],"ConnectError")
        self.assertIsNone(record["status"])
        self.assertNotIn("private-upstream-secret",json.dumps(record))


if __name__ == "__main__":
    unittest.main()
