# AI Gateway

OpenAI-compatible Python gateway. Clients use a gateway-issued key with the OpenAI SDK, and the gateway forwards requests to OpenAI using your real `OPENAI_API_KEY`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in your environment or `.env`.

## Gateway sandboxes

A `Gateway` object owns its upstream routes, client-key membership, guardrails,
and routing configuration. Gateway-scoped client keys can access only that
gateway's routes. Route names and guardrail names are local to each gateway.
Existing keys without a `gateway_id` keep the legacy global configuration.

Edit the upstream keys in `gateway.example.json`, then create the sandbox and
a client key:

```bash
python -m ai_gateway.keys gateway --file gateway.example.json
python -m ai_gateway.keys create --gateway-id my-app --name app-client
```

Both commands support `--sqlite-path`. Otherwise they use the configured SQLite
or PostgreSQL key store. The gateway definition is persisted in `gateway_objects`;
existing key databases are migrated automatically to add gateway membership.

You can also create or replace the full definition through the admin API:

```bash
curl -X POST http://localhost:8001/admin/gateways \
  -H "X-Admin-Token: $GATEWAY_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @gateway.example.json

curl -X POST "http://localhost:8001/admin/keys?gateway_id=my-app&name=app-client" \
  -H "X-Admin-Token: $GATEWAY_ADMIN_TOKEN"

curl http://localhost:8001/admin/gateways/my-app \
  -H "X-Admin-Token: $GATEWAY_ADMIN_TOKEN"
```

Admin responses omit upstream API keys. Updating a definition replaces its routes,
guardrails, and settings; provide the complete definition, including upstream
credentials. Existing client-key membership is preserved. Global `/admin/routes`
and `/admin/guardrails` manage only legacy resources.

The `config` fields are:

| Field | Behavior |
| --- | --- |
| `load_balancing_enabled` | Balance across the gateway's routes, or the pool selected by `X-LLM-Route`. Defaults to `true`. |
| `session_aware` | Use `Gateway-Session` for sticky selection when balancing is enabled. Defaults to `true`. |
| `default_route` | Route used when balancing is disabled and no explicit route is selected. If omitted, use the alphabetically first route. |
| `guardrails_enabled` | Evaluate only this gateway's guardrails. Defaults to `true`. |
| `rate_limiting.enabled` | Cache and enforce upstream 429 cooldowns. Defaults to `true`. |
| `rate_limiting.default_retry_after_seconds` | Fallback cooldown when the upstream supplies no retry/reset header. Defaults to `60`. |

With balancing disabled, `X-LLM-Route` may select one local route; multi-route
pools return 400. A gateway without routes returns 503 rather than falling back
to global upstream credentials. Requests cannot change gateway membership using
headers; the bearer key determines the sandbox. `Gateway-Session` is never forwarded.

Token-budget guardrails apply per client key using tracked usage. Scoped gateways
do not inherit global guardrails or `.gateway` files. Keep usage tracking enabled
when enforcing token budgets. Upstream cooldowns are scoped by gateway, upstream
credentials, and model. Cooldown state remains in memory per worker and resets
on restart; this setting controls upstream cooldowns, not a requests-per-minute
quota. The sandbox is a configuration and routing boundary, not process isolation.

## Key storage

SQLite is the default key database:

```bash
export GATEWAY_KEY_STORE=sqlite
export GATEWAY_SQLITE_PATH=gateway_keys.db
```

For PostgreSQL:

```bash
export GATEWAY_KEY_STORE=postgres
export GATEWAY_POSTGRES_DSN=postgresql://user:password@localhost:5432/ai_gateway
```

When using PostgreSQL, Redis can cache successful key lookups:

```bash
export GATEWAY_REDIS_URL=redis://localhost:6379/0
export GATEWAY_REDIS_CACHE_TTL_SECONDS=300
```

The gateway creates the `gateway_keys` table automatically on startup or when creating a key.

## Token usage tracking

Usage tracking is enabled by default. The gateway records token counts returned by OpenAI in the response `usage` field.

```bash
export GATEWAY_USAGE_TRACKING_ENABLED=true
```

By default, usage records use the same backend as key storage. To configure it separately:

```bash
export GATEWAY_USAGE_STORE=sqlite
export GATEWAY_USAGE_SQLITE_PATH=gateway_keys.db
```

or:

```bash
export GATEWAY_USAGE_STORE=postgres
export GATEWAY_USAGE_POSTGRES_DSN=postgresql://user:password@localhost:5432/ai_gateway
```

Set `GATEWAY_USAGE_TRACKING_ENABLED=false` to disable usage tracking. The gateway creates a `token_usage` table with the gateway key hash, method, path, model, status code, input tokens, output tokens, total tokens, upstream request id, and timestamp.

## Guardrails

Guardrails are enabled by default and are loaded from both the configured database and JSON files in `.gateway`.

```bash
export GATEWAY_GUARDRAILS_ENABLED=true
export GATEWAY_GUARDRAILS_DIR=.gateway
```

By default, stored guardrails use the same backend as key storage. To configure them separately:

```bash
export GATEWAY_GUARDRAIL_STORE=sqlite
export GATEWAY_GUARDRAIL_SQLITE_PATH=gateway_keys.db
```

or:

```bash
export GATEWAY_GUARDRAIL_STORE=postgres
export GATEWAY_GUARDRAIL_POSTGRES_DSN=postgresql://user:password@localhost:5432/ai_gateway
```

JSON guardrails can be a single object or a list of objects:

```json
{
  "name": "monthly-token-cap",
  "type": "token_usage_limit",
  "enabled": true,
  "config": {
    "max_total_tokens": 1000000,
    "include_request_estimate": true,
    "message": "This gateway key has reached its token usage limit."
  }
}
```

The special `token_usage_limit` guardrail checks the current gateway key's tracked `total_tokens` before forwarding a request. If `include_request_estimate` is true, it also counts the request's `max_output_tokens`, `max_completion_tokens`, or `max_tokens` against the projected total.

## Upstream model rate limits

When an upstream model returns `429 Too Many Requests`, the gateway remembers the model cooldown and rejects later requests for that model locally until the retry window expires. This prevents the gateway from continuing to send requests to a model that has already rate limited the organization.

Groq-compatible rate limits are detected from the 429 status and `retry-after` header. If `retry-after` is missing, the gateway falls back to Groq-style `x-ratelimit-reset-requests` and `x-ratelimit-reset-tokens` duration headers, then to a default retry window.

```bash
export GATEWAY_RATE_LIMIT_CACHE_ENABLED=true
export GATEWAY_RATE_LIMIT_DEFAULT_RETRY_AFTER_SECONDS=60
```

To store a guardrail in the database, set `GATEWAY_ADMIN_TOKEN`, then call:

```bash
curl -X POST "http://localhost:8001/admin/guardrails" \
  -H "X-Admin-Token: $GATEWAY_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "monthly-token-cap",
    "type": "token_usage_limit",
    "enabled": true,
    "config": {
      "max_total_tokens": 1000000,
      "include_request_estimate": true
    }
  }'
```

## Create a gateway key

```bash
python -m ai_gateway.keys create --name local-dev
```

The command prints the only plaintext copy of the gateway key and stores its hash in the configured database.

Keys can optionally store the upstream provider API key and base URL used for routing:

```bash
python -m ai_gateway.keys create \
  --name groq-dev \
  --provider groq \
  --api-key gsk-your-real-groq-key
```

Supported OpenAI-compatible provider presets are `openai`, `groq`, `openrouter`, `together`, and `custom`. Pass `--base-url` to override a preset or route to another OpenAI-compatible endpoint:

```bash
python -m ai_gateway.keys create \
  --name local-llm \
  --provider custom \
  --api-key local-key \
  --base-url http://localhost:11434
```

If a gateway key does not have stored upstream credentials, requests fall back to `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

## Header-based routing

You can also keep one client-facing gateway key and route each request to a named upstream config with `X-LLM-Route`.

Create the shared gateway key:

```bash
python -m ai_gateway.keys create --name shared-client-key
```

Create named route configs:

```bash
python -m ai_gateway.keys route \
  --name groq \
  --provider groq \
  --api-key gsk-your-real-groq-key

python -m ai_gateway.keys route \
  --name local \
  --provider custom \
  --api-key local-key \
  --base-url http://localhost:11434
```

Then clients use the same gateway key and select an upstream per request:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer gw-..." \
  -H "X-LLM-Route: groq" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"Hello"}]}'
```

The route header name defaults to `X-LLM-Route` and can be changed with `GATEWAY_ROUTE_HEADER`.

## Session-aware load balancing

Gateway keys without bound upstream credentials automatically balance across all
named routes. Register each upstream API key as a separate named route using the
CLI or `/admin/routes`. Routes in a pool should support the requested model.
Keys with bound upstream credentials continue using those credentials unless a
route header is supplied.

Send `Gateway-Session` to keep a conversation on the same route and upstream key:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer gw-..." \
  -H "Gateway-Session: conversation-123" \
  -H "X-LLM-Route: groq-key-1,groq-key-2" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"Hello"}]}'
```

`X-LLM-Route` optionally selects a comma-separated pool; a single name selects
one route. Without `Gateway-Session`, requests use random balancing. The session
header is consumed by the gateway and never forwarded, including for streaming.
Blank session headers return 400. Affinity is scoped to the authenticated gateway
key and is consistent across workers and restarts with the same route pool.
Adding or removing routes can reassign sessions. A rate-limited session stays on
its assigned key and receives 429 until cooldown expires; cooldowns are scoped
to upstream credentials and model. Requests are not automatically retried.

To force a local SQLite file from the CLI:

```bash
python -m ai_gateway.keys create --name local-dev --sqlite-path gateway_keys.db
```

## Run

```bash
export OPENAI_API_KEY=sk-your-real-openai-key
uvicorn ai_gateway.app:app --host 0.0.0.0 --port 8001
```

## Use with the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="gw-...",
    base_url="http://localhost:8001/v1",
)

response = client.responses.create(
    model="gpt-5-mini",
    input="Say hello through the gateway.",
)

print(response.output_text)
```

The same `base_url` pattern works for other OpenAI SDK calls, including chat completions and embeddings.

## Optional HTTP key creation

Set `GATEWAY_ADMIN_TOKEN`, then call:

```bash
curl -X POST "http://localhost:8001/admin/keys?name=app-1" \
  -H "X-Admin-Token: $GATEWAY_ADMIN_TOKEN"
```

To create a key that routes to a specific provider:

```bash
curl -X POST "http://localhost:8001/admin/keys?name=groq-dev&provider=groq&api_key=gsk-your-real-groq-key" \
  -H "X-Admin-Token: $GATEWAY_ADMIN_TOKEN"
```

To create or update a named route config:

```bash
curl -X POST "http://localhost:8001/admin/routes" \
  -H "X-Admin-Token: $GATEWAY_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "groq",
    "provider": "groq",
    "api_key": "gsk-your-real-groq-key"
  }'
```
