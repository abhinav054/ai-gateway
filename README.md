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

To store a guardrail in the database, set `GATEWAY_ADMIN_TOKEN`, then call:

```bash
curl -X POST "http://localhost:8000/admin/guardrails" \
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

To force a local SQLite file from the CLI:

```bash
python -m ai_gateway.keys create --name local-dev --sqlite-path gateway_keys.db
```

## Run

```bash
export OPENAI_API_KEY=sk-your-real-openai-key
uvicorn ai_gateway.app:app --host 0.0.0.0 --port 8000
```

## Use with the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="gw-...",
    base_url="http://localhost:8000/v1",
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
curl -X POST "http://localhost:8000/admin/keys?name=app-1" \
  -H "X-Admin-Token: $GATEWAY_ADMIN_TOKEN"
```
