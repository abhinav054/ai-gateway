from __future__ import annotations

import os
import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import Body, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from pydantic import ValidationError

from .guardrails import evaluate_guardrails, load_guardrails
from .config import Gateway, Route
from .keys import create_and_store_key, hash_key, provider_base_url, upsert_routing_config
from .storage import GatewayKey, RoutingConfig, build_guardrail_store, build_key_store, build_usage_store


load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "600"))
RATE_LIMIT_CACHE_ENABLED = os.getenv("GATEWAY_RATE_LIMIT_CACHE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
DEFAULT_RATE_LIMIT_RETRY_AFTER_SECONDS = float(os.getenv("GATEWAY_RATE_LIMIT_DEFAULT_RETRY_AFTER_SECONDS", "60"))
GUARDRAILS_ENABLED = os.getenv("GATEWAY_GUARDRAILS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
GUARDRAILS_DIR = Path(os.getenv("GATEWAY_GUARDRAILS_DIR", ".gateway"))
ROUTE_HEADER = os.getenv("GATEWAY_ROUTE_HEADER", "x-llm-route").lower()

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

FORWARDED_REQUEST_HEADERS = {
    "accept",
    "content-type",
    "openai-beta",
    "openai-organization",
    "openai-project",
    "idempotency-key",
}

RATE_LIMIT_HEADER_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|s|m|h)")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.http = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    app.state.key_store = build_key_store()
    app.state.key_store.initialize()
    app.state.usage_store = build_usage_store()
    app.state.usage_store.initialize()
    app.state.guardrail_store = build_guardrail_store()
    app.state.guardrail_store.initialize()
    app.state.rate_limited_until = {}
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="AI Gateway", lifespan=lifespan)


def require_gateway_key(request: Request) -> GatewayKey:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Missing gateway bearer token")

    key_hash = hash_key(token)
    env_hashes = {item.strip() for item in os.getenv("GATEWAY_KEY_HASHES", "").split(",") if item.strip()}
    stored_key = request.app.state.key_store.get_key(key_hash)
    if stored_key:
        return stored_key
    if key_hash in env_hashes:
        return GatewayKey(
            key_hash=key_hash,
            name=None,
            provider="openai",
            api_key=None,
            base_url=OPENAI_BASE_URL,
        )
    raise HTTPException(status_code=401, detail="Invalid gateway bearer token")


def upstream_headers(request: Request, upstream: GatewayKey | RoutingConfig | Route) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() in FORWARDED_REQUEST_HEADERS and key.lower() != ROUTE_HEADER
    }
    headers["authorization"] = f"Bearer {upstream_api_key(upstream)}"

    if os.getenv("OPENAI_ORGANIZATION"):
        headers["openai-organization"] = os.environ["OPENAI_ORGANIZATION"]
    if os.getenv("OPENAI_PROJECT"):
        headers["openai-project"] = os.environ["OPENAI_PROJECT"]

    return headers


def upstream_api_key(upstream: GatewayKey | RoutingConfig | Route) -> str:
    if upstream.api_key:
        return upstream.api_key
    return required_env("OPENAI_API_KEY")


def upstream_base_url(upstream: GatewayKey | RoutingConfig | Route) -> str:
    if upstream.base_url:
        return upstream.base_url.rstrip("/")
    resolved = provider_base_url(upstream.provider)
    if resolved:
        return resolved.rstrip("/")
    return OPENAI_BASE_URL.rstrip("/")


def resolve_upstream(
    request: Request, gateway_key: GatewayKey, session: str | None = None
) -> GatewayKey | RoutingConfig | Route:
    route_name = request.headers.get(ROUTE_HEADER)
    sandbox = request_gateway(request)
    if gateway_key.gateway_id is not None:
        if sandbox is None:
            sandbox = request.app.state.key_store.get_gateway(gateway_key.gateway_id)
        if sandbox is None or sandbox.id != gateway_key.gateway_id:
            raise HTTPException(status_code=403, detail="Gateway is unavailable")
        available = {route.name: route for route in sandbox.routes}
        if route_name:
            names = list(dict.fromkeys(item.strip() for item in route_name.split(",")))
            if any(name not in available for name in names):
                raise HTTPException(status_code=400, detail="Unknown route in this gateway")
            routes = [available[name] for name in names]
        else:
            routes = list(available.values())
        if not routes:
            raise HTTPException(status_code=503, detail="Gateway has no configured routes")
        if not sandbox.config.load_balancing_enabled:
            if route_name and len(routes) != 1:
                raise HTTPException(status_code=400, detail="Load balancing is disabled; select one route")
            if route_name:
                return routes[0]
            default = sandbox.config.default_route
            return available[default] if default else min(routes, key=lambda route: route.name)
        if not sandbox.config.session_aware:
            session = None
    elif route_name:
        routes = []
        for name in dict.fromkeys(item.strip() for item in route_name.split(",")):
            route = request.app.state.key_store.get_route(name)
            if route is None:
                raise HTTPException(status_code=400, detail=f"Unknown LLM route: {name}")
            routes.append(route)
    elif gateway_key.api_key:
        return gateway_key
    else:
        routes = request.app.state.key_store.list_routes()
    if not routes:
        return gateway_key
    if session is None:
        return secrets.choice(routes)
    # Rendezvous hashing preserves most session assignments when the pool changes.
    return max(
        routes,
        key=lambda route: hash_key(json.dumps([gateway_key.key_hash, session, route.name])),
    )


def request_gateway(request: Request) -> Gateway | None:
    return getattr(getattr(request, "state", None), "gateway", None)


def require_admin_token(token: str | None) -> None:
    expected = os.getenv("GATEWAY_ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=404, detail="Admin gateway management is disabled")
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")


@app.post("/admin/gateways")
async def upsert_gateway(
    body: Gateway,
    x_admin_token: str | None = Header(default=None),
) -> JSONResponse:
    require_admin_token(x_admin_token)
    app.state.key_store.upsert_gateway(body)
    return JSONResponse(body.public_dict())


@app.get("/admin/gateways/{gateway_id}")
async def get_gateway(gateway_id: str, x_admin_token: str | None = Header(default=None)) -> JSONResponse:
    require_admin_token(x_admin_token)
    sandbox = app.state.key_store.get_gateway(gateway_id)
    if sandbox is None:
        raise HTTPException(status_code=404, detail="Unknown gateway")
    return JSONResponse(sandbox.public_dict())


@app.put("/admin/gateways/{gateway_id}")
async def update_gateway(gateway_id: str, body: dict = Body(...), x_admin_token: str | None = Header(default=None)) -> JSONResponse:
    require_admin_token(x_admin_token)
    previous = app.state.key_store.get_gateway(gateway_id)
    if previous is None:
        raise HTTPException(status_code=404, detail="Unknown gateway")
    if body.get("id", gateway_id) != gateway_id:
        raise HTTPException(status_code=400, detail="Gateway ID cannot be changed")
    payload = dict(body, id=gateway_id)
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        raise HTTPException(status_code=400, detail="At least one route is required")
    stored_routes = {route.name: route for route in previous.routes}
    for route in routes:
        if not isinstance(route, dict):
            raise HTTPException(status_code=400, detail="Invalid route")
        if not isinstance(route.get("name"), str) or not isinstance(route.get("base_url"), str):
            raise HTTPException(status_code=400, detail="Invalid route")
        key = route.get("api_key")
        if key is None or isinstance(key, str) and not key.strip():
            stored = stored_routes.get(route.get("name"))
            if stored is None or route.get("provider") != stored.provider or route.get("base_url", "").rstrip("/") != stored.base_url:
                raise HTTPException(status_code=400, detail="An API key is required for new or changed routes")
            route["api_key"] = stored.api_key
    try:
        updated = Gateway.model_validate(payload)
    except ValidationError:
        raise HTTPException(status_code=422, detail="Invalid gateway configuration") from None
    app.state.key_store.upsert_gateway(updated)
    return JSONResponse(updated.public_dict())


@app.delete("/admin/gateways/{gateway_id}")
async def disable_gateway(gateway_id: str, x_admin_token: str | None = Header(default=None)) -> Response:
    require_admin_token(x_admin_token)
    previous = app.state.key_store.get_gateway(gateway_id)
    if previous is None:
        raise HTTPException(status_code=404, detail="Unknown gateway")
    disabled = previous.model_copy(update={"routes": [], "config": previous.config.model_copy(update={"default_route": None})})
    app.state.key_store.upsert_gateway(disabled)
    return Response(status_code=204)


def session_parameter(request: Request) -> str | None:
    session = request.headers.get("gateway-session")
    if session is not None and not session.strip():
        raise HTTPException(status_code=400, detail="gateway-session must be a non-empty string")
    return session


def response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "content-encoding"
    }


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise HTTPException(status_code=500, detail=f"{name} is not configured")
    return value


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/admin/keys")
async def create_key(
    x_admin_token: str | None = Header(default=None),
    name: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    gateway_id: str | None = None,
) -> JSONResponse:
    admin_token = os.getenv("GATEWAY_ADMIN_TOKEN")
    if not admin_token:
        raise HTTPException(status_code=404, detail="Admin key creation is disabled")
    if x_admin_token != admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    try:
        key = create_and_store_key(
            name=name,
            store=app.state.key_store,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            gateway_id=gateway_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({
        "key": key,
        "provider": provider,
        "base_url": provider_base_url(provider, base_url),
        "gateway_id": gateway_id,
    })


@app.get("/admin/keys")
async def list_keys(gateway_id: str, x_admin_token: str | None = Header(default=None)) -> JSONResponse:
    admin_token = os.getenv("GATEWAY_ADMIN_TOKEN")
    if not admin_token:
        raise HTTPException(status_code=404, detail="Admin key listing is disabled")
    if not secrets.compare_digest(x_admin_token or "", admin_token):
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return JSONResponse(app.state.key_store.list_keys(gateway_id), headers={"Cache-Control": "no-store"})


def require_key_api(gateway_id: str, token: str | None) -> Gateway:
    require_admin_token(token)
    sandbox = app.state.key_store.get_gateway(gateway_id)
    if sandbox is None:
        raise HTTPException(status_code=404, detail="Unknown gateway")
    if not sandbox.config.key_api_enabled:
        raise HTTPException(status_code=403, detail="Key API is disabled for this gateway")
    return sandbox


@app.delete("/admin/keys/{key_id}", status_code=204)
async def revoke_admin_key(key_id: str, gateway_id: str, x_admin_token: str | None = Header(default=None)) -> Response:
    require_admin_token(x_admin_token)
    if not app.state.key_store.revoke_key(gateway_id, key_id):
        raise HTTPException(status_code=404, detail="Key not found in this gateway")
    return Response(status_code=204, headers={"Cache-Control": "no-store"})


@app.post("/admin/gateways/{gateway_id}/keys", status_code=201)
async def issue_gateway_key(gateway_id: str, body: dict = Body(...), x_admin_token: str | None = Header(default=None)) -> JSONResponse:
    sandbox = require_key_api(gateway_id, x_admin_token)
    if not sandbox.routes:
        raise HTTPException(status_code=409, detail="Gateway has no active routes")
    name = body.get("name")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100:
        raise HTTPException(status_code=400, detail="Client name is required (maximum 100 characters)")
    key = create_and_store_key(name=name.strip(), store=app.state.key_store, gateway_id=gateway_id)
    return JSONResponse({"key": key, "id": hash_key(key), "gateway_id": gateway_id}, status_code=201, headers={"Cache-Control": "no-store"})


@app.delete("/admin/gateways/{gateway_id}/keys/{key_id}", status_code=204)
async def revoke_gateway_key(gateway_id: str, key_id: str, x_admin_token: str | None = Header(default=None)) -> Response:
    require_key_api(gateway_id, x_admin_token)
    if not app.state.key_store.revoke_key(gateway_id, key_id):
        raise HTTPException(status_code=404, detail="Key not found in this gateway")
    return Response(status_code=204, headers={"Cache-Control": "no-store"})


@app.post("/admin/routes")
async def upsert_route(
    x_admin_token: str | None = Header(default=None),
    body: dict = Body(...),
) -> JSONResponse:
    admin_token = os.getenv("GATEWAY_ADMIN_TOKEN")
    if not admin_token:
        raise HTTPException(status_code=404, detail="Admin route management is disabled")
    if x_admin_token != admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    name = body.get("name")
    provider = body.get("provider", "custom")
    api_key = body.get("api_key")
    base_url = body.get("base_url")
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=400, detail="name is required")
    if provider is not None and not isinstance(provider, str):
        raise HTTPException(status_code=400, detail="provider must be a string")
    if not isinstance(api_key, str) or not api_key:
        raise HTTPException(status_code=400, detail="api_key is required")
    if base_url is not None and not isinstance(base_url, str):
        raise HTTPException(status_code=400, detail="base_url must be a string")

    try:
        upsert_routing_config(
            name=name,
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            store=app.state.key_store,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse({"name": name, "provider": provider, "base_url": provider_base_url(provider, base_url)})


@app.post("/admin/guardrails")
async def upsert_guardrail(
    x_admin_token: str | None = Header(default=None),
    body: dict = Body(...),
) -> JSONResponse:
    admin_token = os.getenv("GATEWAY_ADMIN_TOKEN")
    if not admin_token:
        raise HTTPException(status_code=404, detail="Admin guardrail management is disabled")
    if x_admin_token != admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    name = body.get("name")
    guardrail_type = body.get("type")
    config = body.get("config", {})
    enabled = body.get("enabled", True)
    if not isinstance(name, str) or not name:
        raise HTTPException(status_code=400, detail="name is required")
    if not isinstance(guardrail_type, str) or not guardrail_type:
        raise HTTPException(status_code=400, detail="type is required")
    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")

    app.state.guardrail_store.upsert_guardrail(
        name=name,
        guardrail_type=guardrail_type,
        config=config,
        enabled=bool(enabled),
    )
    return JSONResponse({"status": "ok"})


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_openai(path: str, request: Request) -> Response:
    gateway_key = require_gateway_key(request)
    request.state.gateway = None
    if gateway_key.gateway_id is not None:
        request.state.gateway = request.app.state.key_store.get_gateway(gateway_key.gateway_id)
        if request.state.gateway is None:
            raise HTTPException(status_code=403, detail="Gateway is unavailable")
    body = await request.body()
    request_payload = parse_json_body(body)
    session = session_parameter(request)
    upstream = resolve_upstream(request, gateway_key, session)
    base_url = upstream_base_url(upstream)
    url = f"{base_url}/v1/{path}"
    rate_limit_key = model_rate_limit_key(request_payload, base_url, upstream_api_key(upstream))
    if request.state.gateway is not None:
        rate_limit_key = f"gateway:{request.state.gateway.id}:{rate_limit_key}"
    rate_limit_response = enforce_upstream_rate_limit(request, rate_limit_key)
    if rate_limit_response:
        return rate_limit_response

    guardrail_response = enforce_guardrails(request, gateway_key.key_hash, body)
    if guardrail_response:
        return guardrail_response

    client: httpx.AsyncClient = request.app.state.http

    if wants_streaming_response(request, body):
        return StreamingResponse(
            stream_upstream(client, request, upstream, url, body, rate_limit_key),
            media_type="text/event-stream",
        )

    upstream = await client.request(
        request.method,
        url,
        params=request.query_params.multi_items(),
        content=body,
        headers=upstream_headers(request, upstream),
    )
    track_usage(
        request=request,
        path=f"/v1/{path}",
        key_hash=gateway_key.key_hash,
        response_content=upstream.content,
        response_status=upstream.status_code,
        response_headers=upstream.headers,
    )
    remember_upstream_rate_limit(request, rate_limit_key, upstream.status_code, upstream.headers)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


def model_rate_limit_key(
    request_payload: dict | None, base_url: str | None = None, api_key: str | None = None
) -> str:
    credential = f":key:{hash_key(api_key)}" if api_key else ""
    if request_payload and isinstance(request_payload.get("model"), str):
        return f"upstream:{(base_url or OPENAI_BASE_URL).rstrip('/')}{credential}:model:{request_payload['model']}"
    return f"upstream:{(base_url or OPENAI_BASE_URL).rstrip('/')}{credential}"


def enforce_upstream_rate_limit(request: Request, rate_limit_key: str) -> JSONResponse | None:
    sandbox = request_gateway(request)
    enabled = sandbox.config.rate_limiting.enabled if sandbox else RATE_LIMIT_CACHE_ENABLED
    if not enabled:
        return None

    limited_until = request.app.state.rate_limited_until.get(rate_limit_key)
    now = utc_now()
    if not isinstance(limited_until, datetime) or limited_until <= now:
        request.app.state.rate_limited_until.pop(rate_limit_key, None)
        return None

    retry_after = max(1, int((limited_until - now).total_seconds()))
    return JSONResponse(
        status_code=429,
        headers={"retry-after": str(retry_after)},
        content={
            "error": {
                "code": "upstream_model_rate_limited",
                "message": "Upstream model is rate limited. Requests for this model are paused until the retry window resets.",
                "details": {
                    "rate_limit_key": rate_limit_key,
                    "retry_after_seconds": retry_after,
                    "limited_until": limited_until.isoformat(),
                },
            }
        },
    )


def remember_upstream_rate_limit(
    request: Request,
    rate_limit_key: str,
    status_code: int,
    headers: httpx.Headers,
) -> None:
    sandbox = request_gateway(request)
    enabled = sandbox.config.rate_limiting.enabled if sandbox else RATE_LIMIT_CACHE_ENABLED
    if not enabled or status_code != 429:
        return

    fallback = sandbox.config.rate_limiting.default_retry_after_seconds if sandbox else None
    retry_after = rate_limit_retry_after_seconds(headers, fallback)
    request.app.state.rate_limited_until[rate_limit_key] = utc_now() + timedelta(seconds=retry_after)


def rate_limit_retry_after_seconds(headers: httpx.Headers, fallback_seconds: float | None = None) -> float:
    retry_after = parse_retry_after(headers.get("retry-after"))
    if retry_after is not None:
        return retry_after

    reset_candidates = [
        parse_rate_limit_reset(headers.get("x-ratelimit-reset-requests")),
        parse_rate_limit_reset(headers.get("x-ratelimit-reset-tokens")),
    ]
    reset_seconds = [value for value in reset_candidates if value is not None]
    if reset_seconds:
        return max(1.0, min(reset_seconds))

    return max(1.0, fallback_seconds if fallback_seconds is not None else DEFAULT_RATE_LIMIT_RETRY_AFTER_SECONDS)


def parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None

    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(1.0, seconds)


def parse_rate_limit_reset(value: str | None) -> float | None:
    if not value:
        return None

    total = 0.0
    matched = False
    for match in RATE_LIMIT_HEADER_RE.finditer(value.strip()):
        matched = True
        amount = float(match.group("value"))
        unit = match.group("unit")
        if unit == "ms":
            total += amount / 1000
        elif unit == "s":
            total += amount
        elif unit == "m":
            total += amount * 60
        elif unit == "h":
            total += amount * 3600

    return max(1.0, total) if matched else None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def enforce_guardrails(request: Request, key_hash: str, body: bytes) -> JSONResponse | None:
    sandbox = request_gateway(request)
    enabled = sandbox.config.guardrails_enabled if sandbox else GUARDRAILS_ENABLED
    if not enabled:
        return None

    guardrails = (
        [guardrail.model_dump() for guardrail in sandbox.guardrails]
        if sandbox else load_guardrails(request.app.state.guardrail_store, GUARDRAILS_DIR)
    )
    decision = evaluate_guardrails(
        guardrails=guardrails,
        key_hash=key_hash,
        usage_store=request.app.state.usage_store,
        request_payload=parse_json_body(body),
    )
    if decision.allowed:
        return None

    return JSONResponse(
        status_code=decision.status_code,
        content={
            "error": {
                "code": decision.code,
                "message": decision.message,
                "guardrail": decision.guardrail,
                "details": decision.details or {},
            }
        },
    )


def track_usage(
    *,
    request: Request,
    path: str,
    key_hash: str,
    response_content: bytes,
    response_status: int,
    response_headers: httpx.Headers,
) -> None:
    usage, model = extract_usage(response_content)
    if not usage:
        return

    request_id = response_headers.get("x-request-id")
    request.app.state.usage_store.record_usage(
        key_hash=key_hash,
        method=request.method,
        path=path,
        model=model,
        status_code=response_status,
        input_tokens=usage.get("input_tokens") or usage.get("prompt_tokens"),
        output_tokens=usage.get("output_tokens") or usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        request_id=request_id,
    )


def extract_usage(content: bytes) -> tuple[dict[str, int] | None, str | None]:
    try:
        payload = json.loads(content or b"{}")
    except json.JSONDecodeError:
        return None, None

    if not isinstance(payload, dict):
        return None, None

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, payload.get("model") if isinstance(payload.get("model"), str) else None

    numeric_usage = {key: value for key, value in usage.items() if isinstance(value, int)}
    model = payload.get("model") if isinstance(payload.get("model"), str) else None
    return numeric_usage, model


def parse_json_body(body: bytes) -> dict | None:
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def wants_streaming_response(request: Request, body: bytes) -> bool:
    if "text/event-stream" in request.headers.get("accept", ""):
        return True
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        return False

    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        return False
    return payload.get("stream") is True


async def stream_upstream(
    client: httpx.AsyncClient,
    request: Request,
    upstream: GatewayKey | RoutingConfig | Route,
    url: str,
    body: bytes,
    rate_limit_key: str,
) -> AsyncIterator[bytes]:
    async with client.stream(
        request.method,
        url,
        params=request.query_params.multi_items(),
        content=body,
        headers=upstream_headers(request, upstream),
    ) as upstream_response:
        remember_upstream_rate_limit(request, rate_limit_key, upstream_response.status_code, upstream_response.headers)
        if upstream_response.status_code >= 400:
            yield await upstream_response.aread()
            return
        async for chunk in upstream_response.aiter_bytes():
            yield chunk
