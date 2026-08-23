from __future__ import annotations

import os
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

from .keys import create_and_store_key, hash_key
from .storage import build_key_store, build_usage_store


load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "600"))

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.http = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    app.state.key_store = build_key_store()
    app.state.key_store.initialize()
    app.state.usage_store = build_usage_store()
    app.state.usage_store.initialize()
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="AI Gateway", lifespan=lifespan)


def require_gateway_key(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Missing gateway bearer token")

    key_hash = hash_key(token)
    env_hashes = {item.strip() for item in os.getenv("GATEWAY_KEY_HASHES", "").split(",") if item.strip()}
    if key_hash not in env_hashes and not request.app.state.key_store.has_key_hash(key_hash):
        raise HTTPException(status_code=401, detail="Invalid gateway bearer token")
    return key_hash


def upstream_headers(request: Request) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() in FORWARDED_REQUEST_HEADERS
    }
    headers["authorization"] = f"Bearer {required_env('OPENAI_API_KEY')}"

    if os.getenv("OPENAI_ORGANIZATION"):
        headers["openai-organization"] = os.environ["OPENAI_ORGANIZATION"]
    if os.getenv("OPENAI_PROJECT"):
        headers["openai-project"] = os.environ["OPENAI_PROJECT"]

    return headers


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
) -> JSONResponse:
    admin_token = os.getenv("GATEWAY_ADMIN_TOKEN")
    if not admin_token:
        raise HTTPException(status_code=404, detail="Admin key creation is disabled")
    if x_admin_token != admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    key = create_and_store_key(name=name, store=app.state.key_store)
    return JSONResponse({"key": key})


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_openai(path: str, request: Request) -> Response:
    key_hash = require_gateway_key(request)

    url = f"{OPENAI_BASE_URL.rstrip('/')}/v1/{path}"
    body = await request.body()
    client: httpx.AsyncClient = request.app.state.http

    if wants_streaming_response(request, body):
        return StreamingResponse(
            stream_upstream(client, request, url, body),
            media_type="text/event-stream",
        )

    upstream = await client.request(
        request.method,
        url,
        params=request.query_params,
        content=body,
        headers=upstream_headers(request),
    )
    track_usage(
        request=request,
        path=f"/v1/{path}",
        key_hash=key_hash,
        response_content=upstream.content,
        response_status=upstream.status_code,
        response_headers=upstream.headers,
    )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
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
    url: str,
    body: bytes,
) -> AsyncIterator[bytes]:
    async with client.stream(
        request.method,
        url,
        params=request.query_params,
        content=body,
        headers=upstream_headers(request),
    ) as upstream:
        if upstream.status_code >= 400:
            yield await upstream.aread()
            return
        async for chunk in upstream.aiter_bytes():
            yield chunk
