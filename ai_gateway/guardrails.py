from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .storage import GuardrailStore, UsageStore


DEFAULT_GUARDRAILS_DIR = Path(".gateway")


@dataclass
class GuardrailDecision:
    allowed: bool
    status_code: int = 200
    code: str | None = None
    message: str | None = None
    guardrail: str | None = None
    details: dict[str, Any] | None = None


def load_json_guardrails(path: Path = DEFAULT_GUARDRAILS_DIR) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    guardrails: list[dict[str, Any]] = []
    for file_path in sorted(path.glob("*.json")):
        with file_path.open(encoding="utf-8") as file:
            payload = json.load(file)

        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{file_path} must contain a guardrail object or list of objects")
            guardrail = dict(record)
            guardrail.setdefault("source", str(file_path))
            guardrails.append(guardrail)

    return guardrails


def load_guardrails(store: GuardrailStore, folder: Path = DEFAULT_GUARDRAILS_DIR) -> list[dict[str, Any]]:
    return [*store.list_guardrails(), *load_json_guardrails(folder)]


def evaluate_guardrails(
    *,
    guardrails: list[dict[str, Any]],
    key_hash: str,
    usage_store: UsageStore,
    request_payload: dict[str, Any] | None,
) -> GuardrailDecision:
    for guardrail in guardrails:
        if guardrail.get("enabled", True) is False:
            continue

        guardrail_type = guardrail.get("type")
        if guardrail_type == "token_usage_limit":
            decision = evaluate_token_usage_limit(
                guardrail=guardrail,
                key_hash=key_hash,
                usage_store=usage_store,
                request_payload=request_payload,
            )
            if not decision.allowed:
                return decision

    return GuardrailDecision(allowed=True)


def evaluate_token_usage_limit(
    *,
    guardrail: dict[str, Any],
    key_hash: str,
    usage_store: UsageStore,
    request_payload: dict[str, Any] | None,
) -> GuardrailDecision:
    config = guardrail.get("config", {})
    if not isinstance(config, dict):
        return deny(guardrail, "invalid_guardrail_config", "Guardrail config must be an object", 500)

    max_total_tokens = config.get("max_total_tokens")
    if not isinstance(max_total_tokens, int) or max_total_tokens < 0:
        return deny(guardrail, "invalid_guardrail_config", "max_total_tokens must be a non-negative integer", 500)

    consumed_tokens = usage_store.total_tokens_for_key(key_hash)
    estimated_tokens = estimate_requested_tokens(request_payload) if config.get("include_request_estimate", False) else 0
    projected_tokens = consumed_tokens + estimated_tokens

    if projected_tokens >= max_total_tokens:
        return deny(
            guardrail,
            "token_usage_limit_exceeded",
            config.get("message", "Token usage limit exceeded"),
            429,
            {
                "max_total_tokens": max_total_tokens,
                "consumed_tokens": consumed_tokens,
                "estimated_request_tokens": estimated_tokens,
                "projected_tokens": projected_tokens,
            },
        )

    return GuardrailDecision(allowed=True)


def estimate_requested_tokens(request_payload: dict[str, Any] | None) -> int:
    if not request_payload:
        return 0

    total = 0
    for field in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        value = request_payload.get(field)
        if isinstance(value, int) and value > 0:
            total += value
            break

    return total


def deny(
    guardrail: dict[str, Any],
    code: str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> GuardrailDecision:
    return GuardrailDecision(
        allowed=False,
        status_code=status_code,
        code=code,
        message=message,
        guardrail=str(guardrail.get("name", guardrail.get("type", "guardrail"))),
        details=details,
    )
