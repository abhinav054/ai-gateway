from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Route(ConfigModel):
    name: str = Field(min_length=1)
    provider: Literal["openai", "groq", "openrouter", "together", "custom"] = "custom"
    api_key: str = Field(min_length=1, repr=False)
    base_url: str

    @field_validator("name", "api_key")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("base_url")
    @classmethod
    def http_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an HTTP or HTTPS URL")
        return value.rstrip("/")


class Guardrail(ConfigModel):
    name: str = Field(min_length=1)
    type: Literal["token_usage_limit"] = "token_usage_limit"
    enabled: StrictBool = True
    config: dict

    @field_validator("config")
    @classmethod
    def token_limit(cls, value: dict) -> dict:
        limit = value.get("max_total_tokens")
        if type(limit) is not int or limit < 0:
            raise ValueError("max_total_tokens must be a non-negative integer")
        if "include_request_estimate" in value and type(value["include_request_estimate"]) is not bool:
            raise ValueError("include_request_estimate must be a boolean")
        return value


class RateLimiting(ConfigModel):
    enabled: StrictBool = True
    default_retry_after_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)


class GatewayConfig(ConfigModel):
    load_balancing_enabled: StrictBool = True
    session_aware: StrictBool = True
    default_route: str | None = None
    guardrails_enabled: StrictBool = True
    rate_limiting: RateLimiting = Field(default_factory=RateLimiting)


class Gateway(ConfigModel):
    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str | None = None
    routes: list[Route] = Field(default_factory=list)
    guardrails: list[Guardrail] = Field(default_factory=list)
    config: GatewayConfig = Field(default_factory=GatewayConfig)

    @model_validator(mode="after")
    def validate_names(self) -> Gateway:
        names = [route.name for route in self.routes]
        if len(names) != len(set(names)):
            raise ValueError("route names must be unique within a gateway")
        if any("," in name or name != name.strip() for name in names):
            raise ValueError("route names cannot contain commas or surrounding whitespace")
        guardrails = [guardrail.name for guardrail in self.guardrails]
        if len(guardrails) != len(set(guardrails)):
            raise ValueError("guardrail names must be unique within a gateway")
        if self.config.default_route is not None and self.config.default_route not in names:
            raise ValueError("default_route must name a route in this gateway")
        return self

    def public_dict(self) -> dict:
        return self.model_dump(exclude={"routes": {"__all__": {"api_key"}}})
