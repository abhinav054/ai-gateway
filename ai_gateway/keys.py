from __future__ import annotations

import argparse
import hashlib
import secrets
from pathlib import Path

from dotenv import load_dotenv

from .config import Gateway
from .storage import DEFAULT_SQLITE_PATH, KeyStore, SQLiteKeyStore, build_key_store


KEY_PREFIX = "gw-"
PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com",
    "groq": "https://api.groq.com/openai",
    "openrouter": "https://openrouter.ai/api",
    "together": "https://api.together.xyz",
    "custom": None,
}


def create_gateway_key() -> str:
    return f"{KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def provider_base_url(provider: str | None, base_url: str | None = None) -> str | None:
    if base_url:
        return base_url.rstrip("/")
    if not provider:
        return None
    normalized = provider.lower()
    if normalized not in PROVIDER_BASE_URLS:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDER_BASE_URLS[normalized]


def validate_provider_config(provider: str | None, api_key: str | None, base_url: str | None) -> None:
    if not provider:
        return
    if not api_key:
        raise ValueError("api_key is required when provider is set")
    if provider.lower() == "custom" and not base_url:
        raise ValueError("base_url is required when provider=custom")


def create_and_store_key(
    name: str | None = None,
    store: KeyStore | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    gateway_id: str | None = None,
) -> str:
    key = create_gateway_key()
    key_store = store or build_key_store()
    key_store.initialize()
    if gateway_id is not None:
        if key_store.get_gateway(gateway_id) is None:
            raise ValueError(f"Unknown gateway: {gateway_id}")
        if any(value is not None for value in (provider, api_key, base_url)):
            raise ValueError("Gateway-scoped keys use the gateway's routes; omit upstream credentials")
    normalized_provider = provider.lower() if provider else None
    validate_provider_config(normalized_provider, api_key, base_url)
    resolved_base_url = provider_base_url(normalized_provider, base_url)
    key_store.add_key_hash(
        hash_key(key),
        name,
        provider=normalized_provider,
        api_key=api_key,
        base_url=resolved_base_url,
        gateway_id=gateway_id,
    )
    return key


def upsert_routing_config(
    *,
    name: str,
    provider: str | None,
    api_key: str,
    base_url: str | None = None,
    store: KeyStore | None = None,
) -> None:
    if not name:
        raise ValueError("name is required")
    if not api_key:
        raise ValueError("api_key is required")
    normalized_provider = provider.lower() if provider else "custom"
    validate_provider_config(normalized_provider, api_key, base_url)
    resolved_base_url = provider_base_url(normalized_provider, base_url)
    if not resolved_base_url:
        raise ValueError("base_url is required")

    key_store = store or build_key_store()
    key_store.initialize()
    key_store.upsert_route(
        name=name,
        provider=normalized_provider,
        api_key=api_key,
        base_url=resolved_base_url,
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Manage AI gateway keys.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gateway_parser = subparsers.add_parser("gateway", help="Create or replace a gateway sandbox from JSON.")
    gateway_parser.add_argument("--file", required=True, type=Path, help="Gateway definition JSON file.")
    gateway_parser.add_argument("--sqlite-path", default=None, help="Store the gateway in this SQLite database.")

    create_parser = subparsers.add_parser("create", help="Create a new gateway API key.")
    create_parser.add_argument("--name", help="Optional label stored with the key hash.")
    create_parser.add_argument("--gateway-id", help="Restrict this client key to a gateway sandbox.")
    create_parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_BASE_URLS),
        help="LLM provider this gateway key should route to.",
    )
    create_parser.add_argument("--api-key", help="Provider API key to use for requests authenticated by this gateway key.")
    create_parser.add_argument("--base-url", help="Provider base URL. Overrides the selected provider default.")
    create_parser.add_argument(
        "--sqlite-path",
        default=None,
        help="Create the key in a SQLite database at this path.",
    )
    route_parser = subparsers.add_parser("route", help="Create or update a named upstream route.")
    route_parser.add_argument("--name", required=True, help="Route name selected by X-LLM-Route.")
    route_parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_BASE_URLS),
        default="custom",
        help="LLM provider for this route.",
    )
    route_parser.add_argument("--api-key", required=True, help="Provider API key for this route.")
    route_parser.add_argument("--base-url", help="Provider base URL. Overrides the selected provider default.")
    route_parser.add_argument(
        "--sqlite-path",
        default=None,
        help="Store the route in a SQLite database at this path.",
    )

    args = parser.parse_args()
    if args.command == "gateway":
        store = SQLiteKeyStore(Path(args.sqlite_path)) if args.sqlite_path else build_key_store()
        store.initialize()
        sandbox = Gateway.model_validate_json(args.file.read_text(encoding="utf-8"))
        store.upsert_gateway(sandbox)
        print(sandbox.id)
    elif args.command == "create":
        store = SQLiteKeyStore(Path(args.sqlite_path)) if args.sqlite_path else None
        key = create_and_store_key(
            args.name,
            store,
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            gateway_id=args.gateway_id,
        )
        print(key)
    elif args.command == "route":
        store = SQLiteKeyStore(Path(args.sqlite_path)) if args.sqlite_path else None
        upsert_routing_config(
            name=args.name,
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
            store=store,
        )
        print(args.name)


if __name__ == "__main__":
    main()
