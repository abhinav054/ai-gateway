from __future__ import annotations

import argparse
import hashlib
import secrets
from pathlib import Path

from dotenv import load_dotenv

from .storage import DEFAULT_SQLITE_PATH, KeyStore, SQLiteKeyStore, build_key_store


KEY_PREFIX = "gw-"


def create_gateway_key() -> str:
    return f"{KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def create_and_store_key(name: str | None = None, store: KeyStore | None = None) -> str:
    key = create_gateway_key()
    key_store = store or build_key_store()
    key_store.initialize()
    key_store.add_key_hash(hash_key(key), name)
    return key


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Manage AI gateway keys.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create a new gateway API key.")
    create_parser.add_argument("--name", help="Optional label stored with the key hash.")
    create_parser.add_argument(
        "--sqlite-path",
        default=None,
        help="Create the key in a SQLite database at this path.",
    )

    args = parser.parse_args()
    if args.command == "create":
        store = SQLiteKeyStore(Path(args.sqlite_path)) if args.sqlite_path else None
        key = create_and_store_key(args.name, store)
        print(key)


if __name__ == "__main__":
    main()
