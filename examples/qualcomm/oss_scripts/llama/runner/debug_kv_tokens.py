#!/usr/bin/env python3

import argparse
import os
import sqlite3


def format_tokens(blob: bytes | None, limit: int) -> str:
    if not blob:
        return "[]"
    tokens = []
    for i in range(0, len(blob), 8):
        tokens.append(int.from_bytes(blob[i : i + 8], "little", signed=False))
    preview = tokens[:limit]
    suffix = "" if len(tokens) <= limit else " ..."
    return f"{preview}{suffix}"


def format_blob_hex(blob: bytes | None, limit: int) -> str:
    if not blob:
        return ""
    preview = blob[:limit]
    return preview.hex(" ")


def format_blob_u8(blob: bytes | None, limit: int) -> str:
    if not blob:
        return "[]"
    preview = list(blob[:limit])
    suffix = "" if len(blob) <= limit else " ..."
    return f"{preview}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "db_path",
        nargs="?",
        default="qwen3-1.7b-000.bin",
        help="Path to the SQLite KVStore file.",
    )
    parser.add_argument(
        "--preview-tokens",
        type=int,
        default=16,
        help="Number of prompt tokens to preview per row.",
    )
    parser.add_argument(
        "--preview-bytes",
        type=int,
        default=64,
        help="Number of kv_blob bytes to preview per row.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        raise FileNotFoundError(f"DB file not found: {args.db_path}")

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print(f"db: {args.db_path}")
    print("tables:")
    for row in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ):
        print(f"  - {row['name']}")

    print("\nkv_meta:")
    for row in cur.execute("SELECT * FROM kv_meta"):
        print(dict(row))

    print("\nkv_tokens:")
    query = """
        SELECT
            kv_id,
            prompt_len,
            prompt_tokens,
            next_token,
            ori_pos,
            prev_kv_id,
            succ_kv_id,
            root_kv_id,
            chunk_start
        FROM kv_tokens
        ORDER BY kv_id
    """
    for row in cur.execute(query):
        blob = row["prompt_tokens"]
        blob_bytes = len(blob) if blob is not None else 0
        token_count = blob_bytes // 8
        print(
            f"kv_id={row['kv_id']} "
            f"prompt_len={row['prompt_len']} "
            f"token_count={token_count} "
            f"ori_pos={row['ori_pos']} "
            f"prev={row['prev_kv_id']} "
            f"succ={row['succ_kv_id']} "
            f"root={row['root_kv_id']} "
            f"chunk_start={row['chunk_start']} "
            f"next_token={row['next_token']}"
        )
        print(
            f"  prompt_tokens_preview={format_tokens(blob, args.preview_tokens)}"
        )

    print("\nkv_caches:")
    query = """
        SELECT
            kv_id,
            prompt,
            kv_blob,
            length(kv_blob) AS kv_blob_bytes
        FROM kv_caches
        ORDER BY kv_id
    """
    for row in cur.execute(query):
        prompt_preview = row["prompt"]
        if prompt_preview is not None and len(prompt_preview) > 80:
            prompt_preview = prompt_preview[:80] + " ..."
        blob = row["kv_blob"]
        print(
            f"kv_id={row['kv_id']} "
            f"kv_blob_bytes={row['kv_blob_bytes']} "
            f"prompt={prompt_preview!r}"
        )
        print(
            f"  kv_blob_hex_preview={format_blob_hex(blob, args.preview_bytes)}"
        )
        print(
            f"  kv_blob_u8_preview={format_blob_u8(blob, args.preview_bytes)}"
        )

    conn.close()


if __name__ == "__main__":
    main()
