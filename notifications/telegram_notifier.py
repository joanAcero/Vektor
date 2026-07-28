"""
notifications/telegram_notifier.py
-----------------------------------
Minimal Telegram Bot API client: send_message (HTML) and send_photo.

Uses Telegram's "HTML" parse mode rather than "Markdown": Markdown treats a
single unescaped '_', '*', '`' or '[' anywhere in the text as the start of a
formatting entity, and stock data is full of exactly that (column names like
Base_Weeks, Res_Touches; stage values like pre_breakout) -- an odd count of
any one of those characters produces a 400 "can't parse entities" error that
silently kills the whole notification. HTML mode only requires escaping
'&', '<', '>' (via Python's html.escape), which almost never appear in
tickers/sectors/numbers, so callers only need to escape the *values* they
interpolate -- see daily_report.py's _esc() helper.

send_photo's caption is sent WITHOUT a parse_mode (plain text) deliberately:
captions are built from raw ticker/column data and don't need any markup, so
skipping parse_mode sidesteps the whole escaping question for them.

Credentials come from environment variables, never hardcoded and never
committed to git:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID

Set them in a local `.env` file (see .env.example) -- daily_report.py loads it
via python-dotenv -- or export them directly in whatever runs the job later
(systemd unit's Environment=, etc.). Same code, either way.
"""

from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/{method}"
_MAX_MESSAGE_CHARS = 4000  # Telegram's real limit is 4096; leave margin for the truncation note
_MAX_CAPTION_CHARS = 1000  # Telegram's real limit is 1024


def _creds() -> tuple[str, str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set as environment "
            "variables (see .env.example)."
        )
    return token, chat_id


def send_message(text: str, parse_mode: str = "HTML") -> None:
    token, chat_id = _creds()
    if len(text) > _MAX_MESSAGE_CHARS:
        text = text[:_MAX_MESSAGE_CHARS] + "\n... (truncated)"
    resp = requests.post(
        _API.format(token=token, method="sendMessage"),
        data={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
        timeout=15,
    )
    if not resp.ok:
        log.error("Telegram sendMessage failed: %s %s", resp.status_code, resp.text)
        resp.raise_for_status()


def send_photo(path: str, caption: str = "") -> None:
    token, chat_id = _creds()
    if len(caption) > _MAX_CAPTION_CHARS:
        caption = caption[:_MAX_CAPTION_CHARS] + "\n... (truncated)"
    with open(path, "rb") as fh:
        resp = requests.post(
            _API.format(token=token, method="sendPhoto"),
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": fh},
            timeout=30,
        )
    if not resp.ok:
        log.error("Telegram sendPhoto failed for %s: %s %s", path, resp.status_code, resp.text)
        resp.raise_for_status()
