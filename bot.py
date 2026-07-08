"""Kriegspiel bot driven by an OpenAI-compatible model.

The bot keeps the game API as the source of truth:
- the server provides the bot's private board state and legal actions
- the model recommends one action in strict JSON
- the bot validates that action locally before sending it back to the API

If the model output is malformed, stale, or unavailable, the bot falls back to a
deterministic legal action so the service remains playable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE_PATH = BASE_DIR / ".bot-state.json"
DEFAULT_ENV_PATH = BASE_DIR / ".env"
STATE_PATH = DEFAULT_STATE_PATH
ENV_PATH = DEFAULT_ENV_PATH
RULESET_SUMMARY_DIR = BASE_DIR / "ruleset_summaries"


DEFAULT_TIMEOUT_SECONDS = 20
ACTION_SCHEMA_NAME = "kriegspiel_next_action"
DEFAULT_MAX_ACTIVE_GAMES_BEFORE_CREATE = 1
DEFAULT_BOT_CREATE_COOLDOWN_SECONDS = 3600
BOT_JOIN_COOLDOWN_SECONDS = 600
BOT_GAME_PICK_PROBABILITY = 0.01
DEFAULT_MODEL_BATCH_SIZE = 10
DEFAULT_MAX_MODEL_BATCHES_PER_TURN = 5
DEFAULT_MAX_CONCURRENT_MODEL_CALLS = 5
DEFAULT_RESIGN_AFTER_MOVE_NUMBER = 256
DEFAULT_LLM_MAX_PROMPT_TURNS = 10
DEFAULT_LLM_MAX_OUTPUT_TOKENS = 512
DEFAULT_LLM_PREFLIGHT_SUCCESS_TTL_SECONDS = 60.0
DEFAULT_LLM_PREFLIGHT_FAILURE_TTL_SECONDS = 15.0
DEFAULT_LLM_PROVIDER = "openrouter"
DEFAULT_LLM_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_LLM_JSON_MODE = "json_schema"
DEFAULT_MODEL_AVAILABILITY_PROVIDER = "openai"
DEFAULT_MODEL_AVAILABILITY_REPORT_INTERVAL_SECONDS = 30.0
USD_PER_MILLION_TOKENS = 1_000_000
LLM_BOT_CREATE_COOLDOWN_SECONDS_BY_TIER = {
    "t2": 3600,
    "tier2": 3600,
    "2": 3600,
    "t3": 10800,
    "tier3": 10800,
    "3": 10800,
    "t4": 21600,
    "tier4": 21600,
    "4": 21600,
}
SUPPORTED_RULE_VARIANTS = ("berkeley", "berkeley_any", "cincinnati", "wild16", "rand", "english", "crazykrieg")
DEFAULT_SUPPORTED_RULE_VARIANTS = ",".join(SUPPORTED_RULE_VARIANTS)
LEGACY_DEFAULT_SUPPORTED_RULE_VARIANTS = ("berkeley", "berkeley_any")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
_LLM_PREFLIGHT_CACHE = {"ready": None, "expires_at": 0.0, "reason": "unchecked"}
_MODEL_AVAILABILITY_REPORT_CACHE = {"ready": None, "reason": "", "reported_at": 0.0}
_STATE_LOCK = threading.RLock()
_MODEL_CALL_SEMAPHORE_LOCK = threading.Lock()
_MODEL_CALL_SEMAPHORE: threading.BoundedSemaphore | None = None
_MODEL_CALL_SEMAPHORE_LIMIT = 0


def configure_runtime_paths(*, env_path: str | Path | None = None, state_path: str | Path | None = None) -> None:
    global ENV_PATH, STATE_PATH
    ENV_PATH = Path(env_path).expanduser().resolve() if env_path else DEFAULT_ENV_PATH
    STATE_PATH = Path(state_path).expanduser().resolve() if state_path else DEFAULT_STATE_PATH


def load_env_file(path: str | Path | None = None) -> None:
    env_path = Path(path) if path is not None else ENV_PATH
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def base_url() -> str:
    return os.environ.get("KRIEGSPIEL_API_BASE", "http://localhost:8000").rstrip("/")


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return default


def llm_provider() -> str:
    return os.environ.get("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower() or DEFAULT_LLM_PROVIDER


def llm_api_key() -> str:
    return os.environ.get("LLM_API_KEY", "").strip()


def llm_model() -> str:
    return os.environ.get("LLM_MODEL", "").strip()


def llm_base_url() -> str:
    return os.environ.get("LLM_API_BASE", DEFAULT_LLM_API_BASE).rstrip("/")


def llm_timeout_seconds() -> float:
    return max(5.0, env_float("LLM_TIMEOUT_SECONDS", 45.0))


def llm_max_output_tokens() -> int:
    raw = os.environ.get("LLM_MAX_OUTPUT_TOKENS", str(DEFAULT_LLM_MAX_OUTPUT_TOKENS)).strip()
    try:
        return max(32, int(raw))
    except ValueError:
        return DEFAULT_LLM_MAX_OUTPUT_TOKENS


def llm_json_mode() -> str:
    raw = os.environ.get("LLM_JSON_MODE", DEFAULT_LLM_JSON_MODE).strip().lower()
    return raw if raw in {"json_schema", "json_object", "none"} else DEFAULT_LLM_JSON_MODE


def max_concurrent_model_calls() -> int:
    raw = os.environ.get("LLM_BOT_MAX_CONCURRENT_MODEL_CALLS", str(DEFAULT_MAX_CONCURRENT_MODEL_CALLS)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_CONCURRENT_MODEL_CALLS


def configure_model_call_semaphore(limit: int | None = None) -> threading.BoundedSemaphore:
    global _MODEL_CALL_SEMAPHORE, _MODEL_CALL_SEMAPHORE_LIMIT
    configured_limit = max(1, int(limit if limit is not None else max_concurrent_model_calls()))
    with _MODEL_CALL_SEMAPHORE_LOCK:
        if _MODEL_CALL_SEMAPHORE is None or _MODEL_CALL_SEMAPHORE_LIMIT != configured_limit:
            _MODEL_CALL_SEMAPHORE = threading.BoundedSemaphore(configured_limit)
            _MODEL_CALL_SEMAPHORE_LIMIT = configured_limit
    return _MODEL_CALL_SEMAPHORE


def model_call_semaphore() -> threading.BoundedSemaphore:
    with _MODEL_CALL_SEMAPHORE_LOCK:
        if _MODEL_CALL_SEMAPHORE is not None:
            return _MODEL_CALL_SEMAPHORE
    return configure_model_call_semaphore()


def llm_input_usd_per_million_tokens() -> float:
    return max(0.0, env_float("LLM_INPUT_USD_PER_MILLION_TOKENS", 0.0))


def llm_cached_input_usd_per_million_tokens() -> float:
    return max(
        0.0,
        env_float("LLM_CACHED_INPUT_USD_PER_MILLION_TOKENS", llm_input_usd_per_million_tokens()),
    )


def llm_output_usd_per_million_tokens() -> float:
    return max(0.0, env_float("LLM_OUTPUT_USD_PER_MILLION_TOKENS", 0.0))


def model_availability_provider() -> str:
    return (
        os.environ.get("KRIEGSPIEL_MODEL_AVAILABILITY_PROVIDER", DEFAULT_MODEL_AVAILABILITY_PROVIDER).strip().lower()
        or DEFAULT_MODEL_AVAILABILITY_PROVIDER
    )


def llm_preflight_success_ttl_seconds() -> float:
    raw = os.environ.get("LLM_PREFLIGHT_SUCCESS_TTL_SECONDS", str(DEFAULT_LLM_PREFLIGHT_SUCCESS_TTL_SECONDS)).strip()
    try:
        return max(5.0, float(raw))
    except ValueError:
        return DEFAULT_LLM_PREFLIGHT_SUCCESS_TTL_SECONDS


def llm_preflight_failure_ttl_seconds() -> float:
    raw = os.environ.get("LLM_PREFLIGHT_FAILURE_TTL_SECONDS", str(DEFAULT_LLM_PREFLIGHT_FAILURE_TTL_SECONDS)).strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_LLM_PREFLIGHT_FAILURE_TTL_SECONDS


def model_availability_report_interval_seconds() -> float:
    raw = os.environ.get(
        "MODEL_AVAILABILITY_REPORT_INTERVAL_SECONDS",
        str(DEFAULT_MODEL_AVAILABILITY_REPORT_INTERVAL_SECONDS),
    ).strip()
    try:
        return max(5.0, float(raw))
    except ValueError:
        return DEFAULT_MODEL_AVAILABILITY_REPORT_INTERVAL_SECONDS


def usage_token_count(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def llm_input_tokens(usage: dict[str, Any]) -> int:
    return usage_token_count(usage, "input_tokens") or usage_token_count(usage, "prompt_tokens")


def llm_output_tokens(usage: dict[str, Any]) -> int:
    return usage_token_count(usage, "output_tokens") or usage_token_count(usage, "completion_tokens")


def llm_cached_input_tokens(usage: dict[str, Any]) -> int:
    cached = usage_token_count(usage, "cached_input_tokens") or usage_token_count(usage, "cache_read_input_tokens")
    for key in ("input_tokens_details", "prompt_tokens_details"):
        details = usage.get(key) if isinstance(usage.get(key), dict) else {}
        cached = max(cached, usage_token_count(details, "cached_tokens"))
    return cached


def llm_usage_cost_usd(usage: dict[str, Any]) -> float:
    input_tokens = llm_input_tokens(usage)
    cached_tokens = min(llm_cached_input_tokens(usage), input_tokens)
    uncached_input_tokens = max(0, input_tokens - cached_tokens)
    output_tokens = llm_output_tokens(usage)
    return (
        uncached_input_tokens * llm_input_usd_per_million_tokens()
        + cached_tokens * llm_cached_input_usd_per_million_tokens()
        + output_tokens * llm_output_usd_per_million_tokens()
    ) / USD_PER_MILLION_TOKENS


def log_llm_usage(*, game_id: str, model: str, payload: dict[str, Any]) -> None:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    input_tokens = llm_input_tokens(usage)
    cached_tokens = min(llm_cached_input_tokens(usage), input_tokens)
    output_tokens = llm_output_tokens(usage)
    response_id = str(payload.get("id") or "")
    provider = llm_provider()
    cost_usd = llm_usage_cost_usd(usage)
    logger.info(
        (
            "%s: model usage provider=%s model=%s response_id=%s "
            "input_tokens=%s cached_input_tokens=%s output_tokens=%s cost_usd=%.6f"
        ),
        game_id,
        provider,
        model,
        response_id,
        input_tokens,
        cached_tokens,
        output_tokens,
        cost_usd,
    )
    report_model_usage(
        {
            "game_id": game_id,
            "provider": provider,
            "model": model,
            "response_id": response_id or None,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost_usd,
        }
    )


def auth_headers() -> dict[str, str]:
    token = os.environ.get("KRIEGSPIEL_BOT_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def bot_username() -> str:
    return os.environ.get("KRIEGSPIEL_BOT_USERNAME", "").strip().lower()


def _load_state_unlocked() -> dict[str, Any]:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def _save_state_unlocked(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def load_state() -> dict[str, Any]:
    with _STATE_LOCK:
        return _load_state_unlocked()


def save_state(state: dict[str, Any]) -> None:
    with _STATE_LOCK:
        _save_state_unlocked(state)


def save_token(token: str) -> None:
    with _STATE_LOCK:
        state = _load_state_unlocked()
        state["token"] = token
        _save_state_unlocked(state)


def get_conversation_state(game_id: str) -> dict[str, Any]:
    with _STATE_LOCK:
        state = _load_state_unlocked()
        conversations = state.get("conversations")
        if not isinstance(conversations, dict):
            return {}
        conversation = conversations.get(game_id)
        return conversation if isinstance(conversation, dict) else {}


def save_conversation_state(game_id: str, conversation: dict[str, Any]) -> None:
    with _STATE_LOCK:
        state = _load_state_unlocked()
        conversations = state.get("conversations")
        if not isinstance(conversations, dict):
            conversations = {}
        conversations[game_id] = conversation
        state["conversations"] = conversations
        _save_state_unlocked(state)


def clear_conversation_state(game_id: str) -> None:
    with _STATE_LOCK:
        state = _load_state_unlocked()
        conversations = state.get("conversations")
        if not isinstance(conversations, dict) or game_id not in conversations:
            return
        conversations.pop(game_id, None)
        state["conversations"] = conversations
        _save_state_unlocked(state)


def maybe_restore_token() -> None:
    if os.environ.get("KRIEGSPIEL_BOT_TOKEN"):
        return
    if STATE_PATH.exists():
        token = load_state().get("token")
        if token:
            os.environ["KRIEGSPIEL_BOT_TOKEN"] = token


def register_bot() -> None:
    response = requests.post(
        f"{base_url()}/auth/bots/register",
        json={
            "username": os.environ.get("KRIEGSPIEL_BOT_USERNAME", "openrouterbot"),
            "display_name": os.environ.get("KRIEGSPIEL_BOT_DISPLAY_NAME", "OpenRouter Bot"),
            "owner_email": os.environ.get("KRIEGSPIEL_BOT_OWNER_EMAIL", "bot-openai-compatible@kriegspiel.org"),
            "description": os.environ.get(
                "KRIEGSPIEL_BOT_DESCRIPTION",
                "Model-driven Kriegspiel bot that uses an OpenAI-compatible chat-completions provider.",
            ),
            "listed": True,
            "supported_rule_variants": supported_rule_variants(),
        },
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    save_token(payload["api_token"])
    logger.debug("%s", json.dumps(payload, indent=2))


def sync_bot_profile() -> bool:
    try:
        post_json("/bots/profile", {"supported_rule_variants": supported_rule_variants()})
    except requests.RequestException as exc:
        logger.warning("failed to sync bot profile: %s", exc)
        return False
    return True


def get_json(path: str) -> dict[str, Any]:
    response = requests.get(f"{base_url()}{path}", headers=auth_headers(), timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def get_public_user(username: str) -> dict[str, Any]:
    response = requests.get(f"{base_url()}/user/{username}", headers=auth_headers(), timeout=DEFAULT_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.post(
        f"{base_url()}{path}",
        headers=auth_headers(),
        json=payload or {},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def report_model_usage(payload: dict[str, Any]) -> bool:
    if not os.environ.get("KRIEGSPIEL_BOT_TOKEN", "").strip():
        return False
    try:
        post_json("/bots/usage", payload)
    except requests.RequestException as exc:
        logger.warning("failed to report model usage: %s", exc)
        return False
    return True


def report_model_availability(ready: bool, reason: str, *, force: bool = False) -> bool:
    now = time.time()
    cached_ready = _MODEL_AVAILABILITY_REPORT_CACHE["ready"]
    cached_reason = str(_MODEL_AVAILABILITY_REPORT_CACHE["reason"])
    reported_at = float(_MODEL_AVAILABILITY_REPORT_CACHE["reported_at"])
    if (
        not force
        and cached_ready is bool(ready)
        and cached_reason == reason
        and now - reported_at < model_availability_report_interval_seconds()
    ):
        return False

    try:
        post_json(
            "/bots/availability",
            {"provider": model_availability_provider(), "ready": bool(ready), "reason": reason},
        )
    except requests.RequestException as exc:
        logger.warning("failed to report model availability: %s", exc)
        return False

    _MODEL_AVAILABILITY_REPORT_CACHE.update({"ready": bool(ready), "reason": reason, "reported_at": now})
    return True


def report_current_model_availability() -> tuple[bool, str]:
    ready, reason = llm_preflight_status()
    report_model_availability(ready, reason)
    return ready, reason


def auto_create_enabled() -> bool:
    raw = os.environ.get("KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME", "false").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def lobby_create_cooldown_seconds() -> int:
    raw = os.environ.get("KRIEGSPIEL_AUTO_CREATE_COOLDOWN_SECONDS", "").strip()
    if raw:
        try:
            return max(0, int(float(raw)))
        except ValueError:
            return DEFAULT_BOT_CREATE_COOLDOWN_SECONDS

    tier = os.environ.get("KRIEGSPIEL_LLM_BOT_TIER", os.environ.get("KRIEGSPIEL_PUBLIC_BOT_TIER", "t2")).strip().lower()
    return LLM_BOT_CREATE_COOLDOWN_SECONDS_BY_TIER.get(tier, DEFAULT_BOT_CREATE_COOLDOWN_SECONDS)


def can_create_lobby_game(now: float | None = None) -> bool:
    current = time.time() if now is None else now
    with _STATE_LOCK:
        last_created = load_state().get("last_lobby_game_created_at", 0)
    try:
        last_created = float(last_created)
    except (TypeError, ValueError):
        last_created = 0
    return current - last_created >= lobby_create_cooldown_seconds()


def record_lobby_game_created(now: float | None = None) -> None:
    with _STATE_LOCK:
        state = load_state()
        state["last_lobby_game_created_at"] = time.time() if now is None else now
        save_state(state)


def max_active_games_before_create() -> int:
    raw = os.environ.get("KRIEGSPIEL_MAX_ACTIVE_GAMES_BEFORE_CREATE", str(DEFAULT_MAX_ACTIVE_GAMES_BEFORE_CREATE)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MAX_ACTIVE_GAMES_BEFORE_CREATE


def create_payload() -> dict[str, str]:
    return {
        "rule_variant": os.environ.get("KRIEGSPIEL_AUTO_CREATE_RULE_VARIANT", "berkeley_any").strip() or "berkeley_any",
        "play_as": os.environ.get("KRIEGSPIEL_AUTO_CREATE_PLAY_AS", "random").strip() or "random",
        "time_control": "rapid",
        "opponent_type": "human",
    }


def supported_rule_variants() -> list[str]:
    raw = os.environ.get("KRIEGSPIEL_SUPPORTED_RULE_VARIANTS", DEFAULT_SUPPORTED_RULE_VARIANTS)
    variants: list[str] = []
    for item in raw.split(","):
        value = item.strip()
        if value in SUPPORTED_RULE_VARIANTS and value not in variants:
            variants.append(value)
    if tuple(variants) == LEGACY_DEFAULT_SUPPORTED_RULE_VARIANTS:
        return list(SUPPORTED_RULE_VARIANTS)
    return variants or list(SUPPORTED_RULE_VARIANTS)


def active_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [game for game in games if game.get("state") == "active"]


def waiting_games(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [game for game in games if game.get("state") == "waiting"]


def open_bot_lobby_candidates(open_games: list[dict[str, Any]], *, profile_lookup=None) -> list[dict[str, Any]]:
    profile_lookup = profile_lookup or get_public_user
    own_username = bot_username()
    candidates = []
    for game in open_games:
        creator_username = str(game.get("created_by") or "").strip()
        if not creator_username:
            continue
        if str(game.get("rule_variant") or "").strip() not in supported_rule_variants():
            continue
        creator_username_lower = creator_username.lower()
        if creator_username_lower == own_username:
            continue

        try:
            profile = profile_lookup(creator_username)
        except requests.RequestException:
            continue

        is_bot = bool(profile.get("is_bot")) or str(profile.get("role") or "").strip().lower() == "bot"
        if not is_bot:
            continue
        candidates.append(game)
    return candidates


def has_own_waiting_game(open_games: list[dict[str, Any]]) -> bool:
    own_username = bot_username()
    for game in open_games:
        created_by = str(game.get("created_by") or "").strip().lower()
        if created_by and created_by == own_username:
            return True
    return False


def can_attempt_bot_join(now: float | None = None) -> bool:
    current = time.time() if now is None else now
    with _STATE_LOCK:
        last_attempt = load_state().get("last_bot_game_join_attempt_at", 0)
    try:
        last_attempt = float(last_attempt)
    except (TypeError, ValueError):
        last_attempt = 0
    return current - last_attempt >= BOT_JOIN_COOLDOWN_SECONDS


def record_bot_join_attempt(now: float | None = None) -> None:
    with _STATE_LOCK:
        state = load_state()
        state["last_bot_game_join_attempt_at"] = time.time() if now is None else now
        save_state(state)


def should_join_bot_lobby_game(games: list[dict[str, Any]]) -> bool:
    return len(active_games(games)) < max_active_games_before_create()


def choose_bot_game_to_join(open_games: list[dict[str, Any]], *, rng: random.Random = random) -> dict[str, Any] | None:
    candidates = open_bot_lobby_candidates(open_games)
    if not candidates:
        return None
    return rng.choice(candidates)


def maybe_join_bot_lobby_game(games: list[dict[str, Any]], *, rng: random.Random = random) -> bool:
    if not should_join_bot_lobby_game(games):
        return False
    if not can_attempt_bot_join():
        return False

    record_bot_join_attempt()
    open_games = get_json("/game/open").get("games", [])
    candidate = choose_bot_game_to_join(open_games, rng=rng)
    if not candidate:
        return False
    if rng.random() >= BOT_GAME_PICK_PROBABILITY:
        return False

    ready, reason = llm_preflight_status()
    if not ready:
        logger.warning("skipping bot-game join because model provider is unavailable (%s)", reason)
        return False

    game_code = candidate.get("game_code")
    if not isinstance(game_code, str) or not game_code.strip():
        return False

    joined = post_json(f"/game/join/{game_code.strip()}")
    logger.debug("joined bot lobby game %s (%s)", joined["game_id"], joined["game_code"])
    return True


def should_create_lobby_game(games: list[dict[str, Any]]) -> bool:
    if not auto_create_enabled():
        return False
    if not can_create_lobby_game():
        return False
    if waiting_games(games):
        return False
    return len(active_games(games)) < max_active_games_before_create()


def maybe_create_lobby_game(games: list[dict[str, Any]]) -> bool:
    if not should_create_lobby_game(games):
        return False

    open_games = get_json("/game/open").get("games", [])
    if has_own_waiting_game(open_games):
        return False

    created = post_json("/game/create", create_payload())
    record_lobby_game_created()
    logger.debug("created lobby game %s (%s)", created["game_id"], created["game_code"])
    return True


def recent_scoresheet_turns(scoresheet: dict[str, Any], *, max_turns: int) -> list[dict[str, Any]]:
    turns = scoresheet.get("turns") if isinstance(scoresheet.get("turns"), list) else []
    recent_turns = turns[-max_turns:]
    payload: list[dict[str, Any]] = []
    for turn in recent_turns:
        item: dict[str, Any] = {"turn": turn.get("turn")}
        for color in ("white", "black"):
            entries = turn.get(color) if isinstance(turn.get(color), list) else []
            item[color] = [
                normalized
                for normalized in (normalize_scoresheet_entry(entry) for entry in entries)
                if normalized
            ]
        payload.append(item)
    return payload


def summarize_scoresheet_turns(scoresheet: dict[str, Any], *, max_turns: int) -> list[str]:
    lines: list[str] = []
    for turn in recent_scoresheet_turns(scoresheet, max_turns=max_turns):
        turn_number = turn.get("turn")
        for color in ("white", "black"):
            entries = turn.get(color) if isinstance(turn.get(color), list) else []
            for entry in entries:
                lines.append(f"Turn {turn_number} {color}: {entry}")
    return lines


def normalize_scoresheet_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if not isinstance(entry, dict):
        return ""

    prompt = str(entry.get("prompt") or "").strip()
    messages = entry.get("messages") if isinstance(entry.get("messages"), list) else []
    cleaned_messages = [str(item).strip() for item in messages if str(item).strip()]
    if not cleaned_messages:
        message = str(entry.get("message") or "").strip()
        if message:
            cleaned_messages = [message]

    text = " | ".join(dict.fromkeys(cleaned_messages))
    move_uci = str(entry.get("move_uci") or "").strip()

    if move_uci:
        text = f"[{move_uci}] {text}" if text else f"[{move_uci}]"
    if prompt and text:
        return f"{prompt}: {text}"
    return text or prompt


def model_batch_size() -> int:
    raw = os.environ.get("LLM_MODEL_BATCH_SIZE", str(DEFAULT_MODEL_BATCH_SIZE)).strip()
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return DEFAULT_MODEL_BATCH_SIZE


def max_model_batches_per_turn() -> int:
    raw = os.environ.get("LLM_MAX_BATCHES_PER_TURN", str(DEFAULT_MAX_MODEL_BATCHES_PER_TURN)).strip()
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return DEFAULT_MAX_MODEL_BATCHES_PER_TURN


def llm_max_prompt_turns() -> int:
    raw = os.environ.get("LLM_MAX_PROMPT_TURNS", str(DEFAULT_LLM_MAX_PROMPT_TURNS)).strip()
    try:
        return max(DEFAULT_LLM_MAX_PROMPT_TURNS, int(raw))
    except ValueError:
        return DEFAULT_LLM_MAX_PROMPT_TURNS


def resign_after_move_number() -> int:
    raw = os.environ.get("KRIEGSPIEL_RESIGN_AFTER_MOVE_NUMBER", str(DEFAULT_RESIGN_AFTER_MOVE_NUMBER)).strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_RESIGN_AFTER_MOVE_NUMBER


def move_limit_for_state(state: dict[str, Any]) -> int:
    if "llm_bot_ply_limit" not in state:
        return resign_after_move_number()

    raw = state.get("llm_bot_ply_limit")
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return resign_after_move_number()


def completed_ply_count_for_state(state: dict[str, Any]) -> int:
    raw = state.get("ply_count") if "ply_count" in state else state.get("move_number")
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def should_resign_for_move_limit(state: dict[str, Any]) -> bool:
    limit = move_limit_for_state(state)
    if limit <= 0:
        return False
    return completed_ply_count_for_state(state) >= limit


def extract_recent_referee_items(scoresheet: dict[str, Any], *, limit: int = 8) -> list[str]:
    turns = scoresheet.get("turns") if isinstance(scoresheet.get("turns"), list) else []
    lines: list[str] = []
    for turn in turns[-limit:]:
        turn_number = turn.get("turn")
        for color in ("white", "black"):
            entries = turn.get(color) if isinstance(turn.get(color), list) else []
            for entry in entries:
                normalized = normalize_scoresheet_entry(entry)
                if normalized:
                    lines.append(f"Turn {turn_number} {color}: {normalized}")
    return lines[-limit:]


def scoresheet_digest(scoresheet: dict[str, Any]) -> str:
    payload = json.dumps(scoresheet, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def turn_signature(state: dict[str, Any]) -> str:
    return "|".join(
        [
            str(state.get("state") or ""),
            str(state.get("move_number") or ""),
            str(state.get("turn") or ""),
            str(state.get("your_color") or ""),
        ]
    )


def new_recent_items(previous: list[str], current: list[str]) -> list[str]:
    if not previous:
        return current
    best_overlap = 0
    max_overlap = min(len(previous), len(current))
    for size in range(max_overlap, 0, -1):
        if previous[-size:] == current[:size]:
            best_overlap = size
            break
    return current[best_overlap:]


@lru_cache(maxsize=len(SUPPORTED_RULE_VARIANTS) + 1)
def load_ruleset_summary(rule_variant: str) -> str:
    variant = rule_variant if rule_variant in SUPPORTED_RULE_VARIANTS else "berkeley_any"
    return (RULESET_SUMMARY_DIR / f"{variant}.md").read_text(encoding="utf-8").strip()


def build_system_prompt(rule_variant: str) -> str:
    rules_summary = load_ruleset_summary(rule_variant)
    return (
        "You are a strong Kriegspiel player.\n"
        "Use only the provided private information and legal actions.\n"
        "Do not invent moves. Do not suggest illegal actions.\n"
        "Return only a JSON object shaped as {\"m\":[\"e2e4\",\"d2d4\",\"ask_any\"]}.\n"
        "Return minified JSON with no spaces, newlines, or prose.\n"
        "Turn JSON keys: c=your color,t=side to move,mn=move number,fen=private board FEN,"
        "mat=material,hist=recent scorecard,act=possible actions,moves=legal UCI moves,"
        "n=target candidate count,res=reserves,rej=rejected actions,fb=retry feedback.\n"
        "Colors inside mat/hist/res use w=white and b=black; hist items use n=turn,w=white entries,b=black entries.\n"
        "Return exactly n unique m entries ordered strictly from best to worst priority.\n"
        "Entry 1 must be your best choice, entry 2 your next-best choice, and so on.\n"
        "Prioritize strategically strong, tactically sound moves that are robust under uncertainty.\n"
        "Each move entry must exactly match one moves item.\n"
        "Use the string ask_any only when ask_any appears in act.\n"
        "Do not explain the rules. Do not include prose outside the JSON object.\n\n"
        f"{rules_summary}\n"
    )


def _compact_color_key(color: str) -> str:
    return "w" if color == "white" else "b"


def _prompt_material_summary(state: dict[str, Any]) -> dict[str, dict[str, int]]:
    material = state.get("material_summary") if isinstance(state.get("material_summary"), dict) else {}
    payload: dict[str, dict[str, int]] = {}
    for color in ("white", "black"):
        side = material.get(color) if isinstance(material.get(color), dict) else {}
        if not side:
            continue
        item: dict[str, int] = {}
        pieces_remaining = side.get("pieces_remaining")
        if isinstance(pieces_remaining, int):
            item["pieces"] = pieces_remaining
        pawns_captured = side.get("pawns_captured")
        if isinstance(pawns_captured, int):
            item["pawns_captured"] = pawns_captured
        if item:
            payload[_compact_color_key(color)] = item
    return payload


def _prompt_reserve_summary(state: dict[str, Any], *, rule_variant: str) -> dict[str, dict[str, int]]:
    if rule_variant != "crazykrieg":
        return {}
    reserves = state.get("reserve_summary") if isinstance(state.get("reserve_summary"), dict) else {}
    payload: dict[str, dict[str, int]] = {}
    for color in ("white", "black"):
        side = reserves.get(color) if isinstance(reserves.get(color), dict) else {}
        if side:
            payload[_compact_color_key(color)] = {
                piece: int(side.get(piece, 0) or 0)
                for piece in ("pawns", "knights", "bishops", "rooks", "queens")
            }
    return payload


def compact_recent_scoresheet_turns(scoresheet: dict[str, Any], *, max_turns: int) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for turn in recent_scoresheet_turns(scoresheet, max_turns=max_turns):
        item: dict[str, Any] = {"n": turn.get("turn")}
        for color in ("white", "black"):
            key = _compact_color_key(color)
            entries = turn.get(color) if isinstance(turn.get(color), list) else []
            item[key] = entries
        compact.append(item)
    return compact


def build_turn_snapshot_payload(
    state: dict[str, Any],
    *,
    feedback: list[str] | None = None,
    exclude_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rule_variant = str(state.get("rule_variant") or "berkeley_any")
    scoresheet = state.get("scoresheet") if isinstance(state.get("scoresheet"), dict) else {}
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []
    available_action_count = len(allowed_moves) + (1 if "ask_any" in possible_actions else 0)
    target_count = min(model_batch_size(), available_action_count)
    payload: dict[str, Any] = {
        "c": state.get("your_color"),
        "t": state.get("turn"),
        "mn": state.get("move_number"),
        "fen": state.get("your_fen"),
        "mat": _prompt_material_summary(state),
        "hist": compact_recent_scoresheet_turns(scoresheet, max_turns=llm_max_prompt_turns()),
        "act": possible_actions,
        "moves": allowed_moves,
        "n": target_count,
    }
    reserves = _prompt_reserve_summary(state, rule_variant=rule_variant)
    if reserves:
        payload["res"] = reserves
    rejected = [format_action(item) for item in (exclude_actions or [])[-20:]]
    if rejected:
        payload["rej"] = rejected
    retry_feedback = [item for item in (feedback or [])[-6:] if item]
    if retry_feedback:
        payload["fb"] = retry_feedback
    return payload


def build_turn_snapshot_user_prompt(
    state: dict[str, Any],
    *,
    feedback: list[str] | None = None,
    exclude_actions: list[dict[str, Any]] | None = None,
) -> str:
    payload = build_turn_snapshot_payload(
        state,
        feedback=feedback,
        exclude_actions=exclude_actions,
    )
    return f"Current turn JSON:\n{json.dumps(payload, separators=(',', ':'), ensure_ascii=True, sort_keys=True)}"


def build_initial_user_prompt(state: dict[str, Any]) -> str:
    return build_turn_snapshot_user_prompt(state)


def build_followup_user_prompt(
    state: dict[str, Any],
    *,
    feedback: list[str] | None = None,
    exclude_actions: list[dict[str, Any]] | None = None,
    recent_updates: list[str] | None = None,
) -> str:
    _ = recent_updates
    return build_turn_snapshot_user_prompt(
        state,
        feedback=feedback,
        exclude_actions=exclude_actions,
    )


def action_schema() -> dict[str, Any]:
    return {
        "name": ACTION_SCHEMA_NAME,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "m": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                }
            },
            "required": ["m"],
        },
        "strict": True,
    }


def llm_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.environ.get("LLM_HTTP_REFERER", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    title = os.environ.get("LLM_X_OPENROUTER_TITLE", os.environ.get("LLM_X_TITLE", "")).strip()
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def apply_response_format(payload: dict[str, Any]) -> None:
    mode = llm_json_mode()
    if mode == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": action_schema(),
        }
    elif mode == "json_object":
        payload["response_format"] = {"type": "json_object"}


def llm_enabled() -> bool:
    return bool(llm_api_key() and llm_model())


def cache_llm_preflight(ready: bool, *, reason: str, ttl_seconds: float) -> tuple[bool, str]:
    _LLM_PREFLIGHT_CACHE["ready"] = ready
    _LLM_PREFLIGHT_CACHE["reason"] = reason
    _LLM_PREFLIGHT_CACHE["expires_at"] = time.time() + max(0.0, ttl_seconds)
    return ready, reason


def describe_http_error(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc) or exc.__class__.__name__

    pieces = [f"http_{response.status_code}"]
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            for key in ("type", "code", "message"):
                value = error.get(key)
                if value:
                    pieces.append(str(value))
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            pieces.append(detail)
    elif response.text:
        pieces.append(response.text.strip())

    return ": ".join(dict.fromkeys(piece for piece in pieces if piece))


def llm_preflight_status(force: bool = False) -> tuple[bool, str]:
    if not llm_api_key():
        return cache_llm_preflight(False, reason="missing_llm_api_key", ttl_seconds=llm_preflight_failure_ttl_seconds())
    if not llm_model():
        return cache_llm_preflight(False, reason="missing_llm_model", ttl_seconds=llm_preflight_failure_ttl_seconds())

    now = time.time()
    if not force and _LLM_PREFLIGHT_CACHE["ready"] is not None and now < float(_LLM_PREFLIGHT_CACHE["expires_at"]):
        return bool(_LLM_PREFLIGHT_CACHE["ready"]), str(_LLM_PREFLIGHT_CACHE["reason"])

    try:
        with model_call_semaphore():
            response = requests.post(
                f"{llm_base_url()}/chat/completions",
                headers=llm_headers(llm_api_key()),
                json={
                    "model": llm_model(),
                    "messages": [
                        {"role": "system", "content": "Reply with OK."},
                        {"role": "user", "content": "Ping"},
                    ],
                    "max_tokens": 16,
                },
                timeout=llm_timeout_seconds(),
            )
        response.raise_for_status()
    except requests.RequestException as exc:
        reason = describe_http_error(exc)
        logger.warning("llm preflight failed: %s", reason)
        return cache_llm_preflight(False, reason=reason, ttl_seconds=llm_preflight_failure_ttl_seconds())

    return cache_llm_preflight(True, reason="ok", ttl_seconds=llm_preflight_success_ttl_seconds())


def call_llm(
    *,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    api_key = llm_api_key()
    model = llm_model()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": llm_max_output_tokens(),
    }
    apply_response_format(payload)

    with model_call_semaphore():
        response = requests.post(
            f"{llm_base_url()}/chat/completions",
            headers=llm_headers(api_key),
            json=payload,
            timeout=llm_timeout_seconds(),
        )
    response.raise_for_status()
    return response.json()


def extract_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list):
        chunks: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            parsed = message.get("parsed")
            if isinstance(parsed, dict):
                return json.dumps(parsed, separators=(",", ":"), ensure_ascii=True)
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                chunks.append(content.strip())
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks)

    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
        if chunks:
            return "\n".join(chunks)

    for key in ("output_text", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("No text found in OpenAI-compatible response payload")


def parse_model_decision(payload: dict[str, Any]) -> dict[str, Any]:
    text = extract_response_text(payload)
    try:
        decision = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        decision = json.loads(match.group(0))

    if not isinstance(decision, dict):
        raise ValueError("Model response must decode to an object")
    return decision


def normalize_decision(decision: dict[str, Any], state: dict[str, Any]) -> dict[str, Any] | None:
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []

    action = str(decision.get("action") or "").strip().lower()
    uci = decision.get("uci")

    if action == "move":
        if "move" not in possible_actions or not isinstance(uci, str):
            return None
        normalized_uci = uci.strip()
        allowed_by_lower = {str(move).strip().lower(): str(move).strip() for move in allowed_moves}
        allowed_move = allowed_by_lower.get(normalized_uci.lower())
        if allowed_move is None:
            return None
        return {"action": "move", "uci": allowed_move}

    if action == "ask_any":
        if "ask_any" not in possible_actions:
            return None
        return {"action": "ask_any", "uci": None}

    return None


def normalize_ranked_decisions(payload: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    compact_actions = payload.get("m") if isinstance(payload.get("m"), list) else None
    if compact_actions is not None:
        for item in compact_actions:
            if not isinstance(item, str):
                continue
            text = item.strip()
            candidate = {"action": "ask_any", "uci": None} if text == "ask_any" else {"action": "move", "uci": text}
            decision = normalize_decision(candidate, state)
            if decision is None:
                continue
            key = (decision["action"], decision["uci"])
            if key in seen:
                continue
            seen.add(key)
            normalized.append(decision)
        return normalized

    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        decision = normalize_decision(candidate, state)
        if decision is None:
            continue
        key = (decision["action"], decision["uci"])
        if key in seen:
            continue
        seen.add(key)
        normalized.append(decision)
    return normalized


def fallback_decision(state: dict[str, Any]) -> dict[str, Any] | None:
    allowed_moves = state.get("allowed_moves") if isinstance(state.get("allowed_moves"), list) else []
    possible_actions = state.get("possible_actions") if isinstance(state.get("possible_actions"), list) else []

    if allowed_moves:
        ordered = sorted(allowed_moves)
        center_bias = ["d2d4", "e2e4", "d7d5", "e7e5"]
        for preferred in center_bias:
            if preferred in ordered:
                return {"action": "move", "uci": preferred}
        return {"action": "move", "uci": ordered[0]}
    if "ask_any" in possible_actions:
        return {"action": "ask_any", "uci": None}
    return None


def fallback_ranked_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    decision = fallback_decision(state)
    return [decision] if decision is not None else []


def choose_ranked_actions(
    state: dict[str, Any],
    *,
    game_id: str,
    conversation: dict[str, Any] | None = None,
    feedback: list[str] | None = None,
    exclude_actions: list[dict[str, Any]] | None = None,
    recent_updates: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    if not llm_api_key():
        return fallback_ranked_actions(state), "fallback_no_llm_key", None
    if not llm_model():
        return fallback_ranked_actions(state), "fallback_no_llm_model", None

    try:
        system_prompt = build_system_prompt(state.get("rule_variant", "berkeley_any"))
        _ = conversation, recent_updates
        user_prompt = build_turn_snapshot_user_prompt(
            state,
            feedback=feedback,
            exclude_actions=exclude_actions,
        )

        raw_response = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        log_llm_usage(
            game_id=game_id,
            model=llm_model(),
            payload=raw_response,
        )
        decisions = normalize_ranked_decisions(parse_model_decision(raw_response), state)
        if decisions:
            return decisions, "model", None
    except requests.RequestException as exc:
        reason = describe_http_error(exc)
        logger.warning("model selection failed: %s", reason)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("model selection failed: %s", exc)

    return fallback_ranked_actions(state), "fallback", None


def format_action(decision: dict[str, Any]) -> str:
    if decision["action"] == "ask_any":
        return "ask_any"
    return str(decision["uci"])


def maybe_play_game(game_id: str) -> bool:
    metadata = get_json(f"/game/{game_id}")
    feedback: list[str] = []
    tried_actions: list[dict[str, Any]] = []
    acted = False
    conversation = get_conversation_state(game_id)

    for _batch in range(max_model_batches_per_turn()):
        state = get_json(f"/game/{game_id}/state")
        if state.get("state") != "active" or state.get("turn") != state.get("your_color"):
            return acted
        if should_resign_for_move_limit(state):
            result = post_json(f"/game/{game_id}/resign")
            clear_conversation_state(game_id)
            logger.info(
                "%s: resigned at move %s after reaching move limit %s -> %s",
                game_id,
                completed_ply_count_for_state(state),
                move_limit_for_state(state),
                result.get("result"),
            )
            return True
        if isinstance(metadata.get("rule_variant"), str):
            state["rule_variant"] = metadata["rule_variant"]

        scoresheet = state.get("scoresheet") if isinstance(state.get("scoresheet"), dict) else {}
        recent_referee = extract_recent_referee_items(scoresheet, limit=10)
        current_turn_signature = turn_signature(state)
        previous_turn_signature = str(conversation.get("turn_signature") or "")
        if previous_turn_signature != current_turn_signature:
            tried_actions = []
            feedback = []

        previous_recent_referee = conversation.get("recent_referee_items")
        if not isinstance(previous_recent_referee, list):
            previous_recent_referee = []
        recent_updates = new_recent_items(previous_recent_referee, recent_referee)

        decisions, source, _response_id = choose_ranked_actions(
            state,
            game_id=game_id,
            conversation=conversation,
            feedback=feedback,
            exclude_actions=tried_actions,
            recent_updates=recent_updates,
        )
        conversation = {
            "turn_signature": current_turn_signature,
            "scoresheet_digest": scoresheet_digest(scoresheet),
            "recent_referee_items": recent_referee,
        }
        save_conversation_state(game_id, conversation)
        if not decisions:
            return acted

        batch_success = False
        for decision in decisions:
            tried_actions.append(decision)
            if decision["action"] == "move":
                result = post_json(f"/game/{game_id}/move", {"uci": decision["uci"]})
                acted = acted or bool(result.get("move_done"))
                logger.debug("%s: %s move %s -> %s", game_id, source, decision["uci"], result["announcement"])
                if result.get("move_done"):
                    state_after = get_json(f"/game/{game_id}/state")
                    if isinstance(metadata.get("rule_variant"), str):
                        state_after["rule_variant"] = metadata["rule_variant"]
                    scoresheet_after = state_after.get("scoresheet") if isinstance(state_after.get("scoresheet"), dict) else {}
                    recent_after = extract_recent_referee_items(scoresheet_after, limit=10)
                    recent_updates_after = new_recent_items(recent_referee, recent_after)
                    feedback = [f"Move complete: {result.get('announcement')}"]
                    if recent_updates_after:
                        feedback.append("New referee announcements: " + " || ".join(recent_updates_after))
                    conversation = {
                        "turn_signature": turn_signature(state_after),
                        "scoresheet_digest": scoresheet_digest(scoresheet_after),
                        "recent_referee_items": recent_after,
                    }
                    save_conversation_state(game_id, conversation)
                    tried_actions = []
                    batch_success = True
                    break
                feedback.append(f"Rejected move {decision['uci']}: {result.get('announcement')}")
                continue

            result = post_json(f"/game/{game_id}/ask-any")
            acted = acted or bool(result.get("move_done"))
            logger.debug("%s: %s ask-any -> %s", game_id, source, result["announcement"])
            state_after = get_json(f"/game/{game_id}/state")
            if isinstance(metadata.get("rule_variant"), str):
                state_after["rule_variant"] = metadata["rule_variant"]
            scoresheet_after = state_after.get("scoresheet") if isinstance(state_after.get("scoresheet"), dict) else {}
            recent_after = extract_recent_referee_items(scoresheet_after, limit=10)
            recent_updates_after = new_recent_items(recent_referee, recent_after)
            feedback = [f"Ask-any result: {result.get('announcement')}"]
            if recent_updates_after:
                feedback.append("New referee announcements: " + " || ".join(recent_updates_after))
            feedback.append(
                "New possible actions: "
                + json.dumps(
                    {
                        "act": state_after.get("possible_actions"),
                        "moves": state_after.get("allowed_moves"),
                    },
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            )
            conversation = {
                "turn_signature": turn_signature(state_after),
                "scoresheet_digest": scoresheet_digest(scoresheet_after),
                "recent_referee_items": recent_after,
            }
            save_conversation_state(game_id, conversation)
            tried_actions = []
            batch_success = True
            break

        if not batch_success:
            refreshed_state = get_json(f"/game/{game_id}/state")
            if isinstance(metadata.get("rule_variant"), str):
                refreshed_state["rule_variant"] = metadata["rule_variant"]
            refreshed_scoresheet = refreshed_state.get("scoresheet") if isinstance(refreshed_state.get("scoresheet"), dict) else {}
            refreshed_recent = extract_recent_referee_items(refreshed_scoresheet, limit=10)
            feedback.append("Top-ranked batch failed; provide the next best legal candidates.")
            feedback.append(
                "Current legal options now: "
                + json.dumps(
                    {
                        "act": refreshed_state.get("possible_actions"),
                        "moves": refreshed_state.get("allowed_moves"),
                    },
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
            )
            conversation = {
                "turn_signature": turn_signature(refreshed_state),
                "scoresheet_digest": scoresheet_digest(refreshed_scoresheet),
                "recent_referee_items": refreshed_recent,
            }
            save_conversation_state(game_id, conversation)
            continue

    return acted


def http_status_code(exc: requests.RequestException) -> int | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    status_code = getattr(response, "status_code", None)
    return int(status_code) if isinstance(status_code, int) else None


class GameRunner:
    def __init__(self, game_id: str, *, poll_seconds: float) -> None:
        self.game_id = game_id
        self.poll_seconds = max(0.5, float(poll_seconds))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"llm-bot-game-{game_id}", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        logger.info("%s: starting game runner", self.game_id)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._started:
            self.thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._started and self.thread.is_alive()

    def _wait(self) -> None:
        self.stop_event.wait(self.poll_seconds)

    def _run(self) -> None:
        stop_reason = "stopped"
        try:
            while not self.stop_event.is_set():
                try:
                    state = get_json(f"/game/{self.game_id}/state")
                except requests.RequestException as exc:
                    status_code = http_status_code(exc)
                    if status_code in {400, 403, 404, 409}:
                        stop_reason = f"state unavailable http_{status_code}"
                        break
                    logger.warning("%s: runner state poll failed: %s", self.game_id, exc)
                    self._wait()
                    continue

                state_value = state.get("state")
                if state_value != "active":
                    stop_reason = f"state={state_value}"
                    break

                if state.get("turn") == state.get("your_color"):
                    try:
                        maybe_play_game(self.game_id)
                    except requests.RequestException as exc:
                        status_code = http_status_code(exc)
                        if status_code in {400, 403, 404, 409}:
                            stop_reason = f"play stopped http_{status_code}"
                            break
                        logger.warning("%s: runner play failed: %s", self.game_id, exc)

                self._wait()
        finally:
            logger.info("%s: stopped game runner (%s)", self.game_id, stop_reason)


class GameRunnerScheduler:
    def __init__(
        self,
        *,
        poll_seconds: float,
        runner_factory: Any | None = None,
    ) -> None:
        self.poll_seconds = poll_seconds
        self.runner_factory = runner_factory or (lambda game_id: GameRunner(game_id, poll_seconds=poll_seconds))
        self.runners: dict[str, Any] = {}

    @staticmethod
    def game_id_for(game: dict[str, Any]) -> str:
        return str(game.get("game_id") or "").strip()

    def reconcile(self, games: list[dict[str, Any]]) -> None:
        active_ids: set[str] = set()
        for game in active_games(games):
            game_id = self.game_id_for(game)
            if not game_id:
                continue
            active_ids.add(game_id)
            runner = self.runners.get(game_id)
            if runner is not None and runner.is_alive():
                continue
            if runner is not None:
                runner.join(timeout=0)
            runner = self.runner_factory(game_id)
            self.runners[game_id] = runner
            runner.start()

        for game_id, runner in list(self.runners.items()):
            if game_id in active_ids:
                continue
            logger.info("%s: stopping game runner after active-game discovery removed it", game_id)
            runner.stop()
            runner.join(timeout=1.0)
            self.runners.pop(game_id, None)

        self.prune_finished()

    def prune_finished(self) -> None:
        for game_id, runner in list(self.runners.items()):
            if runner.is_alive():
                continue
            runner.join(timeout=0)
            self.runners.pop(game_id, None)

    def stop_all(self) -> None:
        for runner in list(self.runners.values()):
            runner.stop()
        for runner in list(self.runners.values()):
            runner.join(timeout=2.0)
        self.runners.clear()


def run_loop(poll_seconds: float) -> None:
    concurrency = max_concurrent_model_calls()
    configure_model_call_semaphore(concurrency)
    logger.info("model-call concurrency configured: max=%s", concurrency)
    scheduler = GameRunnerScheduler(poll_seconds=poll_seconds)
    try:
        while True:
            try:
                report_current_model_availability()
                mine = get_json("/game/mine/active")
                games = mine.get("games", [])
                maybe_create_lobby_game(games)
                maybe_join_bot_lobby_game(games)
                scheduler.reconcile(games)
            except requests.RequestException as exc:
                logger.warning("poll failed: %s", exc)
            time.sleep(poll_seconds)
    finally:
        scheduler.stop_all()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kriegspiel OpenAI-compatible bot.")
    parser.add_argument(
        "--env-file",
        default=os.environ.get("KRIEGSPIEL_BOT_ENV_FILE", str(DEFAULT_ENV_PATH)),
        help="Path to the bot instance env file.",
    )
    parser.add_argument(
        "--state-file",
        default=os.environ.get("KRIEGSPIEL_BOT_STATE_FILE", str(DEFAULT_STATE_PATH)),
        help="Path to the bot instance state file.",
    )
    parser.add_argument("--register", action="store_true", help="Register the bot and persist the returned token.")
    parser.add_argument("--poll-seconds", type=float, default=3.0, help="Seconds between /game/mine/active polls.")
    args = parser.parse_args()

    configure_runtime_paths(env_path=args.env_file, state_path=args.state_file)
    load_env_file()
    maybe_restore_token()

    if args.register:
        register_bot()
        return

    if not os.environ.get("KRIEGSPIEL_BOT_TOKEN"):
        raise SystemExit("KRIEGSPIEL_BOT_TOKEN is missing. Run with --register first.")
    if not llm_api_key():
        logger.warning("LLM_API_KEY is missing; bot-vs-bot joins will be skipped and turns will use fallback mode.")
    elif not llm_model():
        logger.warning("LLM_MODEL is missing; bot-vs-bot joins will be skipped and turns will use fallback mode.")

    sync_bot_profile()
    run_loop(args.poll_seconds)


if __name__ == "__main__":
    main()
