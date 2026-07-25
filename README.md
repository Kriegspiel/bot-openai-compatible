# bot-openai-compatible

Kriegspiel bot that asks an OpenAI-compatible Chat Completions model to choose the next action from the bot's private game state.

This repo is intended as the shared scaffold for OpenRouter and direct provider experiments. The game logic and prompt format are aligned with the existing model bots; the provider, model name, endpoint, JSON mode, and pricing are environment configuration.

## What it does

- registers as a listed Kriegspiel bot
- syncs supported rulesets with the API on startup
- polls assigned games from the live API
- discovers assigned games every ten seconds with jitter while keeping
  active-game state polls on their faster independent cadence
- does not create waiting lobby games by default
- can join another bot's waiting lobby game using its configured tier
  probability while still under its active-game cap
- keeps one bot process per model instance while running one lightweight runner thread per active assigned game
- gates external model calls through a shared configurable concurrency limit
- builds a stateless compact prompt from ruleset summary, private FEN, public state, recent scorecard turns, legal actions, and retry feedback
- asks the configured model for ranked candidate actions in compact JSON
- validates model output against server-provided legal actions before playing
- honors explicit server-reported caps before asking the model; current
  bot-vs-bot LLM game caps are completed-turn limits and do not count illegal
  attempts
- checks provider availability with a cached preflight before joining new bot-vs-bot games; OpenRouter and direct OpenAI use non-generating metadata endpoints so idle checks are not billed
- falls back safely if the model is missing, unavailable, or returns malformed output

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python bot.py --register
python bot.py
```

The bot uses dedicated prompt summaries in `ruleset_summaries/*.md`, derived from the canonical `ks-content/rules` docs.

By default the bot identity is:

- username: `openrouterbot`
- display name: `OpenRouter Bot`
- owner email: `bot-openai-compatible@kriegspiel.org`

## Multiple Model Instances

Use separate env and state files when running one independent bot per model:

```bash
python bot.py \
  --env-file instances/gemini-flash-lite.env \
  --state-file instances/gemini-flash-lite-state.json \
  --register

python bot.py \
  --env-file instances/gemini-flash-lite.env \
  --state-file instances/gemini-flash-lite-state.json
```

Each instance env must have its own Kriegspiel bot identity:

```env
KRIEGSPIEL_BOT_USERNAME=openrouter_gemini_flash_lite
KRIEGSPIEL_BOT_DISPLAY_NAME=OpenRouter Gemini Flash-Lite
KRIEGSPIEL_BOT_OWNER_EMAIL=bot-openai-compatible@kriegspiel.org
KRIEGSPIEL_BOT_DESCRIPTION=OpenRouter Gemini Flash-Lite Kriegspiel model bot.
LLM_MODEL=google/gemini-2.5-flash-lite
```

Checked-in T2/T3/T4 templates live under `instances/` with `.env.example` suffixes.
Copy one to `.env`, fill in secrets, and use the matching state file when
running or registering that instance.

The shared provider key can be copied from the base production `.env`, but each instance should keep its own `KRIEGSPIEL_BOT_TOKEN` or state file.

## Provider Config

Required:

- `LLM_API_KEY`
- `LLM_MODEL`

OpenRouter default:

```env
LLM_PROVIDER=openrouter
LLM_API_BASE=https://openrouter.ai/api/v1
LLM_MODEL=<openrouter-model-slug>
LLM_API_KEY=<openrouter-api-key>
```

Direct OpenAI default:

```env
LLM_PROVIDER=openai
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=<openai-model-id>
LLM_API_KEY=<openai-api-key>
LLM_MAX_TOKENS_PARAMETER=max_completion_tokens
LLM_REASONING_EFFORT=none
```

Direct OpenAI models that are not served through Chat Completions can opt into
the Responses API instead. GPT-5.5 Pro currently uses `medium` because the
direct model rejected reasoning effort `none` during production validation:

```env
LLM_PROVIDER=openai
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=<openai-model-id>
LLM_API_KEY=<openai-api-key>
LLM_WIRE_API=responses
LLM_REASONING_EFFORT=medium
```

Current production direct OpenAI reasoning defaults:

| Instance | Public model | Wire API | Default reasoning level |
|---|---|---|---|
| `gpt55` | GPT-5.5 | Chat Completions | `none` |
| `gpt55-pro` | GPT-5.5 Pro | Responses API | `medium` |
| `gpt56-luna` | GPT-5.6 Luna | Chat Completions | `none` |
| `gpt56-terra` | GPT-5.6 Terra | Chat Completions | `none` |
| `gpt56-sol` | GPT-5.6 Sol | Chat Completions | `none` |

Direct provider examples:

| Provider | `LLM_API_BASE` | Example model |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-5.6-luna` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.1-8b-instant` |
| Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.5-flash-lite` |
| DeepSeek | `https://api.deepseek.com` | `deepseek-v4-flash` |
| Qwen / Alibaba Model Studio | workspace or region `.../compatible-mode/v1` endpoint | `qwen-plus` |

JSON mode:

- `LLM_JSON_MODE=json_schema` by default
- use `LLM_JSON_MODE=json_object` if a provider supports JSON object mode but not strict schemas
- use `LLM_JSON_MODE=none` if a provider rejects `response_format`; the bot still parses and validates output
- `LLM_USE_TOOLS=true` forces an OpenAI-compatible function call with the same
  compact action schema instead of `response_format`; use it for reasoning
  models that spend their whole completion budget before emitting message text
- `LLM_WIRE_API=responses` uses the OpenAI Responses API with
  `text.format` structured outputs; the default remains
  `LLM_WIRE_API=chat_completions`

Pricing is logged from env so experiments can compare providers without code changes:

```env
LLM_INPUT_USD_PER_MILLION_TOKENS=0
LLM_CACHED_INPUT_USD_PER_MILLION_TOKENS=0
LLM_OUTPUT_USD_PER_MILLION_TOKENS=0
```

Direct OpenAI calls are capped at `$18` per UTC calendar month across all bot
processes on the host. The default shared ledger is
`~/.local/state/kriegspiel/provider-budgets/openai.json`; configure
`OPENAI_MONTHLY_BUDGET_USD` or `OPENAI_MONTHLY_BUDGET_STATE_PATH` only when a
different provider-wide policy is intentional. Pricing must be configured for
direct OpenAI instances so requests can be reserved accurately. The ledger
starts tracking prospectively when first deployed and resets automatically at
the UTC month boundary.

OpenRouter new-game preflight requires at least `$2` of key-level balance.
Override this with `OPENROUTER_MIN_REMAINING_USD` only when changing the shared
operating policy.

Optional OpenRouter attribution headers:

```env
LLM_HTTP_REFERER=https://kriegspiel.org
LLM_X_OPENROUTER_TITLE=Kriegspiel
```

`KRIEGSPIEL_MODEL_AVAILABILITY_PROVIDER` defaults to `openai` because the current backend availability endpoint only accepts existing provider labels. Keep that default until the backend supports arbitrary model providers.

## Gameplay Config

- `KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME=true|false`
- `KRIEGSPIEL_AUTO_CREATE_RULE_VARIANT=berkeley|berkeley_any|cincinnati|wild16|rand|english|crazykrieg`
- `KRIEGSPIEL_AUTO_CREATE_PLAY_AS=white|black|random`
- `KRIEGSPIEL_SUPPORTED_RULE_VARIANTS=berkeley,berkeley_any,cincinnati,wild16,rand,english,crazykrieg`
- `KRIEGSPIEL_ACTIVE_GAME_DISCOVERY_LIMIT=100`
- `KRIEGSPIEL_MAX_ACTIVE_GAMES_BEFORE_CREATE=1`
- `KRIEGSPIEL_LLM_BOT_TIER=T2|T3|T4|T5`
- `KRIEGSPIEL_BOT_GAME_PICK_PROBABILITY=0.001` optional explicit join
  probability override; if unset, tier defaults are T2 `0.0010`, T3 `0.0005`,
  T4 `0.0002`, and T5 `0.0001`
- `KRIEGSPIEL_AUTO_CREATE_COOLDOWN_SECONDS=3600|10800|21600`
- `LLM_BOT_MAX_CONCURRENT_MODEL_CALLS=5`
- `KRIEGSPIEL_RESIGN_AFTER_MOVE_NUMBER=256` fallback used only when the server
  omits an LLM bot limit field

Existing production env files with the old default `KRIEGSPIEL_SUPPORTED_RULE_VARIANTS=berkeley,berkeley_any` are treated as stale defaults and expanded to all supported rulesets.

Bot-vs-bot play is enabled by default:

- the bot samples open waiting games at most once every 10 minutes
- it only considers games created by another bot
- it joins using the configured/tiered budget probability on that scan
- it uses the same active-game cap for intentional bot-vs-bot joins
- it keeps the local cooldown even when no join candidate is found, matching backend bot-join limits and avoiding tight lobby scans

Within one model instance, the main process still polls/discovers assigned
active games. Each active game gets one runner thread that owns only that game
until it completes or disappears. Backend polling and move submission are not
globally throttled, but provider model calls are guarded by
`LLM_BOT_MAX_CONCURRENT_MODEL_CALLS`, which defaults to `5`. This prevents large
tournament batches from timing out behind one serial model loop while avoiding a
process-per-game deployment shape.

The runtime separates assignment discovery from active play:

- `--poll-seconds 2` controls each active game's state-poll cadence
- `--discovery-poll-seconds 10` controls the base
  `/game/mine/active` cadence
- `--discovery-poll-jitter-ratio 0.15` randomizes every discovery delay by
  plus or minus 15 percent so independently started instances do not settle
  into a synchronized polling burst

Active-game discovery requests `/game/mine/active` with
`KRIEGSPIEL_ACTIVE_GAME_DISCOVERY_LIMIT`, defaulting to `100`, so tournament
batches larger than the backend's human-facing default list size still get
runners for every assigned game. Already-running game runners are allowed to
keep polling their own game even if a later discovery response omits them; the
runner stops itself when the game completes or becomes unavailable.

Optional human-lobby creation is still disabled by default for individual model
instances. If an operator enables one selected model instance as the random
tier representative, the built-in create cooldown defaults to T2 hourly, T3
every 3 hours, and T4 every 6 hours; `KRIEGSPIEL_AUTO_CREATE_COOLDOWN_SECONDS`
overrides that cadence.

Prompt defaults:

- `LLM_MAX_PROMPT_TURNS=10` (values below 10 are clamped to 10)
- `LLM_MODEL_BATCH_SIZE=10`
- `LLM_MAX_BATCHES_PER_TURN=5`
- `LLM_BOT_MAX_CONCURRENT_MODEL_CALLS=5`
- `LLM_USE_TOOLS=false`
- `LLM_WIRE_API=chat_completions`
- `LLM_MAX_OUTPUT_TOKENS=512`
- `LLM_MAX_TOKENS_PARAMETER=max_tokens` (`max_completion_tokens` for direct
  OpenAI GPT-5.6/GPT-5.5-class chat completions)
- `LLM_REASONING_EFFORT=` (`none` for direct OpenAI GPT-5.6/GPT-5.5-class
  chat completions that opt into tool calls; `medium` for the direct OpenAI
  GPT-5.5 Pro Responses instance)
- `LLM_PREFLIGHT_SUCCESS_TTL_SECONDS=60`
- `LLM_PREFLIGHT_FAILURE_TTL_SECONDS=15`
- `OPENROUTER_PREFLIGHT_SUCCESS_TTL_SECONDS=300`
- `OPENROUTER_PREFLIGHT_FAILURE_TTL_SECONDS=60`
- `OPENROUTER_MIN_REMAINING_USD=2`
- `OPENAI_MONTHLY_BUDGET_USD=18`
- `PROVIDER_BUDGET_RESERVATION_TTL_SECONDS=1800`

OpenRouter preflight calls `GET /api/v1/key` to validate the configured key and
its key-level spending limit without creating a model generation, and rejects
new work below the configured remaining-balance floor. Direct OpenAI provider
preflight checks the shared monthly ledger before calling
`GET /v1/models/{model}`. The generic LLM preflight settings continue to
control providers that require a tiny model completion for readiness checks.

## Test

```bash
python3 -m unittest discover -s tests
```

## systemd

A production host can run the bot as a service with `deploy/kriegspiel-openai-compatible-bot.service`.
