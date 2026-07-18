# Release Notes

These notes summarize the bot runtime release history reconstructed from the
current repository state. Add a new section at the top for runtime,
deployment-facing, or user-visible bot behavior changes. Test-only and
docs-only changes do not need entries unless they affect operator workflow.

## Non-Billable OpenRouter Availability

- **Idle Cost Fix**: replace OpenRouter's periodic model-completion preflight
  with the authenticated, non-generating `GET /api/v1/key` status endpoint.
- **Cadence**: cache successful OpenRouter status checks for five minutes and
  failures for one minute while continuing to publish fresh availability to the
  Kriegspiel backend.
- **Direct OpenAI**: use the non-generating model metadata endpoint for direct
  OpenAI instances; preserve the tiny completion fallback only for unknown
  OpenAI-compatible providers.

## Tiered Bot Join Budgets

- **Lobby Policy**: derive bot-vs-bot join probability from
  `KRIEGSPIEL_LLM_BOT_TIER` by default: T2 `0.0010`, T3 `0.0005`, T4
  `0.0002`, and T5 `0.0001`.
- **Operator Override**: allow `KRIEGSPIEL_BOT_GAME_PICK_PROBABILITY` or
  `BOT_GAME_PICK_PROBABILITY` to override the tier default per instance.

## OpenAI Responses API Wire Mode

- **Responses API**: add opt-in `LLM_WIRE_API=responses` support for direct
  OpenAI models that are not served by Chat Completions.
- **Structured Actions**: map the existing compact action schema to Responses
  `text.format` structured outputs and parse Responses `function_call` output
  items when tool mode is enabled.
- **Provider Preflight**: use `/responses` for model availability preflight when
  the instance selects Responses wire mode.

## Qwen 3.7 Schema Mode

- **Instance Template**: set Qwen 3.7 Plus to `LLM_USE_TOOLS=false` so it uses
  the JSON-schema text path instead of OpenRouter tool calls that return HTTP
  400 provider errors.

## Direct OpenAI Tool Compatibility

- **Reasoning Effort**: add optional `LLM_REASONING_EFFORT`, applied to
  Chat Completions calls and provider preflight, so direct OpenAI GPT
  tool-call instances can set `reasoning_effort=none`.
- **Instance Templates**: set `LLM_REASONING_EFFORT=none` for direct OpenAI
  GPT-5.6/GPT-5.5-class templates that use OpenAI function tools.

## Direct Provider Catalog

- **Direct OpenAI GPT Models**: add `LLM_MAX_TOKENS_PARAMETER` so direct
  OpenAI GPT-5.6/GPT-5.5-class Chat Completions instances can use
  `max_completion_tokens` while existing OpenAI-compatible provider instances
  continue to use `max_tokens`.
- **Provider Preflight**: use the same token parameter for startup/provider
  availability preflight requests so direct OpenAI instances do not fall back to
  `max_tokens`.
- **Instance Templates**: add GPT-5.6 Sol/Terra/Luna, Grok 4.5, Gemini 3.5
  Flash, Qwen 3.7 Plus, DeepSeek V3.2, MiniMax M3, and Mistral Medium 3.5
  templates with their catalogue tiers.

## Qwen Plus T2 Template

- **Instance Template**: move the Qwen Plus example environment from T3 to T2
  so newly rendered OpenAI-compatible instances match the production catalogue.

## Tool Argument JSON Recovery

- **Tool Calls**: recover valid JSON objects from OpenAI-compatible tool-call
  argument strings that include explanatory text before or after the object,
  matching the existing recovery behavior for normal message text.

## LLM Turn Limit Counter Fix

- **Bot-vs-bot Caps**: compare `llm_bot_turn_limit` against completed legal
  turns derived from `move_number`, not raw transcript attempts in `ply_count`.
- **Runtime Logs**: label automatic cap resignations as turn, ply, or legacy
  move limits so the log message matches the counter that triggered it.

## OpenAI-Compatible Tool Calls

- **Structured Actions**: add optional `LLM_USE_TOOLS=true` support for
  OpenAI-compatible function calls using the existing compact action schema.
- **Reasoning Responses**: parse Chat Completions `tool_calls` before text and
  recognize reasoning text aliases returned by OpenRouter reasoning models.
- **Kimi Template**: opt the Kimi K2 Thinking instance template into tool calls
  so mandatory-reasoning responses still produce structured actions.

## T4 Model Instance Templates

- **T4 Catalogue**: add OpenAI-compatible T4 instance templates for DeepSeek V4
  Pro, Gemini 3.1 Pro Preview, GLM 5.2, Kimi K2.7 Code, and Hermes 4 405B.
- **Bot-vs-bot Caps**: honor the backend's current `llm_bot_turn_limit` field
  before falling back to the legacy `llm_bot_ply_limit` field.

## Current Runtime Baseline

- **Bot Identity**: shared OpenAI-compatible model-bot scaffold; the base
  defaults to `openrouterbot`, while production uses independent per-model bot
  identities.
- **Rulesets**: supports `berkeley`, `berkeley_any`, `cincinnati`, `wild16`,
  `rand`, `english`, and `crazykrieg`, with legacy two-ruleset configs expanded
  to the full supported set.
- **Runtime Shape**: runs one process per model instance, with one lightweight
  runner thread per active game and a configurable shared model call cap that
  defaults to 5 concurrent calls.
- **Lobby Policy**: does not create human lobby games by default, can join a
  compatible bot-created waiting game with 1% probability on a ten-minute scan,
  and checks provider availability before joining new bot-vs-bot games.
- **Move Policy**: uses an OpenAI-compatible Chat Completions provider to rank
  strict JSON candidate actions from compact private-state prompts, then
  validates those actions against server-provided legal actions.
