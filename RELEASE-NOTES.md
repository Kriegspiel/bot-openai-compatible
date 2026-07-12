# Release Notes

These notes summarize the bot runtime release history reconstructed from the
current repository state. Add a new section at the top for runtime,
deployment-facing, or user-visible bot behavior changes. Test-only and
docs-only changes do not need entries unless they affect operator workflow.

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
