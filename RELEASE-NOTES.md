# Release Notes

These notes summarize the bot runtime release history reconstructed from the
current repository state. Add a new section at the top for runtime,
deployment-facing, or user-visible bot behavior changes. Test-only and
docs-only changes do not need entries unless they affect operator workflow.

## LLM Turn Limit Counter Fix

- **Bot-vs-bot Caps**: compare `llm_bot_turn_limit` against completed legal
  turns derived from `move_number`, not raw transcript attempts in `ply_count`.
- **Runtime Logs**: label automatic cap resignations as turn, ply, or legacy
  move limits so the log message matches the counter that triggered it.

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
