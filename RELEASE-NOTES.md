# Release Notes

These notes summarize the bot runtime release history reconstructed from the
current repository state. Add a new section at the top for runtime,
deployment-facing, or user-visible bot behavior changes. Test-only and
docs-only changes do not need entries unless they affect operator workflow.

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
