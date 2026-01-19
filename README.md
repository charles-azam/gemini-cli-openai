# gemini-cli-zai

This is a fork of Gemini CLI that routes requests to Z.ai's GLM-4.7 endpoint and
preserves Z.ai's extended thinking ("reasoning_content") across tool calls.

## Why this fork

- Use GLM-4.7 with the Gemini CLI interface and tooling.
- Support Z.ai's interleaved thinking output and tool calling format.
- Route web search tool calls through Z.ai when GLM auth is active.

## How it is implemented

- Adds a GLM content generator that translates Gemini CLI requests into
  OpenAI-style chat completion payloads for Z.ai.
- Maps tools, tool choices, and function calls to the OpenAI-compatible schema.
- Enables Z.ai thinking mode when thinking is configured, and preserves
  `reasoning_content` by feeding it back into the next assistant turn.
- Parses Z.ai usage metadata (including reasoning tokens) into Gemini CLI usage
  stats.
- Remaps the web search tool to Z.ai's web search in chat when using GLM auth.

Key implementation files:

- `packages/core/src/core/glmContentGenerator.ts`
- `packages/core/src/core/contentGenerator.ts`

## Configuration

- Set `ZAI_API_KEY` to use GLM auth. The CLI will auto-select GLM when it is
  set.
- Optional endpoint override: `model.zai.endpoint` (takes precedence), or
  `ZAI_API_BASE_URL` / `GLM_API_BASE_URL`.
- Use model `glm-4.7` in settings or via `--model glm-4.7`.
- Optional thinking behavior: `model.zai.clearThinking` to clear preserved
  reasoning between turns (requires restart).
- Optional thinking toggle: `model.zai.disableThinking` to request direct
  answers without reasoning (requires restart).

CLI overrides (no settings file needed):

```bash
gemini --model glm-4.7 \
  --zai-endpoint https://api.z.ai/api/coding/paas/v4/chat/completions \
  --zai-clear-thinking \
  --zai-disable-thinking
```

`--zai-model` is an alias for `--model`.

Note: Z.ai thinking mode is requested by default in this fork. Use
`--zai-disable-thinking` or `model.zai.disableThinking` to turn it off. Gemini
`thinkingConfig` settings are not mapped.

Example `settings.json`:

```json
{
  "model": {
    "zai": {
      "endpoint": "https://api.z.ai/api/coding/paas/v4/chat/completions",
      "clearThinking": false,
      "disableThinking": false
    }
  }
}
```

Default endpoint:

- `https://api.z.ai/api/coding/paas/v4/chat/completions`

## Web search

When GLM auth is active, the `google_web_search` tool is routed through Z.ai's
web search in chat and returns Z.ai sources in the tool output. When using
Gemini auth, the existing Google Search integration is used instead.
