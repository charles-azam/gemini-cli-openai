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

## Installation (local)

This fork does not collide with Gemini CLI settings because it uses a separate
config directory (`.gemini-cli-zai`) and distinct API key storage entries. To
avoid command name collisions, use a different shell alias or wrapper script
instead of `gemini`.

1. Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-org>/gemini-cli-zai.git
cd gemini-cli-zai
npm ci
```

2. Build and bundle:

```bash
npm run build
npm run bundle
```

3. Map a shortcut to the bundled CLI (example for zsh):

```bash
echo 'alias gemini-zai="node /path/to/gemini-cli-zai/bundle/gemini.js"' >> ~/.zshrc
source ~/.zshrc
```

Alternative: (example for bash):

```bash
echo 'alias gemini-zai="node /path/to/gemini-cli-zai/bundle/gemini.js"' >> ~/.bashrc
source ~/.bashrc
```

Alternative: create a wrapper script (no alias required):

```bash
cat > ~/bin/gemini-zai <<'EOF'
#!/usr/bin/env bash
exec node /path/to/gemini-cli-zai/bundle/gemini.js "$@"
EOF
chmod +x ~/bin/gemini-zai
```
