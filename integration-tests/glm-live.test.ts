/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, beforeEach, afterEach, expect } from 'vitest';
import { TestRig } from './test-helper.js';

const GLM_MODEL = 'glm-4.7';
const GLM_TEST_TIMEOUT = 60000;

const hasGlmKey = Boolean(process.env['ZAI_API_KEY']);
const describeIfGlm = hasGlmKey ? describe : describe.skip;

function requireGlmKey(): string {
  const key = process.env['ZAI_API_KEY'];
  if (!key) {
    throw new Error('GLM API key not configured');
  }
  return key;
}

function getGlmSettings() {
  return {
    general: {
      disableAutoUpdate: true,
      previewFeatures: true,
    },
    telemetry: {
      enabled: false,
    },
    security: {
      auth: {
        selectedType: 'glm-api-key',
      },
    },
    model: {
      name: GLM_MODEL,
    },
  };
}

function extractGlmMetrics(stats: Record<string, any>) {
  const models = stats?.models ?? {};
  if (models[GLM_MODEL]) {
    return models[GLM_MODEL];
  }
  for (const [name, metrics] of Object.entries(models)) {
    if (name.includes('glm')) {
      return metrics as Record<string, any>;
    }
  }
  throw new Error(`GLM metrics not found. Available keys: ${Object.keys(models)}`);
}

describeIfGlm('GLM live integration', () => {
  let rig: TestRig;

  beforeEach(async () => {
    rig = new TestRig();
  });

  afterEach(async () => {
    await rig.cleanup();
  });

  async function runPrompt(
    prompt: string,
    extraArgs: string[] = [],
    outputFormat: 'json' | 'stream-json' = 'json',
  ) {
    await rig.setup('glm-live-test', { settings: getGlmSettings() });
    const args = ['-p', prompt, '--model', GLM_MODEL, '--output-format', outputFormat, ...extraArgs];
    const result = await rig.run({
      args,
      env: {
        ZAI_API_KEY: requireGlmKey(),
        GEMINI_API_KEY: undefined,
      },
    });
    return result;
  }

  it(
    'streams reasoning tokens and reports stats',
    { timeout: GLM_TEST_TIMEOUT },
    async () => {
      const payload = JSON.parse(
        await runPrompt(
          'Think aloud about counting from one to three before giving the final answer.',
        ),
      );
      expect(payload.response).toBeTruthy();
      expect(typeof payload.session_id).toBe('string');

      const stats = payload.stats;
      expect(stats).toBeTruthy();
      const glmMetrics = extractGlmMetrics(stats);
      const tokens = glmMetrics.tokens ?? {};
      expect(tokens.total).toBeGreaterThan(0);
      expect(tokens.prompt).toBeGreaterThan(0);
      expect(tokens.candidates).toBeGreaterThanOrEqual(0);
      expect(tokens.thoughts).toBeGreaterThan(0);
    },
  );

  it(
    'maintains JSON contract for GLM responses',
    { timeout: GLM_TEST_TIMEOUT },
    async () => {
      const payload = JSON.parse(
        await runPrompt('Summarize why continuous scoring is helpful for scientific benchmarks.'),
      );

      expect(typeof payload.session_id).toBe('string');
      expect(payload.stats).toBeTruthy();
      const glmMetrics = extractGlmMetrics(payload.stats);
      const tokens = glmMetrics.tokens ?? {};
      const total = tokens.total ?? 0;
      const prompt = tokens.prompt ?? 0;
      const candidates = tokens.candidates ?? 0;
      expect(total).toBeGreaterThanOrEqual(prompt + candidates);

      const tools = payload.stats.tools ?? {};
      expect(tools.totalCalls).toBeGreaterThanOrEqual(0);
    },
  );

  it(
    'records tool usage metrics when shell tool is invoked',
    { timeout: GLM_TEST_TIMEOUT },
    async () => {
      const payload = JSON.parse(
        await runPrompt('Use the run_shell_command tool to execute "pwd" and then summarize the output.'),
      );

      const tools = payload.stats?.tools ?? {};
      expect(tools.totalCalls).toBeGreaterThanOrEqual(1);
      expect(tools.totalSuccess).toBeGreaterThanOrEqual(1);
      const shellStats = tools.byName?.['run_shell_command'];
      expect(shellStats).toBeDefined();
      expect(shellStats?.calls).toBeGreaterThanOrEqual(1);

      const glmMetrics = extractGlmMetrics(payload.stats);
      const toolTokens = glmMetrics.tokens?.tool;
      expect((toolTokens ?? 0)).toBeGreaterThanOrEqual(0);
    },
  );

  it(
    'emits structured stream-json events including reasoning and stats',
    { timeout: GLM_TEST_TIMEOUT },
    async () => {
      const raw = await runPrompt(
        'Reason through the steps required to rename a file before answering.',
        [],
        'stream-json',
      );
      const events = raw
        .split('\n')
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => JSON.parse(line));

      expect(events.length).toBeGreaterThan(0);
      const messageEvent = events.find((evt: any) => evt.type === 'message');
      expect(messageEvent).toBeDefined();
      expect(messageEvent?.delta).toBe(true);
      expect(typeof messageEvent?.content).toBe('string');

      const resultEvent = events.find((evt: any) => evt.type === 'result');
      expect(resultEvent).toBeDefined();
      expect(resultEvent?.status).toBe('success');
      expect(resultEvent?.stats).toBeDefined();

      const glmMetrics = extractGlmMetrics(resultEvent!.stats);
      expect(glmMetrics.tokens?.total ?? 0).toBeGreaterThan(0);
    },
  );
});
