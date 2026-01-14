/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { GlmContentGenerator } from './glmContentGenerator.js';

const DEFAULT_OPTIONS = {
  apiKey: 'test-key',
  userAgent: 'test-agent',
};

describe('GlmContentGenerator', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('converts basic responses with reasoning and tool calls', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue({
        ok: true,
        json: async () => ({
          id: 'resp-1',
          model: 'glm-4.7',
          usage: {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            reasoning_tokens: 2,
          },
          choices: [
            {
              finish_reason: 'stop',
              index: 0,
              message: {
                content: [{ type: 'text', text: 'Hello world' }],
                reasoning_content: [{ type: 'text', text: 'thinking' }],
                tool_calls: [
                  {
                    id: 'call_1',
                    function: {
                      name: 'do_work',
                      arguments: '{"path":"foo"}',
                    },
                  },
                ],
              },
            },
          ],
        }),
      } as unknown as Response);

    const generator = new GlmContentGenerator(DEFAULT_OPTIONS);
    const response = await generator.generateContent(
      {
        model: 'glm-4.7',
        contents: [],
        config: {},
      },
      'prompt-123',
    );

    expect(fetchMock).toHaveBeenCalled();
    const fetchArgs = fetchMock.mock.calls[0];
    const requestPayload = JSON.parse(
      (fetchArgs[1] as RequestInit).body as string,
    );
    expect(requestPayload.request_id).toBe('prompt-123');
    expect(requestPayload.thinking).toEqual({
      type: 'enabled',
      clear_thinking: true,
    });
    expect(response.candidates?.[0]?.content?.parts?.length).toBe(3);
    const [thought, text, fnCall] = response.candidates?.[0]?.content?.parts ?? [];
    expect(thought?.thought).toBe(true);
    expect(thought?.text).toBe('thinking');
    expect(text?.text).toBe('Hello world');
    expect(fnCall?.functionCall?.name).toBe('do_work');
    expect(fnCall?.functionCall?.args).toEqual({ path: 'foo' });
    expect(response.usageMetadata?.promptTokenCount).toBe(10);
    expect(response.usageMetadata?.thoughtsTokenCount).toBe(2);
    expect(response.functionCalls?.[0]?.name).toBe('do_work');
  });

  it('streams SSE chunks for text and tool calls', async () => {
    const textEncoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          textEncoder.encode(
            'data: {"id":"chunk","model":"glm-4.7","choices":[{"delta":{"content":[{"type":"text","text":"Hi"}]},"finish_reason":null}]}' +
              '\n\n',
          ),
        );
        controller.enqueue(
          textEncoder.encode(
            'data: {"id":"chunk","model":"glm-4.7","choices":[{"delta":{"tool_calls":[{"id":"call","function":{"name":"plan","arguments":"{\\"x\\":1}"}}]},"finish_reason":"stop"}]}' +
              '\n\n',
          ),
        );
        controller.close();
      },
    });
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      body: stream,
    } as unknown as Response);

    const generator = new GlmContentGenerator(DEFAULT_OPTIONS);
    const streamIterator = await generator.generateContentStream(
      {
        model: 'glm-4.7',
        contents: [],
        config: {},
      },
      'prompt-321',
    );

    const chunks: Array<Awaited<ReturnType<typeof streamIterator.next>>> = [];
    chunks.push(await streamIterator.next());
    chunks.push(await streamIterator.next());
    chunks.push(await streamIterator.next());

    const firstValue = chunks[0].value;
    expect(firstValue?.candidates?.[0]?.content?.parts?.[0]?.text).toBe('Hi');
    const secondValue = chunks[1].value;
    expect(
      secondValue?.candidates?.[0]?.content?.parts?.[0]?.functionCall?.name,
    ).toBe('plan');
    expect(streamIterator.next).toBeDefined();
  });

  it('falls back to secondary endpoint when default coding path fails', async () => {
    const firstResponse = {
      ok: false,
      status: 404,
      statusText: 'Not Found',
      text: async () => 'missing',
    } as Response;
    const secondResponse = {
      ok: true,
      json: async () => ({
        choices: [
          {
            finish_reason: 'stop',
            index: 0,
            message: { content: 'hi' },
          },
        ],
      }),
    } as Response;
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(firstResponse)
      .mockResolvedValueOnce(secondResponse);

    const generator = new GlmContentGenerator(DEFAULT_OPTIONS);
    await generator.generateContent(
      {
        model: 'glm-4.7',
        contents: [],
        config: {},
      },
      'prompt-456',
    );

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'https://api.z.ai/api/coding/paas/v4/chat/completions',
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'https://api.z.ai/api/paas/v4/chat/completions',
      expect.any(Object),
    );
  });

  it('disables thinking when user opts out', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue({
        ok: true,
        json: async () => ({
          choices: [
            {
              finish_reason: 'stop',
              index: 0,
              message: { content: 'hello' },
            },
          ],
        }),
      } as unknown as Response);

    const generator = new GlmContentGenerator(DEFAULT_OPTIONS);
    await generator.generateContent(
      {
        model: 'glm-4.7',
        contents: [],
        config: {
          thinkingConfig: {
            includeThoughts: false,
          },
        },
      },
      'prompt-789',
    );

    const body = JSON.parse(
      (fetchMock.mock.calls[0][1] as RequestInit).body as string,
    );
    expect(body.thinking).toEqual({ type: 'disabled', clear_thinking: true });
  });
});
