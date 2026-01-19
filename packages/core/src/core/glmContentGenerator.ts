/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  Content,
  Part,
  PartListUnion,
  Tool,
  ToolListUnion,
  FunctionCall,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  GenerateContentParameters,
  GenerateContentResponse,
  GenerateContentResponseUsageMetadata,
  FunctionDeclaration,
  FinishReason,
} from '@google/genai';
import { FunctionCallingConfigMode } from '@google/genai';
import { estimateTokenCountSync } from '../utils/tokenCalculation.js';
import { debugLogger } from '../utils/debugLogger.js';

const DEFAULT_GLM_ENDPOINT =
  'https://api.z.ai/api/coding/paas/v4/chat/completions';

const DEFAULT_GLM_MODEL = 'glm-4.7';

function mapModelName(model: string | undefined): string {
  if (!model) {
    return DEFAULT_GLM_MODEL;
  }
  // If it's already a GLM model, use it as-is
  if (model.startsWith('glm-')) {
    return model;
  }
  // Map Gemini model names to GLM equivalent
  return DEFAULT_GLM_MODEL;
}

type PatchedGenerateContentResponse = GenerateContentResponse & {
  functionCalls?: FunctionCall[];
};

function createResponse(
  partial: Partial<PatchedGenerateContentResponse>,
): PatchedGenerateContentResponse {
  return partial as PatchedGenerateContentResponse;
}

interface GlmMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | Array<{ type: string; text?: string }> | null;
  reasoning_content?: string | Array<{ type: string; text?: string }> | null;
  name?: string;
  tool_call_id?: string;
  tool_calls?: GlmToolCall[];
}

interface GlmToolCall {
  id?: string;
  type?: 'function';
  function?: {
    name?: string;
    arguments?: string;
  };
}

interface GlmToolDefinition {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

interface GlmChatCompletionRequest {
  model: string;
  messages: GlmMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream?: boolean;
  tools?: GlmToolDefinition[];
  tool_choice?:
    | 'auto'
    | 'none'
    | 'required'
    | { type: 'function'; function: { name: string } };
  thinking?: {
    type: 'enabled' | 'disabled';
    clear_thinking?: boolean;
  };
}

interface GlmUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  reasoning_tokens?: number;
  completion_tokens_details?: {
    reasoning_tokens?: number;
  };
}

interface GlmChatCompletionChoice {
  index: number;
  message?: {
    role: 'assistant';
    content?: string | Array<{ type: string; text?: string }> | null;
    reasoning_content?: string | Array<{ type: string; text?: string }> | null;
    tool_calls?: GlmToolCall[];
  };
  finish_reason: string | null;
}

interface GlmChatCompletionResponse {
  id?: string;
  model?: string;
  choices?: GlmChatCompletionChoice[];
  usage?: GlmUsage;
}

interface GlmChatCompletionChunkChoice {
  index: number;
  delta?: {
    content?: string | Array<{ type: string; text?: string }> | null;
    reasoning_content?: string | Array<{ type: string; text?: string }> | null;
    tool_calls?: GlmToolCall[];
  };
  finish_reason: string | null;
}

interface GlmChatCompletionChunk {
  id?: string;
  choices?: GlmChatCompletionChunkChoice[];
  usage?: GlmUsage;
  model?: string;
}

interface PendingToolCallState {
  id: string;
  name: string;
  args: string;
}

function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function normalizeContentText(
  content: string | Array<{ type: string; text?: string }> | null | undefined,
): string {
  if (!content) {
    return '';
  }
  if (typeof content === 'string') {
    return content;
  }
  return content
    .map((part) => part.text ?? '')
    .filter((text) => text.length > 0)
    .join('');
}

function normalizeReasoningText(
  content: string | Array<{ type: string; text?: string }> | null | undefined,
): string {
  if (!content) {
    return '';
  }
  if (typeof content === 'string') {
    return content;
  }
  return content
    .map((part) => part.text ?? '')
    .filter((text) => text.length > 0)
    .join('');
}

function toUsageMetadata(
  usage?: GlmUsage,
): GenerateContentResponseUsageMetadata | undefined {
  if (!usage) {
    return undefined;
  }
  const usageMetadata: GenerateContentResponseUsageMetadata = {
    promptTokenCount: usage.prompt_tokens,
    candidatesTokenCount: usage.completion_tokens,
    totalTokenCount: usage.total_tokens,
  };
  const reasoningTokens =
    usage.reasoning_tokens ?? usage.completion_tokens_details?.reasoning_tokens;
  if (reasoningTokens !== undefined) {
    usageMetadata.thoughtsTokenCount = reasoningTokens;
  }
  return usageMetadata;
}

type Contentish = Content | Part | string;

function normalizeContents(
  source?: PartListUnion | Content | Content[],
): Content[] {
  if (!source) {
    return [];
  }
  const toContent = (entry: Contentish): Content => {
    if (typeof entry === 'string') {
      return { role: 'user', parts: [{ text: entry }] };
    }
    if ('role' in (entry as Content)) {
      return entry as Content;
    }
    return { role: 'user', parts: [entry as Part] };
  };
  const arraySource = Array.isArray(source) ? source : [source];
  return arraySource.map((item) => toContent(item as Contentish));
}

function extractTools(toolList: ToolListUnion | undefined): Tool[] {
  if (!toolList) {
    return [];
  }
  const array = Array.isArray(toolList) ? toolList : [toolList];
  const result: Tool[] = [];
  for (const item of array) {
    if ((item as Tool).functionDeclarations) {
      result.push(item as Tool);
    }
  }
  return result;
}

function convertFunctionDeclarations(tools: Tool[] = []): GlmToolDefinition[] {
  const definitions: GlmToolDefinition[] = [];
  for (const tool of tools) {
    if (!tool.functionDeclarations) {
      continue;
    }
    for (const declaration of tool.functionDeclarations) {
      if (!declaration || !declaration.name) {
        continue;
      }
      const parameters =
        (declaration.parametersJsonSchema as Record<string, unknown>) ??
        (declaration.parameters as Record<string, unknown>) ??
        {};
      definitions.push({
        type: 'function',
        function: {
          name: declaration.name,
          description: declaration.description,
          parameters,
        },
      });
    }
  }
  return definitions;
}

function normalizeAllowedFunctions(
  declarations: FunctionDeclaration[] = [],
  allowed?: string[],
): FunctionDeclaration[] {
  if (!allowed || allowed.length === 0) {
    return declarations;
  }
  const allowedSet = new Set(allowed);
  return declarations.filter((decl) => decl.name && allowedSet.has(decl.name));
}

function applyFunctionFilters(tools: Tool[] = [], allowed?: string[]): Tool[] {
  if (!allowed || allowed.length === 0) {
    return tools;
  }
  return tools
    .map((tool) => {
      if (!tool.functionDeclarations) {
        return tool;
      }
      const filtered = normalizeAllowedFunctions(
        tool.functionDeclarations,
        allowed,
      );
      return { ...tool, functionDeclarations: filtered };
    })
    .filter((tool) => (tool.functionDeclarations?.length ?? 0) > 0);
}

function convertFinishReason(
  reason: string | null | undefined,
): FinishReason | undefined {
  switch (reason) {
    case 'stop':
      return 'STOP' as FinishReason;
    case 'length':
      return 'MAX_TOKENS' as FinishReason;
    case 'content_filter':
      return 'SAFETY' as FinishReason;
    case 'tool_calls':
      return 'STOP' as FinishReason;
    default:
      return undefined;
  }
}

function buildToolChoice(
  mode: FunctionCallingConfigMode | undefined,
  allowed?: string[],
):
  | 'auto'
  | 'none'
  | 'required'
  | { type: 'function'; function: { name: string } }
  | undefined {
  if (!mode || mode === FunctionCallingConfigMode.AUTO) {
    return undefined;
  }
  if (mode === FunctionCallingConfigMode.NONE) {
    return 'none';
  }
  if (
    (mode === FunctionCallingConfigMode.ANY ||
      mode === FunctionCallingConfigMode.VALIDATED) &&
    allowed &&
    allowed.length === 1
  ) {
    return { type: 'function', function: { name: allowed[0] } };
  }
  if (
    mode === FunctionCallingConfigMode.ANY ||
    mode === FunctionCallingConfigMode.VALIDATED
  ) {
    return 'required';
  }
  return undefined;
}

export class GlmContentGenerator {
  readonly userTier = undefined;

  constructor(
    private readonly options: {
      apiKey: string;
      userAgent: string;
      endpoint?: string;
      clearThinking?: boolean;
      thinkingEnabled?: boolean;
      extraHeaders?: Record<string, string>;
    },
  ) {}

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const payload = this.toPayload(request, false);
    const response = await this.postCompletion(
      payload,
      request.config?.abortSignal,
    );
    const data = (await response.json()) as GlmChatCompletionResponse;
    if (data.choices?.length) {
      return this.toGenerateContentResponse(data);
    }
    throw new Error('GLM API returned no choices');
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const payload = this.toPayload(request, true);
    const response = await this.postCompletion(
      payload,
      request.config?.abortSignal,
    );
    const body = response.body;
    if (!body) {
      throw new Error('GLM API did not return a response body for streaming');
    }
    return this.streamFromSSE(body, request.model ?? 'glm');
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    const contents = normalizeContents(
      request.contents as Content[] | PartListUnion,
    );
    const parts: Part[] = contents.flatMap((content) => content.parts ?? []);
    return { totalTokens: estimateTokenCountSync(parts) };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error('Embeddings are not supported when using GLM auth');
  }

  private async postCompletion(
    payload: GlmChatCompletionRequest,
    signal: AbortSignal | undefined,
  ): Promise<Response> {
    const endpoint = this.options.endpoint ?? DEFAULT_GLM_ENDPOINT;
    debugLogger.debug(`[GLM] POST ${endpoint} model=${payload.model}`);
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.options.apiKey}`,
        'User-Agent': this.options.userAgent,
        ...this.options.extraHeaders,
      },
      body: JSON.stringify(payload),
      signal,
    });
    if (!response.ok) {
      const text = await response.text().catch(() => '');
      throw new Error(
        `GLM API request failed (${response.status}): ${text || response.statusText}`,
      );
    }
    return response;
  }

  private toPayload(
    request: GenerateContentParameters,
    stream: boolean,
  ): GlmChatCompletionRequest {
    const messages = this.buildMessages(request);
    const payload: GlmChatCompletionRequest = {
      model: mapModelName(request.model),
      messages,
      temperature: request.config?.temperature ?? undefined,
      top_p: request.config?.topP ?? undefined,
      max_tokens: request.config?.maxOutputTokens ?? undefined,
      stream,
    };
    const toolConfig = request.config?.toolConfig;
    let tools: Tool[] = extractTools(request.config?.tools as ToolListUnion);
    const allowedFunctions =
      toolConfig?.functionCallingConfig?.allowedFunctionNames;
    if (allowedFunctions && allowedFunctions.length > 0) {
      tools = applyFunctionFilters(tools, allowedFunctions);
    }
    const toolDefinitions = convertFunctionDeclarations(tools);
    if (toolDefinitions.length > 0) {
      payload.tools = toolDefinitions;
    }
    const toolChoice = buildToolChoice(
      toolConfig?.functionCallingConfig?.mode,
      allowedFunctions,
    );
    if (toolChoice) {
      payload.tool_choice = toolChoice;
    }
    const thinkingEnabled = this.options.thinkingEnabled ?? true;
    if (thinkingEnabled) {
      payload.thinking = {
        type: 'enabled',
        clear_thinking: this.options.clearThinking ?? false,
      };
    } else {
      payload.thinking = { type: 'disabled' };
    }
    return payload;
  }

  private buildMessages(request: GenerateContentParameters): GlmMessage[] {
    const messages: GlmMessage[] = [];
    const systemInstruction = request.config?.systemInstruction;
    if (systemInstruction) {
      messages.push({
        role: 'system',
        content: this.extractSystemInstruction(systemInstruction),
      });
    }
    const contents = normalizeContents(request.contents);
    for (const content of contents) {
      messages.push(...this.convertContent(content));
    }
    return messages;
  }

  private extractSystemInstruction(
    instruction: GenerateContentParameters['config'] extends {
      systemInstruction?: infer T;
    }
      ? T
      : unknown,
  ): string {
    if (typeof instruction === 'string') {
      return instruction;
    }
    if (!instruction) {
      return '';
    }
    if (Array.isArray((instruction as Content).parts)) {
      const content = instruction as Content;
      return (content.parts ?? [])
        .map((part) => part.text)
        .filter((text): text is string => Boolean(text))
        .join('\n');
    }
    if (Array.isArray(instruction)) {
      return instruction
        .map((part) => (typeof part === 'string' ? part : (part.text ?? '')))
        .join('\n');
    }
    return String(instruction);
  }

  private convertContent(content: Content): GlmMessage[] {
    const role = content.role === 'model' ? 'assistant' : 'user';
    const textParts: string[] = [];
    const reasoningParts: string[] = [];
    const messages: GlmMessage[] = [];
    const toolCalls: GlmToolCall[] = [];

    for (const part of content.parts ?? []) {
      if (part.thought) {
        if (part.text) {
          reasoningParts.push(part.text);
        }
        continue;
      }
      if (part.text) {
        textParts.push(part.text);
        continue;
      }
      if (part.functionResponse && role === 'user') {
        messages.push({
          role: 'tool',
          name: part.functionResponse.name,
          tool_call_id:
            part.functionResponse.id ?? part.functionResponse.name ?? 'tool',
          content: safeJsonStringify(part.functionResponse.response ?? {}),
        });
        continue;
      }
      if (part.functionCall && role === 'assistant') {
        toolCalls.push({
          id: part.functionCall.id,
          type: 'function',
          function: {
            name: part.functionCall.name,
            arguments: safeJsonStringify(part.functionCall.args ?? {}),
          },
        });
        continue;
      }
      if (part.inlineData || part.fileData) {
        const description =
          part.inlineData?.mimeType || part.fileData?.mimeType;
        textParts.push(
          `[Attachment omitted${description ? `: ${description}` : ''}]`,
        );
      }
    }

    if (role === 'assistant') {
      if (
        textParts.length > 0 ||
        toolCalls.length > 0 ||
        reasoningParts.length > 0
      ) {
        messages.push({
          role: 'assistant',
          content: textParts.length > 0 ? textParts.join('\n') : '',
          reasoning_content:
            reasoningParts.length > 0 ? reasoningParts.join('') : undefined,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        });
      }
    } else if (textParts.length > 0) {
      messages.push({ role: 'user', content: textParts.join('\n') });
    }

    return messages;
  }

  private toGenerateContentResponse(
    response: GlmChatCompletionResponse,
  ): GenerateContentResponse {
    const choice =
      response.choices && response.choices.length > 0
        ? response.choices[0]
        : undefined;
    const parts: Part[] = [];
    const reasoningText = normalizeReasoningText(
      choice?.message?.reasoning_content,
    );
    if (reasoningText) {
      parts.push({ text: reasoningText, thought: true });
    }
    const messageText = normalizeContentText(choice?.message?.content);
    if (messageText) {
      parts.push({ text: messageText });
    }
    const functionCalls = (choice?.message?.tool_calls ?? []).map(
      (toolCall) => ({
        functionCall: {
          id: toolCall.id,
          name: toolCall.function?.name,
          args: this.parseArguments(toolCall.function?.arguments),
        },
      }),
    );
    if (functionCalls.length > 0) {
      parts.push(...functionCalls);
    }
    return createResponse({
      responseId: response.id,
      modelVersion: response.model,
      candidates: [
        {
          content: { role: 'model', parts },
          finishReason: convertFinishReason(choice?.finish_reason),
        },
      ],
      usageMetadata: toUsageMetadata(response.usage),
      functionCalls: functionCalls.map((part) => part.functionCall),
    });
  }

  private parseArguments(argumentString: string | undefined) {
    if (!argumentString) {
      return {};
    }
    try {
      return JSON.parse(argumentString);
    } catch {
      return { raw: argumentString };
    }
  }

  private async *streamFromSSE(
    body: ReadableStream<Uint8Array>,
    model: string,
  ): AsyncGenerator<GenerateContentResponse> {
    const reader = body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    const pendingToolCalls = new Map<string, PendingToolCallState>();

    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
          const chunk = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          yield* this.processSseChunk(chunk, model, pendingToolCalls);
          boundary = buffer.indexOf('\n\n');
        }
      }
      if (buffer.trim().length > 0) {
        yield* this.processSseChunk(buffer, model, pendingToolCalls);
      }
    } finally {
      reader.releaseLock();
    }
  }

  private *processSseChunk(
    chunk: string,
    model: string,
    pendingToolCalls: Map<string, PendingToolCallState>,
  ): Generator<GenerateContentResponse> {
    const lines = chunk.split('\n').map((line) => line.trim());
    for (const line of lines) {
      if (!line.startsWith('data:')) {
        continue;
      }
      const data = line.slice(5).trim();
      if (!data || data === '[DONE]') {
        continue;
      }
      try {
        const parsed = JSON.parse(data) as GlmChatCompletionChunk;
        yield* this.chunkToResponses(parsed, model, pendingToolCalls);
      } catch (error) {
        debugLogger.warn('Failed to parse GLM SSE chunk', error);
      }
    }
  }

  private *chunkToResponses(
    chunk: GlmChatCompletionChunk,
    model: string,
    pendingToolCalls: Map<string, PendingToolCallState>,
  ): Generator<GenerateContentResponse> {
    for (const choice of chunk.choices ?? []) {
      const parts: Part[] = [];
      const responses: GenerateContentResponse[] = [];
      const reasoningDelta = normalizeReasoningText(
        choice.delta?.reasoning_content,
      );
      if (reasoningDelta) {
        parts.push({ text: reasoningDelta, thought: true });
      }
      const textDelta = normalizeContentText(choice.delta?.content);
      if (textDelta) {
        parts.push({ text: textDelta });
      }
      if (parts.length > 0) {
        responses.push(
          createResponse({
            responseId: chunk.id,
            modelVersion: chunk.model ?? model,
            candidates: [
              {
                content: { role: 'model', parts },
                finishReason: convertFinishReason(choice.finish_reason),
              },
            ],
            usageMetadata: toUsageMetadata(chunk.usage),
          }),
        );
      }
      if (choice.delta?.tool_calls) {
        for (const call of choice.delta.tool_calls) {
          const id = call.id ?? call.function?.name ?? 'tool';
          const existing = pendingToolCalls.get(id) ?? {
            id,
            name: call.function?.name ?? 'tool',
            args: '',
          };
          existing.args += call.function?.arguments ?? '';
          existing.name = call.function?.name ?? existing.name;
          pendingToolCalls.set(id, existing);
        }
      }
      const finishReason = convertFinishReason(choice.finish_reason);
      if (
        (finishReason === 'STOP' || finishReason === 'MAX_TOKENS') &&
        pendingToolCalls.size > 0
      ) {
        const functionCalls = Array.from(pendingToolCalls.values()).map(
          ({ id, name, args }) => ({
            functionCall: {
              id,
              name,
              args: this.parseArguments(args),
            },
          }),
        );
        pendingToolCalls.clear();
        responses.push(
          createResponse({
            responseId: chunk.id,
            modelVersion: chunk.model ?? model,
            candidates: [
              {
                content: { role: 'model', parts: functionCalls },
                finishReason,
              },
            ],
            functionCalls: functionCalls.map((part) => part.functionCall),
            usageMetadata: toUsageMetadata(chunk.usage),
          }),
        );
      } else if (!parts.length && finishReason) {
        responses.push(
          createResponse({
            responseId: chunk.id,
            modelVersion: chunk.model ?? model,
            candidates: [
              {
                content: { role: 'model', parts: [] },
                finishReason,
              },
            ],
            usageMetadata: toUsageMetadata(chunk.usage),
          }),
        );
      }
      for (const resp of responses) {
        yield resp;
      }
    }
  }
}
