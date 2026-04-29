// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import 'package:genkit/plugin.dart';
import 'package:http/http.dart' as http;
import 'package:meta/meta.dart';
import 'package:openai_dart/openai_dart.dart' as sdk;
import 'package:schemantic/schemantic.dart';

import '../genkit_openai.dart';
import 'chat.dart' as chat;

/// Builds an OpenAI Responses [sdk.TextConfig] from a Genkit output schema.
sdk.TextConfig? buildOpenAITextConfig(
  bool isJsonMode,
  Map<String, dynamic>? schema,
) {
  if (!isJsonMode) {
    return null;
  }
  if (schema == null) {
    return const sdk.TextConfig(format: sdk.JsonObjectFormat());
  }
  final flattened = schema.flatten();
  return sdk.TextConfig(
    format: sdk.JsonSchemaFormat(
      name: 'output',
      schema: {...flattened, 'additionalProperties': false},
      strict: true,
    ),
  );
}

/// Core Genkit plugin implementation for OpenAI-compatible APIs.
///
/// Automatically discovers models from the OpenAI API (when no custom
/// [baseUrl] is set) and registers them in the Genkit action registry.
/// Additional models can be provided via [customModels].
class OpenAIPlugin extends GenkitPlugin {
  final String _pluginName;

  @override
  String get name => _pluginName;

  /// The static API key used to authenticate requests.
  final String? apiKey;

  /// An asynchronous callback that returns the API key on each request.
  final OpenAIApiKeyProvider? apiKeyProvider;

  /// Custom base URL for OpenAI-compatible APIs (e.g. Groq, DeepSeek).
  final String? baseUrl;

  /// Additional models to register beyond those discovered from the API.
  final List<CustomModelDefinition> customModels;

  /// Extra HTTP headers sent with every request.
  final Map<String, String>? headers;

  /// Optional HTTP client for dependency injection and testing.
  final http.Client? httpClient;
  final OpenAIWireApi wireApi;

  /// Creates an [OpenAIPlugin].
  ///
  /// Provide either [apiKey] or [apiKeyProvider], but not both.
  OpenAIPlugin({
    String name = defaultOpenAINamespace,
    this.apiKey,
    this.apiKeyProvider,
    this.baseUrl,
    this.customModels = const [],
    this.headers,
    this.httpClient,
    this.wireApi = OpenAIWireApi.responses,
  }) : _pluginName = name {
    if (name.isEmpty || name.contains('/')) {
      throw GenkitException(
        'Plugin name must be non-empty and must not contain "/". Got: "$name"',
        status: StatusCodes.INVALID_ARGUMENT,
      );
    }
    if (apiKey != null && apiKeyProvider != null) {
      throw GenkitException(
        'Provide either apiKey or apiKeyProvider, not both.',
        status: StatusCodes.INVALID_ARGUMENT,
      );
    }
  }

  @override
  Future<List<Action>> init() async {
    final actions = <Action>[];

    // Fetch and register models from OpenAI API only for default OpenAI host.
    if (baseUrl == null) {
      try {
        final availableModelIds = await _fetchAvailableModels();

        for (final modelId in availableModelIds) {
          final modelType = getModelType(modelId);

          if (modelType != 'chat' && modelType != 'unknown') {
            continue;
          }

          final info = modelInfoFor(modelId);
          actions.add(_createModel(modelId, info));
        }
      } catch (e) {
        throw GenkitException(
          'Error fetching available models from $_pluginName: $e',
          underlyingException: e,
        );
      }
    }

    // Register custom models
    for (final model in customModels) {
      actions.add(_createModel(model.name, model.info));
    }

    return actions;
  }

  /// Fetch available model IDs from OpenAI API
  Future<List<String>> _fetchAvailableModels() async {
    final resolvedConfig = await _resolveClientConfig();

    final client = sdk.OpenAIClient.withApiKey(
      resolvedConfig.apiKey,
      baseUrl: resolvedConfig.baseUrl,
      defaultHeaders: resolvedConfig.headers,
      httpClient: httpClient,
    );

    try {
      final response = await client.models.list();
      final modelIds = <String>[];

      // Collect all model IDs
      for (final model in response.data) {
        modelIds.add(model.id);
      }

      return modelIds;
    } finally {
      if (httpClient == null) {
        client.close();
      }
    }
  }

  Future<_ResolvedClientConfig> _resolveClientConfig() async {
    final configuredApiKey = await _resolveApiKey();
    if (configuredApiKey == null || configuredApiKey.trim().isEmpty) {
      throw GenkitException(
        '[$_pluginName] API key is required. Provide it via apiKey or apiKeyProvider in the plugin constructor.',
        status: StatusCodes.INVALID_ARGUMENT,
      );
    }

    return _ResolvedClientConfig(
      apiKey: configuredApiKey.trim(),
      baseUrl: baseUrl,
      headers: headers,
    );
  }

  Future<String?> _resolveApiKey() async {
    final configuredApiKeyProvider = apiKeyProvider;
    if (configuredApiKeyProvider != null) {
      return await configuredApiKeyProvider();
    }
    return apiKey;
  }

  @override
  Future<List<ActionMetadata<dynamic, dynamic, dynamic, dynamic>>>
  list() async {
    try {
      final modelIds = await _fetchAvailableModels();
      final modelMetadataList =
          <ActionMetadata<dynamic, dynamic, dynamic, dynamic>>[];

      for (final modelId in modelIds) {
        final modelType = getModelType(modelId);
        if (modelType != 'chat' && modelType != 'unknown') {
          continue;
        }

        modelMetadataList.add(
          modelMetadata(
            '$_pluginName/$modelId',
            modelInfo: modelInfoFor(modelId),
            customOptions: chat.chatModelOptionsSchema(),
          ),
        );
      }

      return modelMetadataList;
    } catch (e, stackTrace) {
      throw GenkitException(
        'Error listing models from $_pluginName: $e',
        underlyingException: e,
        stackTrace: stackTrace,
      );
    }
  }

  @override
  Action? resolve(String actionType, String name) {
    if (actionType == 'model') {
      return _createModel(name, null);
    }
    return null;
  }

  Model _createModel(String modelName, ModelInfo? info) {
    final modelInfo = info ?? modelInfoFor(modelName);

    return Model(
      name: '$_pluginName/$modelName',
      customOptions: chat.chatModelOptionsSchema(),
      metadata: {'model': modelInfo.toJson()},
      fn: (req, ctx) async {
        final modelRequest = req!;
        final options = chat.parseChatModelOptions(modelRequest.config);
        _validateReasoningEffort(options.reasoningEffort);
        _validateReasoningSummary(options.reasoningSummary);

        final resolvedConfig = await _resolveClientConfig();
        final client = sdk.OpenAIClient.withApiKey(
          resolvedConfig.apiKey,
          baseUrl: resolvedConfig.baseUrl,
          defaultHeaders: resolvedConfig.headers,
          httpClient: httpClient,
        );

        try {
          final supports = modelInfo.supports;
          final supportsTools = supports?['tools'] == true;

          final isJsonMode = chat.isJsonStructuredOutput(
            modelRequest.output?.format,
            modelRequest.output?.contentType,
          );
          final responseFormat = chat.buildOpenAIResponseFormat(
            modelRequest.output?.schema,
          );
          switch (wireApi) {
            case OpenAIWireApi.responses:
              final request = sdk.CreateResponseRequest(
                model: options.version ?? modelName,
                input: GenkitConverter.toOpenAIResponseInput(
                  modelRequest.messages,
                  options.visualDetailLevel,
                ),
                tools: supportsTools
                    ? modelRequest.tools
                          ?.map(GenkitConverter.toOpenAIResponseTool)
                          .toList()
                    : null,
                maxOutputTokens: options.maxTokens,
                temperature: options.temperature,
                topP: options.topP,
                presencePenalty: options.presencePenalty,
                frequencyPenalty: options.frequencyPenalty,
                parallelToolCalls: supportsTools ? true : null,
                reasoning: _buildResponsesReasoningConfig(options),
                text: buildOpenAITextConfig(
                  isJsonMode,
                  modelRequest.output?.schema,
                ),
              );
              if (ctx.streamingRequested) {
                return await _handleResponsesStreaming(client, request, ctx);
              }
              return await _handleResponsesNonStreaming(client, request);
            case OpenAIWireApi.chatCompletions:
              final request = sdk.ChatCompletionCreateRequest(
                model: options.version ?? modelName,
                messages: GenkitConverter.toOpenAIMessages(
                  modelRequest.messages,
                  options.visualDetailLevel,
                ),
                tools: supportsTools
                    ? modelRequest.tools
                          ?.map(GenkitConverter.toOpenAITool)
                          .toList()
                    : null,
                temperature: options.temperature,
                topP: options.topP,
                maxCompletionTokens: options.maxTokens,
                stop: options.stop,
                presencePenalty: options.presencePenalty,
                frequencyPenalty: options.frequencyPenalty,
                seed: options.seed,
                user: options.user,
                reasoningEffort: _mapReasoningEffort(options.reasoningEffort),
                openRouterReasoning: _mapOpenRouterReasoning(
                  options.reasoningSummary,
                ),
                responseFormat: isJsonMode ? responseFormat : null,
              );
              if (ctx.streamingRequested) {
                return await _handleStreamingWithReasoningSummaryFallback(
                  client,
                  request,
                  ctx,
                );
              } else {
                return await _handleNonStreamingWithReasoningSummaryFallback(
                  client,
                  request,
                );
              }
          }
        } catch (e, stackTrace) {
          if (e is GenkitException) {
            rethrow;
          }

          StatusCodes? status;
          String? details;

          if (e is sdk.ApiException) {
            status = StatusCodes.fromHttpStatus(e.statusCode);
            details = e.body?.toString();
          }

          throw GenkitException(
            'OpenAI API error: $e',
            status: status,
            details: details ?? e.toString(),
            underlyingException: e,
            stackTrace: stackTrace,
          );
        } finally {
          if (httpClient == null) {
            client.close();
          }
        }
      },
    );
  }

  /// Handle streaming response
  Future<ModelResponse> _handleStreamingWithReasoningSummaryFallback(
    sdk.OpenAIClient client,
    sdk.ChatCompletionCreateRequest request,
    ({
      bool streamingRequested,
      void Function(ModelResponseChunk) sendChunk,
      Map<String, dynamic>? context,
      Stream<ModelRequest>? inputStream,
      void init,
    })
    ctx,
  ) async {
    var receivedAnyChunk = false;
    try {
      return await _handleStreaming(
        client,
        request,
        ctx,
        onChunkReceived: () => receivedAnyChunk = true,
      );
    } on sdk.ApiException catch (error) {
      if (!_shouldRetryWithoutReasoningSummary(
        error: error,
        request: request,
        receivedAnyChunk: receivedAnyChunk,
      )) {
        rethrow;
      }

      final fallbackRequest = request.copyWith(openRouterReasoning: null);
      return _handleStreaming(client, fallbackRequest, ctx);
    }
  }

  Future<ModelResponse> _handleStreaming(
    sdk.OpenAIClient client,
    sdk.ChatCompletionCreateRequest request,
    ({
      bool streamingRequested,
      void Function(ModelResponseChunk) sendChunk,
      Map<String, dynamic>? context,
      Stream<ModelRequest>? inputStream,
      void init,
    })
    ctx, {
    void Function()? onChunkReceived,
  }) async {
    final stream = client.chat.completions.createStream(request);
    final accumulator = sdk.ChatStreamAccumulator();

    try {
      await for (final chunk in stream) {
        onChunkReceived?.call();
        accumulator.add(chunk);

        final textDelta = chunk.textDelta;
        if (textDelta != null) {
          ctx.sendChunk(
            ModelResponseChunk(index: 0, content: [TextPart(text: textDelta)]),
          );
        }
        final delta = chunk.firstChoice?.delta;
        if (delta?.reasoningDetails != null) {
          for (final detail in delta!.reasoningDetails!.indexed) {
            if (!detail.$2.isSummary || (detail.$2.text?.isEmpty ?? true)) {
              continue;
            }
            ctx.sendChunk(
              ModelResponseChunk(
                index: 0,
                content: <Part>[
                  ReasoningPart(
                    reasoning: detail.$2.text!,
                    metadata: <String, dynamic>{
                      'reasoningType': 'summary',
                      if (detail.$1 > 0) 'sectionBreak': true,
                    },
                  ),
                ],
              ),
            );
          }
        } else if (delta?.reasoning != null && delta!.reasoning!.isNotEmpty) {
          ctx.sendChunk(
            ModelResponseChunk(
              index: 0,
              content: <Part>[
                ReasoningPart(
                  reasoning: delta.reasoning!,
                  metadata: const <String, dynamic>{'reasoningType': 'summary'},
                ),
              ],
            ),
          );
        }
      }
    } catch (e, stackTrace) {
      if (e is GenkitException) rethrow;
      throw GenkitException(
        'Error in streaming: $e',
        underlyingException: e,
        stackTrace: stackTrace,
      );
    }

    final response = accumulator.toChatCompletion();
    final choice = response.choices.first;
    final message = GenkitConverter.fromOpenAIAssistantMessage(choice.message);

    return ModelResponse(
      finishReason: GenkitConverter.mapFinishReason(choice.finishReason?.name),
      message: message,
      raw: response.toJson(),
    );
  }

  Future<ModelResponse> _handleResponsesStreaming(
    sdk.OpenAIClient client,
    sdk.CreateResponseRequest request,
    ({
      bool streamingRequested,
      void Function(ModelResponseChunk) sendChunk,
      Map<String, dynamic>? context,
      Stream<ModelRequest>? inputStream,
      void init,
    })
    ctx,
  ) async {
    sdk.Response? finalResponse;
    final seenSummarySections = <String>{};
    final streamState = _ResponsesStreamState();

    try {
      await for (final event in client.responses.createStream(request)) {
        streamState.add(event);
        switch (event) {
          case sdk.OutputTextDeltaEvent():
            final parts = _assistantTextPartsFromResponsesEvent(
              event,
              streamState,
            );
            if (parts.isNotEmpty) {
              ctx.sendChunk(ModelResponseChunk(index: 0, content: parts));
            }
          case sdk.OutputTextDoneEvent():
            final parts = _assistantTextPartsFromResponsesEvent(
              event,
              streamState,
            );
            if (parts.isNotEmpty) {
              ctx.sendChunk(ModelResponseChunk(index: 0, content: parts));
            }
          case sdk.ContentPartDoneEvent():
            final parts = _assistantTextPartsFromResponsesEvent(
              event,
              streamState,
            );
            if (parts.isNotEmpty) {
              ctx.sendChunk(ModelResponseChunk(index: 0, content: parts));
            }
          case sdk.ReasoningSummaryTextDeltaEvent(
            :final itemId,
            :final outputIndex,
            :final summaryIndex,
            :final delta,
          ):
            if (delta.isEmpty) {
              continue;
            }
            final sectionKey = '${itemId ?? outputIndex}:$summaryIndex';
            final isNewSection = seenSummarySections.add(sectionKey);
            ctx.sendChunk(
              ModelResponseChunk(
                index: 0,
                content: <Part>[
                  ReasoningPart(
                    reasoning: delta,
                    metadata: <String, dynamic>{
                      'reasoningType': 'summary',
                      if (isNewSection && seenSummarySections.length > 1)
                        'sectionBreak': true,
                    },
                  ),
                ],
              ),
            );
          case sdk.OutputItemDoneEvent():
            final parts = _assistantTextPartsFromResponsesEvent(
              event,
              streamState,
            );
            if (parts.isNotEmpty) {
              ctx.sendChunk(ModelResponseChunk(index: 0, content: parts));
            }
          case sdk.ResponseCompletedEvent(:final response):
            finalResponse = response;
          case sdk.ResponseIncompleteEvent(:final response):
            finalResponse = response;
          case sdk.ResponseFailedEvent(:final response):
            finalResponse = response;
          case sdk.ErrorEvent(:final message, :final code):
            throw GenkitException(
              'OpenAI Responses API error: $message',
              status: StatusCodes.INTERNAL,
              details: code,
            );
          default:
            break;
        }
      }
    } catch (e, stackTrace) {
      if (e is GenkitException) rethrow;
      throw GenkitException(
        'Error in responses streaming: $e',
        underlyingException: e,
        stackTrace: stackTrace,
      );
    }

    final response = finalResponse;
    if (response == null) {
      throw GenkitException('Responses stream ended without a final response.');
    }
    return _modelResponseFromOpenAIResponse(
      _rebuildResponseFromResponsesStream(response, streamState),
    );
  }

  /// Handle non-streaming response
  Future<ModelResponse> _handleResponsesNonStreaming(
    sdk.OpenAIClient client,
    sdk.CreateResponseRequest request,
  ) async {
    final response = await client.responses.create(request);
    return _modelResponseFromOpenAIResponse(response);
  }

  ModelResponse _modelResponseFromOpenAIResponse(sdk.Response response) {
    return ModelResponse(
      finishReason: GenkitConverter.mapResponseFinishReason(response),
      finishMessage: _finishMessageFromOpenAIResponse(response),
      message: GenkitConverter.fromOpenAIResponse(response),
      raw: response.toJson(),
    );
  }

  String? _finishMessageFromOpenAIResponse(sdk.Response response) {
    if (response.status == sdk.ResponseStatus.failed) {
      return response.error?.message;
    }
    if (response.status == sdk.ResponseStatus.incomplete) {
      return response.incompleteDetails?.reason;
    }
    return null;
  }

  Future<ModelResponse> _handleNonStreamingWithReasoningSummaryFallback(
    sdk.OpenAIClient client,
    sdk.ChatCompletionCreateRequest request,
  ) async {
    try {
      return await _handleNonStreaming(client, request);
    } on sdk.ApiException catch (error) {
      if (!_shouldRetryWithoutReasoningSummary(
        error: error,
        request: request,
        receivedAnyChunk: false,
      )) {
        rethrow;
      }

      final fallbackRequest = request.copyWith(openRouterReasoning: null);
      return _handleNonStreaming(client, fallbackRequest);
    }
  }

  Future<ModelResponse> _handleNonStreaming(
    sdk.OpenAIClient client,
    sdk.ChatCompletionCreateRequest request,
  ) async {
    final response = await client.chat.completions.create(request);

    if (response.choices.isEmpty) {
      throw GenkitException('Model returned no choices.');
    }

    final choice = response.choices.first;
    final message = GenkitConverter.fromOpenAIAssistantMessage(choice.message);

    return ModelResponse(
      finishReason: GenkitConverter.mapFinishReason(choice.finishReason?.name),
      message: message,
      raw: response.toJson(),
    );
  }
}

sdk.ReasoningConfig? _buildResponsesReasoningConfig(OpenAIOptions options) {
  final effort = _mapReasoningEffort(options.reasoningEffort);
  final summary = _mapResponsesReasoningSummary(options.reasoningSummary);
  if (effort == null && summary == null) {
    return null;
  }
  return sdk.ReasoningConfig(effort: effort, summary: summary);
}

sdk.ReasoningEffort? _mapReasoningEffort(String? value) {
  final normalized = _normalizeReasoningEffort(value);
  return switch (normalized) {
    'low' => sdk.ReasoningEffort.low,
    'medium' => sdk.ReasoningEffort.medium,
    'high' => sdk.ReasoningEffort.high,
    'xhigh' => sdk.ReasoningEffort.xhigh,
    _ => null,
  };
}

String? _normalizeReasoningEffort(String? value) {
  final normalized = value?.trim().toLowerCase();
  if (normalized == null || normalized.isEmpty) {
    return null;
  }

  return normalized;
}

sdk.ReasoningSummary? _mapResponsesReasoningSummary(String? value) {
  final normalized = _normalizeReasoningSummary(value);
  return switch (normalized) {
    'auto' => sdk.ReasoningSummary.auto,
    'concise' => sdk.ReasoningSummary.concise,
    'detailed' => sdk.ReasoningSummary.detailed,
    'none' || null => null,
    _ => null,
  };
}

String? _normalizeReasoningSummary(String? value) {
  final normalized = value?.trim().toLowerCase();
  if (normalized == null || normalized.isEmpty) {
    return null;
  }
  return normalized;
}

void _validateReasoningEffort(String? value) {
  final normalized = _normalizeReasoningEffort(value);
  if (normalized == null) {
    return;
  }
  if (_mapReasoningEffort(normalized) != null) {
    return;
  }

  throw GenkitException(
    'Unsupported reasoningEffort: $normalized',
    status: StatusCodes.INVALID_ARGUMENT,
  );
}

void _validateReasoningSummary(String? value) {
  final normalized = _normalizeReasoningSummary(value);
  if (normalized == null) {
    return;
  }
  if (normalized == 'auto' ||
      normalized == 'concise' ||
      normalized == 'detailed' ||
      normalized == 'none') {
    return;
  }
  throw GenkitException(
    'Unsupported reasoningSummary: $normalized',
    status: StatusCodes.INVALID_ARGUMENT,
  );
}

sdk.OpenRouterReasoning? _mapOpenRouterReasoning(String? reasoningSummary) {
  final normalized = _normalizeReasoningSummary(reasoningSummary);
  if (normalized == null) {
    return null;
  }
  return switch (normalized) {
    'none' => const sdk.OpenRouterReasoning(enabled: false, exclude: true),
    'auto' ||
    'concise' ||
    'detailed' => const sdk.OpenRouterReasoning(enabled: true, exclude: false),
    _ => null,
  };
}

bool _shouldRetryWithoutReasoningSummary({
  required sdk.ApiException error,
  required sdk.ChatCompletionCreateRequest request,
  required bool receivedAnyChunk,
}) {
  if (receivedAnyChunk || request.openRouterReasoning == null) {
    return false;
  }

  final isRetryableStatus =
      error.statusCode == 400 ||
      error.statusCode == 404 ||
      error.statusCode == 422;
  if (!isRetryableStatus) {
    return false;
  }

  final combinedText = <String>[
    error.message,
    error.param ?? '',
    error.code ?? '',
    error.type ?? '',
    error.body?.toString() ?? '',
  ].join(' ').toLowerCase();

  return combinedText.contains('reasoning');
}

@visibleForTesting
sdk.ReasoningEffort? mapReasoningEffortForTest(String? value) =>
    _mapReasoningEffort(value);

@visibleForTesting
void validateReasoningEffortForTest(String? value) =>
    _validateReasoningEffort(value);

@visibleForTesting
sdk.OpenRouterReasoning? mapOpenRouterReasoningForTest(
  String? reasoningSummary,
) => _mapOpenRouterReasoning(reasoningSummary);

@visibleForTesting
bool shouldRetryWithoutReasoningSummaryForTest({
  required sdk.ApiException error,
  required sdk.ChatCompletionCreateRequest request,
  required bool receivedAnyChunk,
}) => _shouldRetryWithoutReasoningSummary(
  error: error,
  request: request,
  receivedAnyChunk: receivedAnyChunk,
);

@visibleForTesting
sdk.Response rebuildResponseFromResponsesStreamForTest(
  sdk.Response response,
  Iterable<sdk.ResponseStreamEvent> events,
) {
  final state = _ResponsesStreamState();
  for (final event in events) {
    state.add(event);
  }
  return _rebuildResponseFromResponsesStream(response, state);
}

@visibleForTesting
List<TextPart> assistantTextPartsFromResponsesEventsForTest(
  Iterable<sdk.ResponseStreamEvent> events,
) {
  final state = _ResponsesStreamState();
  final parts = <TextPart>[];
  for (final event in events) {
    state.add(event);
    parts.addAll(_assistantTextPartsFromResponsesEvent(event, state));
  }
  return parts;
}

@visibleForTesting
List<String> assistantTextChunksFromResponsesEventsForTest(
  Iterable<sdk.ResponseStreamEvent> events,
) {
  return assistantTextPartsFromResponsesEventsForTest(
    events,
  ).where((part) => part.text.isNotEmpty).map((part) => part.text).toList();
}

List<TextPart> _assistantTextPartsFromResponsesEvent(
  sdk.ResponseStreamEvent event,
  _ResponsesStreamState state,
) {
  final chunks = switch (event) {
    sdk.OutputTextDeltaEvent() => state.captureOutputTextDelta(event),
    sdk.OutputTextDoneEvent() => state.captureOutputTextDone(event),
    sdk.ContentPartDoneEvent() => state.captureContentPartDone(event),
    sdk.OutputItemDoneEvent() => state.captureOutputItemDone(event),
    _ => const <_StreamedAssistantTextChunk>[],
  };
  return chunks.map(_textPartForStreamChunk).toList(growable: false);
}

TextPart _textPartForStreamChunk(_StreamedAssistantTextChunk chunk) {
  return TextPart(
    text: chunk.text,
    metadata: <String, dynamic>{
      'assistantMessageId': chunk.itemId,
      if (chunk.phase != null) 'assistantMessagePhase': chunk.phase!.toJson(),
      if (chunk.isBoundaryCompleted) 'assistantMessageBoundary': 'completed',
    },
  );
}

sdk.Response _rebuildResponseFromResponsesStream(
  sdk.Response response,
  _ResponsesStreamState state,
) {
  if (!state.hasRecoveredOutput) {
    return response;
  }
  final mergedOutput = state.mergeWithResponse(response);
  if (_sameOutputAsResponse(response.output, mergedOutput)) {
    return response;
  }
  return sdk.Response.fromJson(<String, dynamic>{
    ...response.toJson(),
    'output': mergedOutput.map((item) => item.toJson()).toList(growable: false),
  });
}

bool _sameOutputAsResponse(
  List<sdk.OutputItem> current,
  List<sdk.OutputItem> next,
) {
  if (identical(current, next)) {
    return true;
  }
  if (current.length != next.length) {
    return false;
  }
  for (var index = 0; index < current.length; index += 1) {
    if (current[index] != next[index]) {
      return false;
    }
  }
  return true;
}

final class _ResponsesStreamState {
  final Map<int, sdk.OutputItem> _addedItems = <int, sdk.OutputItem>{};
  final Map<int, sdk.OutputItem> _completedItems = <int, sdk.OutputItem>{};
  final Map<int, sdk.FunctionCallArgumentsDoneEvent> _functionCalls =
      <int, sdk.FunctionCallArgumentsDoneEvent>{};
  final Set<String> _emittedTextPartKeys = <String>{};

  bool get hasRecoveredOutput =>
      _completedItems.isNotEmpty || _functionCalls.isNotEmpty;

  void add(sdk.ResponseStreamEvent event) {
    switch (event) {
      case sdk.OutputItemAddedEvent(:final outputIndex, :final item):
        _addedItems[outputIndex] = item;
      case sdk.OutputItemDoneEvent(:final outputIndex, :final item):
        _completedItems[outputIndex] = item;
      case sdk.FunctionCallArgumentsDoneEvent(:final outputIndex):
        _functionCalls[outputIndex] = event;
      default:
        break;
    }
  }

  List<_StreamedAssistantTextChunk> captureOutputTextDelta(
    sdk.OutputTextDeltaEvent event,
  ) {
    final delta = event.delta;
    if (delta.isEmpty) {
      return const <_StreamedAssistantTextChunk>[];
    }
    _emittedTextPartKeys.add(
      _textPartKey(
        itemId: event.itemId,
        outputIndex: event.outputIndex,
        contentIndex: event.contentIndex,
      ),
    );
    return <_StreamedAssistantTextChunk>[
      _buildChunk(
        itemId: event.itemId,
        outputIndex: event.outputIndex,
        text: delta,
      ),
    ];
  }

  List<_StreamedAssistantTextChunk> captureOutputTextDone(
    sdk.OutputTextDoneEvent event,
  ) {
    final text = event.text;
    if (text.isEmpty) {
      return const <_StreamedAssistantTextChunk>[];
    }
    final key = _textPartKey(
      itemId: event.itemId,
      outputIndex: event.outputIndex,
      contentIndex: event.contentIndex,
    );
    if (!_emittedTextPartKeys.add(key)) {
      return const <_StreamedAssistantTextChunk>[];
    }
    return <_StreamedAssistantTextChunk>[
      _buildChunk(
        itemId: event.itemId,
        outputIndex: event.outputIndex,
        text: text,
      ),
    ];
  }

  List<_StreamedAssistantTextChunk> captureContentPartDone(
    sdk.ContentPartDoneEvent event,
  ) {
    final part = event.part;
    if (part is! sdk.OutputTextContent || part.text.isEmpty) {
      return const <_StreamedAssistantTextChunk>[];
    }
    final key = _textPartKey(
      itemId: event.itemId,
      outputIndex: event.outputIndex,
      contentIndex: event.contentIndex,
    );
    if (!_emittedTextPartKeys.add(key)) {
      return const <_StreamedAssistantTextChunk>[];
    }
    return <_StreamedAssistantTextChunk>[
      _buildChunk(
        itemId: event.itemId,
        outputIndex: event.outputIndex,
        text: part.text,
      ),
    ];
  }

  List<_StreamedAssistantTextChunk> captureOutputItemDone(
    sdk.OutputItemDoneEvent event,
  ) {
    final item = event.item;
    if (item is! sdk.MessageOutputItem) {
      return const <_StreamedAssistantTextChunk>[];
    }
    final chunks = <_StreamedAssistantTextChunk>[];
    for (final indexedContent in item.content.indexed) {
      final content = indexedContent.$2;
      if (content is! sdk.OutputTextContent || content.text.isEmpty) {
        continue;
      }
      final key = _textPartKey(
        itemId: item.id,
        outputIndex: event.outputIndex,
        contentIndex: indexedContent.$1,
      );
      if (!_emittedTextPartKeys.add(key)) {
        continue;
      }
      chunks.add(
        _StreamedAssistantTextChunk(
          text: content.text,
          itemId: item.id,
          phase: item.phase,
        ),
      );
    }
    chunks.add(
      _StreamedAssistantTextChunk(
        text: '',
        itemId: item.id,
        phase: item.phase,
        isBoundaryCompleted: true,
      ),
    );
    return chunks;
  }

  _StreamedAssistantTextChunk _buildChunk({
    required String? itemId,
    required int outputIndex,
    required String text,
  }) {
    return _StreamedAssistantTextChunk(
      text: text,
      itemId: itemId ?? 'output_$outputIndex',
      phase: _messagePhaseFor(itemId: itemId, outputIndex: outputIndex),
    );
  }

  sdk.MessagePhase? _messagePhaseFor({
    required String? itemId,
    required int outputIndex,
  }) {
    final completed = _completedItems[outputIndex];
    if (completed case sdk.MessageOutputItem(:final phase)) {
      return phase;
    }
    final added = _addedItems[outputIndex];
    if (added case sdk.MessageOutputItem(:final phase)) {
      return phase;
    }
    return null;
  }

  List<sdk.OutputItem> mergeWithResponse(sdk.Response response) {
    final merged = <int, sdk.OutputItem>{};
    for (final indexed in response.output.indexed) {
      merged[indexed.$1] = indexed.$2;
    }
    for (final entry in _completedItems.entries) {
      merged[entry.key] = entry.value;
    }
    for (final entry in _functionCalls.entries) {
      if (merged.containsKey(entry.key)) {
        continue;
      }
      final synthesized = _synthesizedFunctionCallFor(entry.key, entry.value);
      if (synthesized != null) {
        merged[entry.key] = synthesized;
      }
    }
    final orderedKeys = merged.keys.toList()..sort();
    return orderedKeys.map((key) => merged[key]!).toList(growable: false);
  }

  sdk.FunctionCallOutputItemResponse? _synthesizedFunctionCallFor(
    int outputIndex,
    sdk.FunctionCallArgumentsDoneEvent doneEvent,
  ) {
    final addedItem = _addedItems[outputIndex];
    if (addedItem is sdk.FunctionCallOutputItemResponse) {
      return sdk.FunctionCallOutputItemResponse(
        id: addedItem.id,
        callId: addedItem.callId,
        name: addedItem.name,
        arguments: doneEvent.arguments,
        status: addedItem.status,
        namespace: addedItem.namespace,
      );
    }
    final name = doneEvent.name?.trim();
    if (name == null || name.isEmpty) {
      return null;
    }
    final itemId = doneEvent.itemId?.trim();
    return sdk.FunctionCallOutputItemResponse(
      id: itemId?.isNotEmpty == true ? itemId! : 'fc_$outputIndex',
      callId: itemId?.isNotEmpty == true ? itemId! : 'call_$outputIndex',
      name: name,
      arguments: doneEvent.arguments,
    );
  }

  String _textPartKey({
    required String? itemId,
    required int outputIndex,
    required int contentIndex,
  }) {
    return '${itemId ?? 'output_$outputIndex'}:$contentIndex';
  }
}

final class _StreamedAssistantTextChunk {
  const _StreamedAssistantTextChunk({
    required this.text,
    required this.itemId,
    this.phase,
    this.isBoundaryCompleted = false,
  });

  final String text;
  final String itemId;
  final sdk.MessagePhase? phase;
  final bool isBoundaryCompleted;
}

@visibleForTesting
bool shouldFallbackStreamingToNonStreamingForTest({
  required Object error,
  required bool receivedAnyChunk,
}) => _shouldFallbackStreamingToNonStreaming(
  error: error,
  receivedAnyChunk: receivedAnyChunk,
);

bool _shouldFallbackStreamingToNonStreaming({
  required Object error,
  required bool receivedAnyChunk,
}) {
  if (receivedAnyChunk) {
    return false;
  }
  if (error is http.ClientException) {
    return true;
  }
  return error.toString().toLowerCase().contains(
    'stream finished without a final result chunk',
  );
}

final class _ResolvedClientConfig {
  final String apiKey;
  final String? baseUrl;
  final Map<String, String>? headers;

  const _ResolvedClientConfig({
    required this.apiKey,
    required this.baseUrl,
    required this.headers,
  });
}
