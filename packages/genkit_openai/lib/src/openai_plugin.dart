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

import 'dart:convert';

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
          sdk.ChatCompletionCreateRequest buildChatCompletionsRequest() {
            return sdk.ChatCompletionCreateRequest(
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
          }

          switch (wireApi) {
            case OpenAIWireApi.responses:
              final request = sdk.CreateResponseRequest(
                model: options.version ?? modelName,
                instructions: GenkitConverter.toOpenAIResponseInstructions(
                  modelRequest.messages,
                ),
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
                var receivedAnyChunk = false;
                try {
                  return await _handleResponsesStreaming(
                    client,
                    request,
                    ctx,
                    onChunkReceived: () => receivedAnyChunk = true,
                  );
                } catch (error) {
                  if (!_shouldFallbackResponsesToChatCompletions(
                    error: error,
                    receivedAnyChunk: receivedAnyChunk,
                  )) {
                    rethrow;
                  }
                  return _handleNonStreamingWithReasoningSummaryFallback(
                    client,
                    buildChatCompletionsRequest(),
                  );
                }
              }
              try {
                return await _handleResponsesNonStreaming(client, request);
              } catch (error) {
                if (!_shouldFallbackResponsesToChatCompletions(
                  error: error,
                  receivedAnyChunk: false,
                )) {
                  rethrow;
                }
                return _handleNonStreamingWithReasoningSummaryFallback(
                  client,
                  buildChatCompletionsRequest(),
                );
              }
            case OpenAIWireApi.chatCompletions:
              final request = buildChatCompletionsRequest();
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
    ctx, {
    void Function()? onChunkReceived,
  }) async {
    sdk.Response? finalResponse;
    final seenSummarySections = <String>{};
    final streamState = _ResponsesStreamState();
    var receivedAnyChunk = false;

    void sendChunk(ModelResponseChunk chunk) {
      receivedAnyChunk = true;
      onChunkReceived?.call();
      ctx.sendChunk(chunk);
    }

    try {
      await for (final event in _createResponsesStream(client, request)) {
        streamState.add(event);
        switch (event) {
          case sdk.OutputTextDeltaEvent():
            final parts = _assistantTextPartsFromResponsesEvent(
              event,
              streamState,
            );
            if (parts.isNotEmpty) {
              sendChunk(ModelResponseChunk(index: 0, content: parts));
            }
          case sdk.OutputTextDoneEvent():
            final parts = _assistantTextPartsFromResponsesEvent(
              event,
              streamState,
            );
            if (parts.isNotEmpty) {
              sendChunk(ModelResponseChunk(index: 0, content: parts));
            }
          case sdk.ContentPartDoneEvent():
            final parts = _assistantTextPartsFromResponsesEvent(
              event,
              streamState,
            );
            if (parts.isNotEmpty) {
              sendChunk(ModelResponseChunk(index: 0, content: parts));
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
            sendChunk(
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
              sendChunk(ModelResponseChunk(index: 0, content: parts));
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
      if (_shouldFallbackStreamingToNonStreaming(
        error: e,
        receivedAnyChunk: receivedAnyChunk,
      )) {
        return _handleResponsesNonStreaming(client, request);
      }
      throw GenkitException(
        'Error in responses streaming: $e',
        underlyingException: e,
        stackTrace: stackTrace,
      );
    }

    final response = _finalResponseForResponsesStream(
      finalResponse,
      streamState,
    );
    if (response == null) {
      final error = GenkitException(
        'Responses stream ended without a final response.',
      );
      if (_shouldFallbackStreamingToNonStreaming(
        error: error,
        receivedAnyChunk: receivedAnyChunk,
      )) {
        return _handleResponsesNonStreaming(client, request);
      }
      throw error;
    }
    return _modelResponseFromOpenAIResponse(
      _rebuildResponseFromResponsesStream(response, streamState),
    );
  }

  Stream<sdk.ResponseStreamEvent> _createResponsesStream(
    sdk.OpenAIClient client,
    sdk.CreateResponseRequest request,
  ) {
    final requestBody = request.toJson();
    requestBody['stream'] = true;
    return client.responses
        .streamSseEvents(endpoint: '/responses', body: requestBody)
        .map(_responseStreamEventFromJson);
  }

  /// Handle non-streaming response
  Future<ModelResponse> _handleResponsesNonStreaming(
    sdk.OpenAIClient client,
    sdk.CreateResponseRequest request,
  ) async {
    final response = await _createResponse(client, request);
    return _modelResponseFromOpenAIResponse(response);
  }

  Future<sdk.Response> _createResponse(
    sdk.OpenAIClient client,
    sdk.CreateResponseRequest request,
  ) async {
    final url = _buildOpenAIUrl(client.config.baseUrl, '/responses');
    final httpRequest = http.Request('POST', url)
      ..headers.addAll(_buildOpenAIHeaders(client.config))
      ..body = jsonEncode(request.toJson());
    final httpResponse = await client.interceptorChain.execute(httpRequest);
    if (httpResponse.statusCode >= 400) {
      _throwOpenAIHttpError(httpResponse);
    }
    final decoded = jsonDecode(httpResponse.body) as Map<String, dynamic>;
    return sdk.Response.fromJson(
      _normalizeOpenAIResponseJson(
        decoded,
        fallbackStatus: 'completed',
        fallbackId: _nonEmptyString(decoded['id']) ?? 'resp_non_streaming',
      ),
    );
  }

  Never _throwOpenAIHttpError(http.Response response) {
    final decoded = _tryDecodeJsonObject(response.body);
    final error = _stringKeyedMap(decoded?['error']);
    final fallbackMessage = response.body.trim().isNotEmpty
        ? response.body.trim()
        : 'Provider returned ${response.statusCode} ${response.reasonPhrase ?? ''}'
              .trim();
    throw sdk.ApiException(
      message: _nonEmptyString(error?['message']) ?? fallbackMessage,
      statusCode: response.statusCode,
      type: _nonEmptyString(error?['type']),
      code: _nonEmptyString(error?['code']),
      param: _nonEmptyString(error?['param']),
      requestId: response.headers['x-request-id'],
      body: decoded,
    );
  }

  ModelResponse _modelResponseFromOpenAIResponse(sdk.Response response) {
    if (response.status == sdk.ResponseStatus.failed) {
      throw _responsesFailureException(response);
    }
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

  GenkitException _responsesFailureException(sdk.Response response) {
    final error = response.error;
    final message = error?.message ?? 'Response ${response.id} failed.';
    return GenkitException(
      'OpenAI Responses API error: $message',
      status: StatusCodes.INTERNAL,
      details: error?.code ?? error?.type,
    );
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
sdk.ResponseStreamEvent responseStreamEventFromJsonForTest(
  Map<String, dynamic> json,
) => _responseStreamEventFromJson(json);

@visibleForTesting
ModelResponse modelResponseFromOpenAIResponseForTest(sdk.Response response) {
  return OpenAIPlugin()._modelResponseFromOpenAIResponse(response);
}

@visibleForTesting
ModelResponse modelResponseFromResponsesEventsForTest(
  Iterable<sdk.ResponseStreamEvent> events,
) {
  final state = _ResponsesStreamState();
  sdk.Response? finalResponse;
  for (final event in events) {
    state.add(event);
    switch (event) {
      case sdk.ResponseCompletedEvent(:final response):
      case sdk.ResponseIncompleteEvent(:final response):
      case sdk.ResponseFailedEvent(:final response):
        finalResponse = response;
      default:
        break;
    }
  }
  final response = _finalResponseForResponsesStream(finalResponse, state);
  if (response == null) {
    throw GenkitException('Responses stream ended without a final response.');
  }
  return OpenAIPlugin()._modelResponseFromOpenAIResponse(
    _rebuildResponseFromResponsesStream(response, state),
  );
}

sdk.ResponseStreamEvent _responseStreamEventFromJson(
  Map<String, dynamic> json,
) {
  return sdk.ResponseStreamEvent.fromJson(
    _normalizeResponseStreamEventJson(json),
  );
}

Map<String, dynamic> _normalizeResponseStreamEventJson(
  Map<String, dynamic> json,
) {
  final type = json['type'];
  if (type == 'response.output_item.added' ||
      type == 'response.output_item.done') {
    final item = _stringKeyedMap(json['item']);
    if (item == null) {
      return json;
    }
    final normalizedItem = _normalizeResponseOutputItemJson(
      item,
      fallbackIndex: _intValue(json['output_index']) ?? 0,
    );
    if (normalizedItem == null) {
      return json;
    }
    return <String, dynamic>{...json, 'item': normalizedItem};
  }

  final fallbackStatus = _statusForResponseStreamEvent(type);
  if (fallbackStatus == null) {
    return json;
  }
  final response = _stringKeyedMap(json['response']) ?? <String, dynamic>{};
  final fallbackId =
      _nonEmptyString(response['id']) ??
      _nonEmptyString(json['response_id']) ??
      _nonEmptyString(json['id']) ??
      'resp_stream_${_intValue(json['sequence_number']) ?? type}';
  return <String, dynamic>{
    ...json,
    'response': _normalizeOpenAIResponseJson(
      response,
      fallbackStatus: fallbackStatus,
      fallbackId: fallbackId,
    ),
  };
}

@visibleForTesting
Map<String, dynamic> normalizeOpenAIResponseJsonForTest(
  Map<String, dynamic> json, {
  String fallbackStatus = 'completed',
  String fallbackId = 'resp_test',
}) => _normalizeOpenAIResponseJson(
  json,
  fallbackStatus: fallbackStatus,
  fallbackId: fallbackId,
);

Uri _buildOpenAIUrl(String baseUrl, String path) {
  final baseUri = Uri.parse(baseUrl);
  final basePath = baseUri.path.endsWith('/')
      ? baseUri.path.substring(0, baseUri.path.length - 1)
      : baseUri.path;
  final normalizedPath = path.startsWith('/') ? path : '/$path';
  return baseUri.replace(
    path: '$basePath$normalizedPath',
    queryParameters: baseUri.queryParameters.isEmpty
        ? null
        : baseUri.queryParameters,
  );
}

Map<String, String> _buildOpenAIHeaders(sdk.OpenAIConfig config) {
  final headers = <String, String>{
    'Content-Type': 'application/json',
    ...config.defaultHeaders,
  };
  final apiVersion = _nonEmptyString(config.apiVersion);
  if (apiVersion != null) {
    headers['OpenAI-Version'] = apiVersion;
  }
  final organization = _nonEmptyString(config.organization);
  if (organization != null) {
    headers['OpenAI-Organization'] = organization;
  }
  final project = _nonEmptyString(config.project);
  if (project != null) {
    headers['OpenAI-Project'] = project;
  }
  return headers;
}

String? _statusForResponseStreamEvent(Object? type) {
  return switch (type) {
    'response.created' => 'in_progress',
    'response.queued' => 'queued',
    'response.in_progress' => 'in_progress',
    'response.completed' => 'completed',
    'response.incomplete' => 'incomplete',
    'response.failed' => 'failed',
    _ => null,
  };
}

Map<String, dynamic> _normalizeOpenAIResponseJson(
  Map<String, dynamic> json, {
  required String fallbackStatus,
  required String fallbackId,
}) {
  final normalized = <String, dynamic>{...json};
  final error = _stringKeyedMap(normalized['error']);
  normalized['id'] = _nonEmptyString(normalized['id']) ?? fallbackId;
  normalized['object'] = _nonEmptyString(normalized['object']) ?? 'response';
  normalized['created_at'] = _intValue(normalized['created_at']) ?? 0;
  normalized['status'] =
      _nonEmptyString(normalized['status']) ??
      (error == null ? fallbackStatus : 'failed');
  normalized['output'] = _normalizeResponseOutputJson(normalized['output']);

  final metadata = _stringKeyedMap(normalized['metadata']);
  if (metadata == null) {
    normalized.remove('metadata');
  } else {
    normalized['metadata'] = <String, String>{
      for (final entry in metadata.entries)
        if (entry.value != null) entry.key: entry.value.toString(),
    };
  }

  if (error != null) {
    normalized['error'] = _normalizeResponseErrorJson(error);
  } else if (normalized['status'] == 'failed') {
    normalized['error'] = const <String, dynamic>{
      'type': 'error',
      'message': 'OpenAI Responses API request failed.',
    };
  }

  return normalized;
}

Map<String, dynamic> _normalizeResponseErrorJson(Map<String, dynamic> json) {
  final normalized = <String, dynamic>{
    'type': _nonEmptyString(json['type']) ?? 'error',
    'message':
        _nonEmptyString(json['message']) ??
        'OpenAI Responses API request failed.',
  };
  final code = _nonEmptyString(json['code']);
  if (code != null) {
    normalized['code'] = code;
  }
  final param = _nonEmptyString(json['param']);
  if (param != null) {
    normalized['param'] = param;
  }
  return normalized;
}

List<Object?> _normalizeResponseOutputJson(Object? output) {
  if (output is! List) {
    return const <Object?>[];
  }
  final normalized = <Object?>[];
  for (var i = 0; i < output.length; i += 1) {
    final item = _stringKeyedMap(output[i]);
    if (item == null) {
      continue;
    }
    final normalizedItem = _normalizeResponseOutputItemJson(
      item,
      fallbackIndex: i,
    );
    if (normalizedItem != null) {
      normalized.add(normalizedItem);
    }
  }
  return normalized;
}

Map<String, dynamic>? _normalizeResponseOutputItemJson(
  Map<String, dynamic> item, {
  required int fallbackIndex,
}) {
  final type = _nonEmptyString(item['type']);
  if (type == null) {
    return null;
  }
  final normalized = <String, dynamic>{...item, 'type': type};
  normalized['id'] =
      _nonEmptyString(normalized['id']) ?? '${type}_$fallbackIndex';

  switch (type) {
    case 'message':
      normalized['role'] = _nonEmptyString(normalized['role']) ?? 'assistant';
      normalized['content'] = _normalizeOutputContentJson(
        normalized['content'],
      );
      if (_nonEmptyString(normalized['status']) == null) {
        normalized.remove('status');
      }
      if (_nonEmptyString(normalized['phase']) == null) {
        normalized.remove('phase');
      }
    case 'function_call':
      normalized['call_id'] =
          _nonEmptyString(normalized['call_id']) ??
          _nonEmptyString(normalized['id']) ??
          'call_$fallbackIndex';
      normalized['name'] =
          _nonEmptyString(normalized['name']) ?? 'unknown_function';
      normalized['arguments'] = _jsonString(
        normalized['arguments'],
        defaultValue: '{}',
      );
    case 'reasoning':
      normalized['summary'] = _normalizeReasoningSummaryJson(
        normalized['summary'],
      );
      if (normalized['content'] is! List) {
        normalized.remove('content');
      }
    default:
      break;
  }

  return normalized;
}

List<Object?> _normalizeOutputContentJson(Object? content) {
  if (content is! List) {
    return const <Object?>[];
  }
  final normalized = <Object?>[];
  for (final entry in content) {
    final item = _stringKeyedMap(entry);
    if (item == null) {
      continue;
    }
    final type = _nonEmptyString(item['type']);
    if (type == null) {
      continue;
    }
    final normalizedItem = <String, dynamic>{...item, 'type': type};
    switch (type) {
      case 'output_text':
      case 'reasoning_text':
      case 'summary_text':
      case 'input_text':
        normalizedItem['text'] = _nonEmptyString(normalizedItem['text']) ?? '';
        if (normalizedItem['annotations'] is! List) {
          normalizedItem.remove('annotations');
        }
        if (normalizedItem['logprobs'] is! List) {
          normalizedItem.remove('logprobs');
        }
      case 'refusal':
        normalizedItem['refusal'] =
            _nonEmptyString(normalizedItem['refusal']) ?? '';
      default:
        break;
    }
    normalized.add(normalizedItem);
  }
  return normalized;
}

List<Object?> _normalizeReasoningSummaryJson(Object? summary) {
  if (summary is! List) {
    return const <Object?>[];
  }
  final normalized = <Object?>[];
  for (final entry in summary) {
    final item = _stringKeyedMap(entry);
    if (item == null) {
      continue;
    }
    normalized.add(<String, dynamic>{
      ...item,
      'type': _nonEmptyString(item['type']) ?? 'summary_text',
      'text': _nonEmptyString(item['text']) ?? '',
    });
  }
  return normalized;
}

Map<String, dynamic>? _stringKeyedMap(Object? value) {
  if (value is! Map) {
    return null;
  }
  return <String, dynamic>{
    for (final entry in value.entries)
      if (entry.key is String) entry.key as String: entry.value,
  };
}

Map<String, dynamic>? _tryDecodeJsonObject(String value) {
  try {
    return _stringKeyedMap(jsonDecode(value));
  } catch (_) {
    return null;
  }
}

String? _nonEmptyString(Object? value) {
  if (value is! String) {
    return null;
  }
  final trimmed = value.trim();
  return trimmed.isEmpty ? null : trimmed;
}

int? _intValue(Object? value) {
  return switch (value) {
    int() => value,
    num() => value.toInt(),
    _ => null,
  };
}

String _jsonString(Object? value, {String defaultValue = ''}) {
  if (value is String) {
    return value.isEmpty ? defaultValue : value;
  }
  if (value == null) {
    return defaultValue;
  }
  try {
    return jsonEncode(value);
  } catch (_) {
    return value.toString();
  }
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

sdk.Response? _finalResponseForResponsesStream(
  sdk.Response? finalResponse,
  _ResponsesStreamState state,
) {
  if (finalResponse != null) {
    return finalResponse;
  }
  return state.synthesizeCompletedResponse();
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
  sdk.Response? _latestLifecycleResponse;
  final Map<int, sdk.OutputItem> _addedItems = <int, sdk.OutputItem>{};
  final Map<int, sdk.OutputItem> _completedItems = <int, sdk.OutputItem>{};
  final Map<int, sdk.FunctionCallArgumentsDoneEvent> _functionCalls =
      <int, sdk.FunctionCallArgumentsDoneEvent>{};
  final Map<int, Map<int, _StreamedTextAccumulator>> _textContent =
      <int, Map<int, _StreamedTextAccumulator>>{};
  final Set<String> _emittedTextPartKeys = <String>{};

  bool get hasRecoveredOutput =>
      _completedItems.isNotEmpty ||
      _functionCalls.isNotEmpty ||
      _textContent.values.any(
        (parts) => parts.values.any((part) => part.text.isNotEmpty),
      );

  void add(sdk.ResponseStreamEvent event) {
    switch (event) {
      case sdk.ResponseCreatedEvent(:final response):
      case sdk.ResponseQueuedEvent(:final response):
      case sdk.ResponseInProgressEvent(:final response):
        _latestLifecycleResponse = response;
      case sdk.OutputItemAddedEvent(:final outputIndex, :final item):
        _addedItems[outputIndex] = item;
      case sdk.OutputItemDoneEvent(:final outputIndex, :final item):
        _completedItems[outputIndex] = item;
      case sdk.FunctionCallArgumentsDoneEvent(:final outputIndex):
        _functionCalls[outputIndex] = event;
      case sdk.OutputTextDeltaEvent(
        :final outputIndex,
        :final contentIndex,
        :final itemId,
        :final delta,
      ):
        if (delta.isNotEmpty) {
          _recordText(
            outputIndex: outputIndex,
            contentIndex: contentIndex,
            itemId: itemId,
            text: delta,
            complete: false,
          );
        }
      case sdk.OutputTextDoneEvent(
        :final outputIndex,
        :final contentIndex,
        :final itemId,
        :final text,
      ):
        if (text.isNotEmpty) {
          _recordText(
            outputIndex: outputIndex,
            contentIndex: contentIndex,
            itemId: itemId,
            text: text,
            complete: true,
          );
        }
      case sdk.ContentPartDoneEvent(
        :final outputIndex,
        :final contentIndex,
        :final itemId,
        :final part,
      ):
        if (part is sdk.OutputTextContent && part.text.isNotEmpty) {
          _recordText(
            outputIndex: outputIndex,
            contentIndex: contentIndex,
            itemId: itemId,
            text: part.text,
            complete: true,
          );
        }
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
    for (final entry in _textContent.entries) {
      if (merged.containsKey(entry.key)) {
        continue;
      }
      final synthesized = _synthesizedMessageFor(entry.key, entry.value);
      if (synthesized != null) {
        merged[entry.key] = synthesized;
      }
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

  sdk.Response? synthesizeCompletedResponse() {
    if (!hasRecoveredOutput) {
      return null;
    }
    final base = _latestLifecycleResponse;
    final synthesized = sdk.Response.fromJson(<String, dynamic>{
      if (base != null) ...base.toJson(),
      'id': base?.id ?? 'resp_stream_synthesized',
      'object': base?.object ?? 'response',
      'created_at': base?.createdAt ?? 0,
      'status': 'completed',
      'output': const <Object?>[],
    });
    return _rebuildResponseFromResponsesStream(synthesized, this);
  }

  void _recordText({
    required int outputIndex,
    required int contentIndex,
    required String? itemId,
    required String text,
    required bool complete,
  }) {
    final parts = _textContent.putIfAbsent(
      outputIndex,
      () => <int, _StreamedTextAccumulator>{},
    );
    final accumulator = parts.putIfAbsent(
      contentIndex,
      _StreamedTextAccumulator.new,
    );
    accumulator.record(itemId: itemId, text: text, complete: complete);
  }

  sdk.MessageOutputItem? _synthesizedMessageFor(
    int outputIndex,
    Map<int, _StreamedTextAccumulator> parts,
  ) {
    final contentEntries =
        parts.entries
            .where((entry) => entry.value.text.isNotEmpty)
            .toList(growable: false)
          ..sort((a, b) => a.key.compareTo(b.key));
    if (contentEntries.isEmpty) {
      return null;
    }
    final addedItem = _addedItems[outputIndex];
    final addedMessage = addedItem is sdk.MessageOutputItem ? addedItem : null;
    String? streamedItemId;
    for (final entry in contentEntries) {
      final itemId = entry.value.itemId;
      if (itemId != null && itemId.isNotEmpty) {
        streamedItemId = itemId;
        break;
      }
    }
    final itemId = streamedItemId ?? addedMessage?.id ?? 'msg_$outputIndex';
    return sdk.MessageOutputItem(
      id: itemId,
      role: addedMessage?.role ?? sdk.MessageRole.assistant,
      phase: addedMessage?.phase,
      status: addedMessage?.status,
      content: contentEntries
          .map((entry) => sdk.OutputTextContent(text: entry.value.text))
          .toList(growable: false),
    );
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

final class _StreamedTextAccumulator {
  final StringBuffer _deltaText = StringBuffer();
  String? _doneText;
  String? itemId;

  String get text => _doneText ?? _deltaText.toString();

  void record({
    required String? itemId,
    required String text,
    required bool complete,
  }) {
    final normalizedItemId = itemId?.trim();
    if (normalizedItemId != null && normalizedItemId.isNotEmpty) {
      this.itemId = normalizedItemId;
    }
    if (complete) {
      _doneText = text;
    } else {
      _deltaText.write(text);
    }
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

@visibleForTesting
bool shouldFallbackResponsesToChatCompletionsForTest({
  required Object error,
  required bool receivedAnyChunk,
}) => _shouldFallbackResponsesToChatCompletions(
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
  final text = error.toString().toLowerCase();
  return text.contains('stream finished without a final result chunk') ||
      text.contains('responses stream ended without a final response');
}

bool _shouldFallbackResponsesToChatCompletions({
  required Object error,
  required bool receivedAnyChunk,
}) {
  if (receivedAnyChunk) {
    return false;
  }
  if (error is sdk.ApiException) {
    return error.statusCode == 500 ||
        error.statusCode == 502 ||
        error.statusCode == 503 ||
        error.statusCode == 504;
  }
  final text = error.toString().toLowerCase();
  return text.contains('bad gateway') ||
      text.contains('service unavailable') ||
      text.contains('gateway timeout') ||
      text.contains('provider returned 500') ||
      text.contains('provider returned 502') ||
      text.contains('provider returned 503') ||
      text.contains('provider returned 504');
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
