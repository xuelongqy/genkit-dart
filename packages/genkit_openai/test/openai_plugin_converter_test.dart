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

import 'package:genkit/genkit.dart' hide Tool;
import 'package:genkit_openai/genkit_openai.dart';
import 'package:genkit_openai/src/openai_plugin.dart'
    show
        assistantTextChunksFromResponsesEventsForTest,
        assistantTextPartsFromResponsesEventsForTest,
        mapOpenRouterReasoningForTest,
        mapReasoningEffortForTest,
        rebuildResponseFromResponsesStreamForTest,
        shouldFallbackStreamingToNonStreamingForTest,
        shouldRetryWithoutReasoningSummaryForTest,
        validateReasoningEffortForTest;
import 'package:http/http.dart' as http;
import 'package:openai_dart/openai_dart.dart'
    show
        AssistantMessage,
        ChatMessage,
        ContentPart,
        FunctionCall,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage;
import 'package:openai_dart/openai_dart.dart' as sdk;
import 'package:test/test.dart';

void main() {
  group('OpenAIOptions', () {
    test('parses temperature', () {
      final options = OpenAIOptions.$schema.parse({'temperature': 0.7});
      expect(options.temperature, 0.7);
    });

    test('parses maxTokens', () {
      final options = OpenAIOptions.$schema.parse({'maxTokens': 100});
      expect(options.maxTokens, 100);
    });

    test('parses jsonMode', () {
      final options = OpenAIOptions.$schema.parse({'jsonMode': true});
      expect(options.jsonMode, true);
    });

    test('parses stop sequences', () {
      final options = OpenAIOptions.$schema.parse({
        'stop': ['stop1', 'stop2'],
      });
      expect(options.stop, ['stop1', 'stop2']);
    });

    test('creates default options', () {
      final options = OpenAIOptions();
      expect(options.temperature, isNull);
      expect(options.maxTokens, isNull);
    });

    test('stores supported reasoning effort values', () {
      final options = OpenAIOptions.$schema.parse({'reasoningEffort': 'xhigh'});
      expect(options.reasoningEffort, 'xhigh');
    });
  });

  group('mapReasoningEffortForTest', () {
    test('maps supported reasoning effort values to SDK enums', () {
      expect(mapReasoningEffortForTest('low'), sdk.ReasoningEffort.low);
      expect(mapReasoningEffortForTest('medium'), sdk.ReasoningEffort.medium);
      expect(mapReasoningEffortForTest('high'), sdk.ReasoningEffort.high);
      expect(mapReasoningEffortForTest('xhigh'), sdk.ReasoningEffort.xhigh);
    });

    test('returns null for unsupported reasoning effort values', () {
      expect(mapReasoningEffortForTest('minimal'), isNull);
      expect(mapReasoningEffortForTest('none'), isNull);
    });
  });

  group('validateReasoningEffortForTest', () {
    test('accepts supported reasoning effort values', () {
      expect(() => validateReasoningEffortForTest('low'), returnsNormally);
      expect(() => validateReasoningEffortForTest('medium'), returnsNormally);
      expect(() => validateReasoningEffortForTest('high'), returnsNormally);
      expect(() => validateReasoningEffortForTest('xhigh'), returnsNormally);
    });

    test('rejects unsupported reasoning effort values', () {
      expect(
        () => validateReasoningEffortForTest('minimal'),
        throwsA(isA<GenkitException>()),
      );
    });
  });

  group('mapOpenRouterReasoningForTest', () {
    test('maps supported reasoning summary modes for generic proxies', () {
      expect(
        mapOpenRouterReasoningForTest('auto'),
        const sdk.OpenRouterReasoning(enabled: true, exclude: false),
      );
      expect(
        mapOpenRouterReasoningForTest('concise'),
        const sdk.OpenRouterReasoning(enabled: true, exclude: false),
      );
      expect(
        mapOpenRouterReasoningForTest('detailed'),
        const sdk.OpenRouterReasoning(enabled: true, exclude: false),
      );
      expect(
        mapOpenRouterReasoningForTest('none'),
        const sdk.OpenRouterReasoning(enabled: false, exclude: true),
      );
    });
  });

  group('shouldRetryWithoutReasoningSummaryForTest', () {
    test(
      'retries when a proxy rejects the reasoning field before streaming',
      () {
        final shouldRetry = shouldRetryWithoutReasoningSummaryForTest(
          error: const sdk.ApiException(
            message: 'Unknown parameter: reasoning',
            statusCode: 400,
            param: 'reasoning',
          ),
          request: const sdk.ChatCompletionCreateRequest(
            model: 'gpt-5.4',
            messages: <sdk.ChatMessage>[],
            openRouterReasoning: sdk.OpenRouterReasoning(
              enabled: true,
              exclude: false,
            ),
          ),
          receivedAnyChunk: false,
        );

        expect(shouldRetry, isTrue);
      },
    );

    test('does not retry after any response chunk was already observed', () {
      final shouldRetry = shouldRetryWithoutReasoningSummaryForTest(
        error: const sdk.ApiException(
          message: 'Unknown parameter: reasoning',
          statusCode: 400,
          param: 'reasoning',
        ),
        request: const sdk.ChatCompletionCreateRequest(
          model: 'gpt-5.4',
          messages: <sdk.ChatMessage>[],
          openRouterReasoning: sdk.OpenRouterReasoning(
            enabled: true,
            exclude: false,
          ),
        ),
        receivedAnyChunk: true,
      );

      expect(shouldRetry, isFalse);
    });
  });

  group('streaming fallback', () {
    test(
      'falls back for transport-level client exceptions before any chunk',
      () {
        final shouldFallback = shouldFallbackStreamingToNonStreamingForTest(
          error: http.ClientException('Connection closed while receiving data'),
          receivedAnyChunk: false,
        );

        expect(shouldFallback, isTrue);
      },
    );

    test('falls back for truncated streaming errors by message', () {
      final shouldFallback = shouldFallbackStreamingToNonStreamingForTest(
        error: StateError('stream finished without a final result chunk'),
        receivedAnyChunk: false,
      );

      expect(shouldFallback, isTrue);
    });

    test('does not fall back after any streaming chunk was received', () {
      final shouldFallback = shouldFallbackStreamingToNonStreamingForTest(
        error: http.ClientException('Connection closed while receiving data'),
        receivedAnyChunk: true,
      );

      expect(shouldFallback, isFalse);
    });

    test('does not fall back for unrelated errors', () {
      final shouldFallback = shouldFallbackStreamingToNonStreamingForTest(
        error: ArgumentError('invalid request'),
        receivedAnyChunk: false,
      );

      expect(shouldFallback, isFalse);
    });
  });

  group('GenkitConverter.toOpenAIMessage', () {
    test('converts system message', () {
      final msg = Message(
        role: Role.system,
        content: [TextPart(text: 'You are helpful.')],
      );
      final result = GenkitConverter.toOpenAIMessage(msg, null);
      expect(result, isA<SystemMessage>());
      expect((result as SystemMessage).content, 'You are helpful.');
    });

    test('converts user message with text', () {
      final msg = Message(
        role: Role.user,
        content: [TextPart(text: 'Hello!')],
      );
      final result = GenkitConverter.toOpenAIMessage(msg, null);
      expect(result, isA<UserMessage>());
    });

    test('converts model message with tool calls', () {
      final msg = Message(
        role: Role.model,
        content: [
          TextPart(text: 'I will call a tool.'),
          ToolRequestPart(
            toolRequest: ToolRequest(
              ref: 'call_123',
              name: 'getWeather',
              input: {'location': 'Boston'},
            ),
          ),
        ],
      );
      final result = GenkitConverter.toOpenAIMessage(msg, null);
      expect(result, isA<AssistantMessage>());
      final assistantMsg = result as AssistantMessage;
      expect(assistantMsg.toolCalls, isNotNull);
      expect(assistantMsg.toolCalls!.length, 1);
    });

    test('converts tool message', () {
      final msg = Message(
        role: Role.tool,
        content: [
          ToolResponsePart(
            toolResponse: ToolResponse(
              ref: 'call_123',
              name: 'getWeather',
              output: {'temperature': 72},
            ),
          ),
        ],
      );
      final results = GenkitConverter.toOpenAIMessages([msg], null);
      expect(results.length, 1);
      expect(results[0], isA<ToolMessage>());
      final toolMsg = results[0] as ToolMessage;
      expect(toolMsg.toolCallId, 'call_123');
    });

    test('converts tool message with multiple responses', () {
      final msg = Message(
        role: Role.tool,
        content: [
          ToolResponsePart(
            toolResponse: ToolResponse(
              ref: 'call_123',
              name: 'getWeather',
              output: {'temperature': 72},
            ),
          ),
          ToolResponsePart(
            toolResponse: ToolResponse(
              ref: 'call_456',
              name: 'calculate',
              output: {'result': 42},
            ),
          ),
        ],
      );
      final results = GenkitConverter.toOpenAIMessages([msg], null);
      expect(results.length, 2);
      expect(results[0], isA<ToolMessage>());
      expect(results[1], isA<ToolMessage>());
      final toolMsg1 = results[0] as ToolMessage;
      final toolMsg2 = results[1] as ToolMessage;
      expect(toolMsg1.toolCallId, 'call_123');
      expect(toolMsg2.toolCallId, 'call_456');
    });

    test('throws on tool message with missing ref', () {
      final msg = Message(
        role: Role.tool,
        content: [
          ToolResponsePart(
            toolResponse: ToolResponse(
              name: 'getWeather',
              output: {'temperature': 72},
            ),
          ),
        ],
      );
      expect(
        () => GenkitConverter.toOpenAIMessages([msg], null),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('GenkitConverter.toOpenAIContentPart', () {
    test('converts text part', () {
      final part = TextPart(text: 'Hello');
      final result = GenkitConverter.toOpenAIContentPart(part, null);
      expect(result, isA<ContentPart>());
    });

    test('converts media part with URL', () {
      final part = MediaPart(
        media: Media(
          url: 'https://example.com/image.png',
          contentType: 'image/png',
        ),
      );
      final result = GenkitConverter.toOpenAIContentPart(part, 'high');
      expect(result, isA<ContentPart>());
    });

    test('converts media part with base64 data URI', () {
      final part = MediaPart(
        media: Media(
          url: 'data:image/png;base64,iVBORw0KGgoAAAANS',
          contentType: 'image/png',
        ),
      );
      final result = GenkitConverter.toOpenAIContentPart(part, null);
      expect(result, isA<ContentPart>());
    });

    test('converts deserialized Part with media data', () {
      final part = Part.fromJson({
        'media': {
          'url': 'https://example.com/document.pdf',
          'contentType': 'application/pdf',
        },
      });

      final result = GenkitConverter.toOpenAIContentPart(part, null);
      expect(result.toJson(), {
        'type': 'image_url',
        'image_url': {
          'url': 'https://example.com/document.pdf',
          'detail': 'auto',
        },
      });
    });

    test(
      'converts media part rehydrated from Message.content as base Part',
      () {
        final msg = Message(
          role: Role.user,
          content: [
            MediaPart(
              media: Media(
                url: 'data:image/png;base64,iVBORw0KGgoAAAANS',
                contentType: 'image/png',
              ),
            ),
          ],
        );

        final part = msg.content.single;
        expect(part, isA<Part>());
        expect(part, isNot(isA<MediaPart>()));
        expect(part.isMedia, isTrue);

        final result = GenkitConverter.toOpenAIContentPart(part, null);
        expect(result, isA<ContentPart>());
      },
    );

    test('converts audio media part with base64 data URI', () {
      final part = MediaPart(
        media: Media(
          url: 'data:audio/wav;base64,UklGRg==',
          contentType: 'audio/wav',
        ),
      );

      final result = GenkitConverter.toOpenAIContentPart(part, null);
      expect(result, isA<ContentPart>());
      expect(result.toJson()['type'], 'input_audio');
    });

    test('throws for audio media part with non-data URL', () {
      final part = MediaPart(
        media: Media(
          url: 'https://example.com/audio.wav',
          contentType: 'audio/wav',
        ),
      );

      expect(
        () => GenkitConverter.toOpenAIContentPart(part, null),
        throwsA(isA<UnimplementedError>()),
      );
    });
  });

  group('GenkitConverter.toOpenAITool', () {
    test('converts tool definition', () {
      final tool = ToolDefinition(
        name: 'getWeather',
        description: 'Get weather for a location',
        inputSchema: {
          'type': 'object',
          'properties': {
            'location': {'type': 'string'},
          },
        },
      );
      final result = GenkitConverter.toOpenAITool(tool);
      expect(result.function.name, 'getWeather');
      expect(result.function.description, 'Get weather for a location');
    });
  });

  group('GenkitConverter.toOpenAIResponseInput', () {
    test('converts mixed message history into responses input items', () {
      final input = GenkitConverter.toOpenAIResponseInput(<Message>[
        Message(
          role: Role.system,
          content: <Part>[TextPart(text: 'You are helpful.')],
        ),
        Message(
          role: Role.user,
          content: <Part>[TextPart(text: 'What is the weather?')],
        ),
        Message(
          role: Role.model,
          content: <Part>[
            TextPart(text: 'I will check.'),
            ToolRequestPart(
              toolRequest: ToolRequest(
                ref: 'call_123',
                name: 'getWeather',
                input: <String, Object?>{'location': 'Boston'},
              ),
            ),
          ],
        ),
        Message(
          role: Role.tool,
          content: <Part>[
            ToolResponsePart(
              toolResponse: ToolResponse(
                ref: 'call_123',
                name: 'getWeather',
                output: <String, Object?>{'temperature': 72},
              ),
            ),
          ],
        ),
      ], null);

      final json = input.toJson() as List<dynamic>;
      expect(json, hasLength(5));
      expect(json[0], containsPair('role', 'system'));
      expect(json[1], containsPair('role', 'user'));
      expect(json[2], containsPair('type', 'message'));
      expect((json[2] as Map<String, dynamic>)['role'], 'assistant');
      expect(json[3], containsPair('type', 'function_call'));
      expect(json[4], containsPair('type', 'function_call_output'));
    });
  });

  group('GenkitConverter.fromOpenAIAssistantMessage', () {
    test('handles refusal', () {
      final msg = AssistantMessage(refusal: 'I cannot do that.');
      final result = GenkitConverter.fromOpenAIAssistantMessage(msg);
      expect(result.content.length, 1);
      expect(result.text, '[Refusal] I cannot do that.');
    });

    test('converts JSON content', () {
      final message =
          ChatMessage.assistant(content: '{"name": "Test", "age": 25}')
              as AssistantMessage;
      final genkitMessage = GenkitConverter.fromOpenAIAssistantMessage(message);
      expect(genkitMessage.role, Role.model);
      expect(genkitMessage.text, '{"name": "Test", "age": 25}');
    });

    test('converts message with tool calls', () {
      final message =
          ChatMessage.assistant(
                content: '{"result": "ok"}',
                toolCalls: [
                  ToolCall.functionCall(
                    id: 'call_123',
                    call: FunctionCall(
                      name: 'getWeather',
                      arguments: '{"location": "NYC"}',
                    ),
                  ),
                ],
              )
              as AssistantMessage;
      final genkitMessage = GenkitConverter.fromOpenAIAssistantMessage(message);
      expect(genkitMessage.text, '{"result": "ok"}');
      final toolParts = genkitMessage.content
          .where((p) => p.isToolRequest)
          .toList();
      expect(toolParts.length, 1);
      expect(toolParts.first.toolRequest!.name, 'getWeather');
    });
  });

  group('GenkitConverter.fromOpenAIResponse', () {
    test('reconstructs text, tool calls, and reasoning summaries', () {
      final response = sdk.Response(
        id: 'resp_1',
        object: 'response',
        createdAt: 0,
        status: sdk.ResponseStatus.completed,
        output: <sdk.OutputItem>[
          sdk.ReasoningItem(
            id: 'reasoning_1',
            summary: const <sdk.ReasoningSummaryContent>[
              sdk.ReasoningSummaryContent(text: 'Checked the repo.'),
            ],
          ),
          sdk.MessageOutputItem(
            id: 'msg_1',
            role: sdk.MessageRole.assistant,
            content: const <sdk.OutputContent>[
              sdk.OutputTextContent(text: 'Done.'),
            ],
          ),
          const sdk.FunctionCallOutputItemResponse(
            id: 'call_1',
            callId: 'tool_1',
            name: 'read_file',
            arguments: '{"path":"README.md"}',
          ),
        ],
      );

      final message = GenkitConverter.fromOpenAIResponse(response);
      expect(message.role, Role.model);
      expect(message.text, 'Done.');
      expect(
        message.content.where((part) => part.isReasoning).single.reasoning,
        'Checked the repo.',
      );
      expect(
        message.content
            .where((part) => part.isToolRequest)
            .single
            .toolRequest
            ?.name,
        'read_file',
      );
    });

    test('preserves assistant message phase metadata on text parts', () {
      final response = sdk.Response(
        id: 'resp_1',
        object: 'response',
        createdAt: 0,
        status: sdk.ResponseStatus.completed,
        output: <sdk.OutputItem>[
          sdk.MessageOutputItem(
            id: 'msg_1',
            role: sdk.MessageRole.assistant,
            phase: sdk.MessagePhase.commentary,
            content: const <sdk.OutputContent>[
              sdk.OutputTextContent(text: 'Working through the repo.'),
            ],
          ),
        ],
      );

      final message = GenkitConverter.fromOpenAIResponse(response);
      final part = message.content.single;
      expect(part.isText, isTrue);
      expect(part.metadata?['assistantMessageId'], 'msg_1');
      expect(part.metadata?['assistantMessagePhase'], 'commentary');
    });
  });

  group('rebuildResponseFromResponsesStreamForTest', () {
    test(
      'recovers completed output items when final response output is empty',
      () {
        final finalResponse = sdk.Response(
          id: 'resp_1',
          object: 'response',
          createdAt: 0,
          status: sdk.ResponseStatus.completed,
          output: const <sdk.OutputItem>[],
        );

        final rebuilt = rebuildResponseFromResponsesStreamForTest(
          finalResponse,
          <sdk.ResponseStreamEvent>[
            sdk.OutputItemDoneEvent(
              outputIndex: 0,
              item: sdk.ReasoningItem(
                id: 'reasoning_1',
                summary: const <sdk.ReasoningSummaryContent>[
                  sdk.ReasoningSummaryContent(text: 'Checked the repo.'),
                ],
              ),
            ),
            const sdk.OutputItemDoneEvent(
              outputIndex: 1,
              item: sdk.FunctionCallOutputItemResponse(
                id: 'call_1',
                callId: 'tool_1',
                name: 'update_plan',
                arguments: '{"items":[]}',
              ),
            ),
          ],
        );

        expect(rebuilt.output, hasLength(2));
        expect(rebuilt.output.first, isA<sdk.ReasoningItem>());
        expect(rebuilt.output.last, isA<sdk.FunctionCallOutputItemResponse>());
        expect(
          (rebuilt.output.last as sdk.FunctionCallOutputItemResponse).name,
          'update_plan',
        );
      },
    );

    test('synthesizes function call output from added item and done args', () {
      final finalResponse = sdk.Response(
        id: 'resp_1',
        object: 'response',
        createdAt: 0,
        status: sdk.ResponseStatus.completed,
        output: const <sdk.OutputItem>[],
      );

      final rebuilt = rebuildResponseFromResponsesStreamForTest(
        finalResponse,
        <sdk.ResponseStreamEvent>[
          const sdk.OutputItemAddedEvent(
            outputIndex: 0,
            item: sdk.FunctionCallOutputItemResponse(
              id: 'call_1',
              callId: 'tool_1',
              name: 'read_file',
              arguments: '',
            ),
          ),
          const sdk.FunctionCallArgumentsDoneEvent(
            outputIndex: 0,
            itemId: 'call_1',
            arguments: '{"path":"README.md"}',
          ),
        ],
      );

      expect(rebuilt.output, hasLength(1));
      final toolCall =
          rebuilt.output.single as sdk.FunctionCallOutputItemResponse;
      expect(toolCall.name, 'read_file');
      expect(toolCall.arguments, '{"path":"README.md"}');
      expect(toolCall.callId, 'tool_1');
    });
  });

  group('assistantTextChunksFromResponsesEventsForTest', () {
    test('emits output_text.done when no text delta was observed', () {
      final chunks = assistantTextChunksFromResponsesEventsForTest(
        <sdk.ResponseStreamEvent>[
          const sdk.OutputTextDoneEvent(
            outputIndex: 0,
            contentIndex: 0,
            text: 'Final answer',
          ),
        ],
      );

      expect(chunks, <String>['Final answer']);
    });

    test(
      'does not duplicate text when delta already covered the same part',
      () {
        final chunks = assistantTextChunksFromResponsesEventsForTest(
          <sdk.ResponseStreamEvent>[
            const sdk.OutputTextDeltaEvent(
              outputIndex: 0,
              contentIndex: 0,
              delta: 'Hello',
            ),
            const sdk.OutputTextDoneEvent(
              outputIndex: 0,
              contentIndex: 0,
              text: 'Hello',
            ),
          ],
        );

        expect(chunks, <String>['Hello']);
      },
    );

    test(
      'falls back to output item text when the proxy only emits item done',
      () {
        final chunks = assistantTextChunksFromResponsesEventsForTest(
          <sdk.ResponseStreamEvent>[
            sdk.OutputItemDoneEvent(
              outputIndex: 0,
              item: sdk.MessageOutputItem(
                id: 'msg_1',
                role: sdk.MessageRole.assistant,
                content: const <sdk.OutputContent>[
                  sdk.OutputTextContent(text: 'Chunk from item done'),
                ],
              ),
            ),
          ],
        );

        expect(chunks, <String>['Chunk from item done']);
      },
    );

    test(
      'emits phase metadata and a completion boundary for message items',
      () {
        final parts = assistantTextPartsFromResponsesEventsForTest(
          <sdk.ResponseStreamEvent>[
            sdk.OutputItemDoneEvent(
              outputIndex: 0,
              item: sdk.MessageOutputItem(
                id: 'msg_1',
                role: sdk.MessageRole.assistant,
                phase: sdk.MessagePhase.commentary,
                content: const <sdk.OutputContent>[
                  sdk.OutputTextContent(text: 'Chunk from item done'),
                ],
              ),
            ),
          ],
        );

        expect(parts, hasLength(2));
        expect(parts.first.text, 'Chunk from item done');
        expect(parts.first.metadata?['assistantMessageId'], 'msg_1');
        expect(parts.first.metadata?['assistantMessagePhase'], 'commentary');
        expect(parts.last.text, isEmpty);
        expect(parts.last.metadata?['assistantMessageBoundary'], 'completed');
      },
    );
  });

  group('GenkitConverter.mapFinishReason', () {
    test('maps stop', () {
      expect(GenkitConverter.mapFinishReason('stop'), FinishReason.stop);
    });

    test('maps length', () {
      expect(GenkitConverter.mapFinishReason('length'), FinishReason.length);
    });

    test('maps content_filter', () {
      expect(
        GenkitConverter.mapFinishReason('content_filter'),
        FinishReason.blocked,
      );
    });

    test('maps tool_calls', () {
      expect(GenkitConverter.mapFinishReason('tool_calls'), FinishReason.stop);
    });

    test('maps unknown', () {
      expect(GenkitConverter.mapFinishReason('unknown'), FinishReason.unknown);
      expect(GenkitConverter.mapFinishReason(null), FinishReason.unknown);
    });
  });
}
