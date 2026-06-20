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

import 'package:genkit/genkit.dart';
import 'package:openai_dart/openai_dart.dart' as sdk;

/// Converter class for transforming between Genkit and OpenAI formats
abstract final class GenkitConverter {
  /// Convert Genkit messages to OpenAI format
  static List<sdk.ChatMessage> toOpenAIMessages(
    List<Message> messages,
    String? visualDetailLevel,
  ) {
    final result = <sdk.ChatMessage>[];
    for (final message in messages) {
      // Tool messages may contain multiple responses and need to be expanded
      if (message.role == Role.tool) {
        final toolResponses = message.content
            .where((p) => p.isToolResponse)
            .map((p) => p.toolResponse!)
            .toList();

        if (toolResponses.isEmpty) {
          throw ArgumentError(
            'Tool message must contain at least one ToolResponsePart',
          );
        }

        // Create a separate message for each tool response
        for (final toolResponse in toolResponses) {
          final ref = toolResponse.ref;
          if (ref == null || ref.isEmpty) {
            throw ArgumentError(
              'ToolResponse.ref must be a non-empty string for tool messages',
            );
          }
          result.add(
            sdk.ChatMessage.tool(
              toolCallId: ref,
              content: jsonEncode(toolResponse.output),
            ),
          );
        }
      } else {
        result.add(toOpenAIMessage(message, visualDetailLevel));
      }
    }
    return result;
  }

  /// Convert Genkit messages to Responses API input items.
  static sdk.ResponseInput toOpenAIResponseInput(
    List<Message> messages,
    String? visualDetailLevel,
  ) {
    final itemMaps = <Map<String, dynamic>>[];
    for (final message in messages) {
      if (message.role == Role.system) {
        continue;
      }
      itemMaps.addAll(_toOpenAIResponseItemMaps(message, visualDetailLevel));
    }
    return sdk.ResponseInput.fromOutputItems(itemMaps);
  }

  /// Convert Genkit system messages to the Responses API instructions field.
  static String? toOpenAIResponseInstructions(List<Message> messages) {
    final sections = <String>[];
    for (final message in messages) {
      if (message.role != Role.system) {
        continue;
      }
      final text = message.content
          .where((part) => part.isText)
          .map((part) => part.text!.trim())
          .where((text) => text.isNotEmpty)
          .join('\n')
          .trim();
      if (text.isNotEmpty) {
        sections.add(text);
      }
    }
    if (sections.isEmpty) {
      return null;
    }
    return sections.join('\n\n');
  }

  /// Convert a single Genkit message to OpenAI format
  /// Note: Tool messages are handled separately in toOpenAIMessages()
  static sdk.ChatMessage toOpenAIMessage(
    Message msg,
    String? visualDetailLevel,
  ) {
    if (msg.role == Role.system) {
      return sdk.ChatMessage.system(msg.text);
    }
    if (msg.role == Role.user) {
      final parts = msg.content
          .map((p) => toOpenAIContentPart(p, visualDetailLevel))
          .toList();
      return sdk.ChatMessage.user(parts);
    }
    if (msg.role == Role.model) {
      final toolCalls = _extractToolCalls(msg.content);
      return sdk.ChatMessage.assistant(
        content: msg.text,
        toolCalls: toolCalls.isNotEmpty ? toolCalls : null,
      );
    }
    if (msg.role == Role.tool) {
      throw ArgumentError(
        'Tool messages should be handled by toOpenAIMessages(), not toOpenAIMessage()',
      );
    }
    throw UnimplementedError('Unsupported role: ${msg.role}');
  }

  /// Convert Genkit Part to OpenAI content part
  static sdk.ContentPart toOpenAIContentPart(
    Part part,
    String? visualDetailLevel,
  ) {
    if (part.isText) {
      return sdk.ContentPart.text(part.text!);
    }
    if (part.isMedia) {
      final media = part.media!;
      final mimeType = media.contentType?.toLowerCase();
      if (media.url.startsWith('data:')) {
        // Parse data URI: data:<mediaType>;base64,<data>
        final commaIdx = media.url.indexOf(',');
        final base64Data = media.url.substring(commaIdx + 1);
        if (_isAudioMimeType(mimeType)) {
          return sdk.ContentPart.inputAudio(
            data: base64Data,
            format: _mapAudioFormat(mimeType!),
          );
        }
        return sdk.ContentPart.imageBase64(
          data: base64Data,
          mediaType: mimeType ?? 'image/png',
          detail: _mapVisualDetailLevel(visualDetailLevel),
        );
      }
      if (_isAudioMimeType(mimeType)) {
        throw UnimplementedError(
          'Audio MediaPart currently requires a data: URI payload.',
        );
      }
      return sdk.ContentPart.imageUrl(
        media.url,
        detail: _mapVisualDetailLevel(visualDetailLevel),
      );
    }
    throw UnimplementedError('Unsupported part type: $part');
  }

  /// Map visual detail level string to enum
  static sdk.ImageDetail _mapVisualDetailLevel(String? level) {
    return switch (level) {
      'low' => sdk.ImageDetail.low,
      'high' => sdk.ImageDetail.high,
      _ => sdk.ImageDetail.auto,
    };
  }

  static bool _isAudioMimeType(String? mimeType) {
    return mimeType != null && mimeType.startsWith('audio/');
  }

  static sdk.AudioFormat _mapAudioFormat(String mimeType) {
    return switch (mimeType) {
      'audio/wav' || 'audio/x-wav' => sdk.AudioFormat.wav,
      'audio/mpeg' || 'audio/mp3' => sdk.AudioFormat.mp3,
      'audio/flac' => sdk.AudioFormat.flac,
      'audio/ogg' ||
      'audio/opus' ||
      'audio/ogg; codecs=opus' => sdk.AudioFormat.opus,
      'audio/pcm' || 'audio/pcm16' => sdk.AudioFormat.pcm16,
      _ => throw UnimplementedError(
        'Unsupported audio MediaPart contentType: $mimeType',
      ),
    };
  }

  /// Extract tool calls from message content
  static List<sdk.ToolCall> _extractToolCalls(List<Part> content) {
    final toolCalls = <sdk.ToolCall>[];
    for (final part in content) {
      if (part.isToolRequest) {
        final toolRequest = part.toolRequest!;
        final ref = toolRequest.ref;
        if (ref == null || ref.isEmpty) {
          throw ArgumentError(
            'ToolRequest.ref must be a non-empty string when converting to OpenAI tool calls',
          );
        }
        toolCalls.add(
          sdk.ToolCall.functionCall(
            id: ref,
            call: sdk.FunctionCall(
              name: toolRequest.name,
              arguments: jsonEncode(toolRequest.input ?? {}),
            ),
          ),
        );
      }
    }
    return toolCalls;
  }

  /// Convert Genkit tool to OpenAI format
  static sdk.Tool toOpenAITool(ToolDefinition tool) {
    // OpenAI requires parameters to be a valid JSON Schema object
    // If no schema is provided, use an empty object schema
    var parameters = tool.inputSchema;

    if (parameters == null) {
      parameters = {'type': 'object', 'properties': {}};
    } else if (!parameters.containsKey('type')) {
      // Ensure the schema has a type field
      parameters = {'type': 'object', ...parameters};
    }

    return sdk.Tool.function(
      name: tool.name,
      description: tool.description,
      parameters: parameters,
    );
  }

  /// Convert Genkit tool to Responses API format.
  static sdk.ResponseTool toOpenAIResponseTool(ToolDefinition tool) {
    final metadata = tool.metadata ?? const <String, dynamic>{};
    final nativeToolType = metadata['nativeToolType'] as String?;
    if (nativeToolType == 'image_generation') {
      return sdk.ResponseTool.imageGeneration(
        outputFormat: metadata['outputFormat'] as String? ?? 'png',
      );
    }

    var parameters = tool.inputSchema;

    if (parameters == null) {
      parameters = {'type': 'object', 'properties': {}};
    } else if (!parameters.containsKey('type')) {
      parameters = {'type': 'object', ...parameters};
    }

    return sdk.ResponseTool.function(
      name: tool.name,
      description: tool.description,
      parameters: parameters,
    );
  }

  /// Convert OpenAI assistant message to Genkit format.
  ///
  /// This is used for converting response messages from the OpenAI API.
  /// For responses, we always get an [sdk.AssistantMessage] with
  /// optional text content, refusal, and/or tool calls.
  static Message fromOpenAIAssistantMessage(sdk.AssistantMessage msg) {
    final parts = <Part>[];

    final summaryDetails =
        (msg.reasoningDetails ?? const <sdk.ReasoningDetail>[])
            .where(
              (detail) =>
                  detail.isSummary && (detail.text?.isNotEmpty ?? false),
            )
            .toList(growable: false);
    if (summaryDetails.isNotEmpty) {
      for (final detail in summaryDetails.indexed) {
        parts.add(
          ReasoningPart(
            reasoning: detail.$2.text!,
            metadata: <String, dynamic>{
              'reasoningType': 'summary',
              if (detail.$1 > 0) 'sectionBreak': true,
            },
          ),
        );
      }
    } else if (msg.reasoning?.isNotEmpty ?? false) {
      parts.add(
        ReasoningPart(
          reasoning: msg.reasoning!,
          metadata: const <String, dynamic>{'reasoningType': 'summary'},
        ),
      );
    }

    // Handle refusal
    if (msg.refusal != null && msg.refusal!.isNotEmpty) {
      parts.add(TextPart(text: '[Refusal] ${msg.refusal}'));
    }

    // Handle text content (always a String? for assistant messages)
    if (msg.content != null && msg.content!.isNotEmpty) {
      parts.add(TextPart(text: msg.content!));
    }

    // Handle tool calls
    if (msg.toolCalls != null) {
      for (final toolCall in msg.toolCalls!) {
        parts.add(
          ToolRequestPart(
            toolRequest: ToolRequest(
              ref: toolCall.id,
              name: toolCall.function.name,
              input: toolCall.function.arguments.isNotEmpty
                  ? jsonDecode(toolCall.function.arguments)
                        as Map<String, dynamic>?
                  : null,
            ),
          ),
        );
      }
    }

    return Message(role: Role.model, content: parts);
  }

  /// Convert a Responses API response to Genkit format.
  static Message fromOpenAIResponse(sdk.Response response) {
    final parts = <Part>[];
    var emittedReasoningSummary = false;

    for (final item in response.output) {
      if (item is sdk.ReasoningItem) {
        for (final summary in item.summary) {
          parts.add(
            ReasoningPart(
              reasoning: summary.text,
              metadata: <String, dynamic>{
                'reasoningType': 'summary',
                if (emittedReasoningSummary) 'sectionBreak': true,
              },
            ),
          );
          emittedReasoningSummary = true;
        }
        continue;
      }

      if (item is sdk.MessageOutputItem) {
        final metadata = _assistantTextMetadata(
          itemId: item.id,
          phase: item.phase,
        );
        for (final content in item.content) {
          switch (content) {
            case sdk.OutputTextContent(:final text):
              parts.add(TextPart(text: text, metadata: metadata));
            case sdk.RefusalContent(:final refusal):
              parts.add(
                TextPart(text: '[Refusal] $refusal', metadata: metadata),
              );
            default:
              break;
          }
        }
        continue;
      }

      if (item is sdk.FunctionCallOutputItemResponse) {
        parts.add(
          ToolRequestPart(
            toolRequest: ToolRequest(
              ref: item.callId,
              name: item.name,
              input: item.argumentsMap,
            ),
          ),
        );
      }

      if (item is sdk.ImageGenerationCallOutputItem) {
        parts.add(
          CustomPart(
            custom: <String, dynamic>{
              'type': 'image_generation_call',
              'id': item.id,
              if (item.prompt != null) 'prompt': item.prompt,
              if (item.revisedPrompt != null)
                'revisedPrompt': item.revisedPrompt,
              if (item.result != null) 'result': item.result,
              if (item.status != null) 'status': item.status!.toJson(),
            },
          ),
        );
      }
    }

    return Message(role: Role.model, content: parts);
  }

  /// Map OpenAI finish reason to Genkit FinishReason
  static FinishReason mapFinishReason(String? reason) {
    return switch (reason) {
      'stop' => FinishReason.stop,
      'length' => FinishReason.length,
      'content_filter' => FinishReason.blocked,
      'tool_calls' => FinishReason.stop,
      _ => FinishReason.unknown,
    };
  }

  /// Map Responses API terminal state to Genkit FinishReason.
  static FinishReason mapResponseFinishReason(sdk.Response response) {
    return switch (response.status) {
      sdk.ResponseStatus.completed => FinishReason.stop,
      sdk.ResponseStatus.incomplete =>
        switch (response.incompleteDetails?.reason) {
          'max_output_tokens' || 'max_tokens' => FinishReason.length,
          _ => FinishReason.unknown,
        },
      sdk.ResponseStatus.failed =>
        _isContentFilteredResponse(response)
            ? FinishReason.blocked
            : FinishReason.unknown,
      _ => FinishReason.unknown,
    };
  }

  static List<Map<String, dynamic>> _toOpenAIResponseItemMaps(
    Message message,
    String? visualDetailLevel,
  ) {
    if (message.role == Role.system) {
      return const <Map<String, dynamic>>[];
    }

    if (message.role == Role.user) {
      return <Map<String, dynamic>>[
        <String, dynamic>{
          'type': 'message',
          'role': 'user',
          'content': message.content
              .map(
                (part) => _toOpenAIResponseContentPartMap(
                  part,
                  visualDetailLevel,
                  assistant: false,
                ),
              )
              .toList(growable: false),
        },
      ];
    }

    if (message.role == Role.model) {
      final items = <Map<String, dynamic>>[];
      final assistantContent = message.content
          .where((part) => part.isText)
          .map(
            (part) => _toOpenAIResponseContentPartMap(
              part,
              visualDetailLevel,
              assistant: true,
            ),
          )
          .toList(growable: false);
      if (assistantContent.isNotEmpty) {
        items.add(<String, dynamic>{
          'type': 'message',
          'role': 'assistant',
          'content': assistantContent,
        });
      }
      for (final part in message.content) {
        if (!part.isToolRequest) {
          continue;
        }
        final toolRequest = part.toolRequest!;
        final ref = toolRequest.ref;
        if (ref == null || ref.isEmpty) {
          throw ArgumentError(
            'ToolRequest.ref must be a non-empty string when converting to OpenAI responses items',
          );
        }
        items.add(<String, dynamic>{
          'type': 'function_call',
          'call_id': ref,
          'name': toolRequest.name,
          'arguments': jsonEncode(
            toolRequest.input ?? const <String, dynamic>{},
          ),
        });
      }
      return items;
    }

    final toolResponses = message.content
        .where((p) => p.isToolResponse)
        .map((p) => p.toolResponse!)
        .toList(growable: false);
    if (toolResponses.isEmpty) {
      throw ArgumentError(
        'Tool message must contain at least one ToolResponsePart',
      );
    }
    return toolResponses
        .map((toolResponse) {
          final ref = toolResponse.ref;
          if (ref == null || ref.isEmpty) {
            throw ArgumentError(
              'ToolResponse.ref must be a non-empty string for tool messages',
            );
          }
          return <String, dynamic>{
            'type': 'function_call_output',
            'call_id': ref,
            'output': jsonEncode(toolResponse.output),
          };
        })
        .toList(growable: false);
  }

  static Map<String, dynamic> _toOpenAIResponseContentPartMap(
    Part part,
    String? visualDetailLevel, {
    required bool assistant,
  }) {
    if (part.isText) {
      return <String, dynamic>{
        'type': assistant ? 'output_text' : 'input_text',
        'text': part.text!,
      };
    }
    if (!part.isMedia) {
      throw UnimplementedError('Unsupported part type: $part');
    }

    final media = part.media!;
    final mimeType = media.contentType?.toLowerCase();
    final detail = _mapVisualDetailLevel(visualDetailLevel);
    if (media.url.startsWith('data:')) {
      final commaIdx = media.url.indexOf(',');
      final base64Data = media.url.substring(commaIdx + 1);
      if (_isAudioMimeType(mimeType)) {
        return <String, dynamic>{
          'type': 'input_audio',
          'input_audio': <String, dynamic>{
            'data': base64Data,
            'format': _mapAudioFormat(mimeType!).toJson(),
          },
        };
      }
      if (_isVideoMimeType(mimeType)) {
        throw UnimplementedError(
          'Video MediaPart currently requires a non-data URL payload for Responses API.',
        );
      }
      if (_isImageMimeType(mimeType) || mimeType == null) {
        return <String, dynamic>{
          'type': 'input_image',
          'image_url': media.url,
          'detail': detail.toJson(),
        };
      }
      return <String, dynamic>{'type': 'input_file', 'file_data': media.url};
    }
    if (_isAudioMimeType(mimeType)) {
      throw UnimplementedError(
        'Audio MediaPart currently requires a data: URI payload.',
      );
    }
    if (_isVideoMimeType(mimeType)) {
      return <String, dynamic>{'type': 'input_video', 'video_url': media.url};
    }
    if (_isImageMimeType(mimeType) || mimeType == null) {
      return <String, dynamic>{
        'type': 'input_image',
        'image_url': media.url,
        'detail': detail.toJson(),
      };
    }
    return <String, dynamic>{'type': 'input_file', 'file_url': media.url};
  }

  static bool _isImageMimeType(String? mimeType) {
    return mimeType != null && mimeType.startsWith('image/');
  }

  static bool _isVideoMimeType(String? mimeType) {
    return mimeType != null && mimeType.startsWith('video/');
  }

  static bool _isContentFilteredResponse(sdk.Response response) {
    final type = response.error?.type.toLowerCase();
    final code = response.error?.code?.toLowerCase();
    final message = response.error?.message.toLowerCase() ?? '';
    return type == 'content_filter' ||
        code == 'content_filter' ||
        message.contains('content filter');
  }

  static Map<String, dynamic>? _assistantTextMetadata({
    required String itemId,
    required sdk.MessagePhase? phase,
  }) {
    final normalizedItemId = itemId.trim();
    if (normalizedItemId.isEmpty && phase == null) {
      return null;
    }
    return <String, dynamic>{
      if (normalizedItemId.isNotEmpty) 'assistantMessageId': normalizedItemId,
      if (phase != null) 'assistantMessagePhase': phase.toJson(),
    };
  }
}
