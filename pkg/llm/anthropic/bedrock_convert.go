package anthropic

import (
	"encoding/json"
	"strings"
	"time"

	sdkanthropic "github.com/anthropics/anthropic-sdk-go"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
)

// convertToSDKParams converts an internal CompletionRequest to official SDK MessageNewParams
func convertToSDKParams(req *CompletionRequest) sdkanthropic.MessageNewParams {
	params := sdkanthropic.MessageNewParams{
		MaxTokens: int64(req.MaxTokens),
		Model:     sdkanthropic.Model(req.Model),
	}

	// Convert messages
	sdkMessages := make([]sdkanthropic.MessageParam, 0, len(req.Messages))
	for _, msg := range req.Messages {
		switch msg.Role {
		case "user":
			sdkMessages = append(sdkMessages, sdkanthropic.NewUserMessage(
				sdkanthropic.NewTextBlock(msg.Content),
			))
		case "assistant":
			sdkMessages = append(sdkMessages, sdkanthropic.NewAssistantMessage(
				sdkanthropic.NewTextBlock(msg.Content),
			))
		}
	}
	params.Messages = sdkMessages

	// Convert system message
	if req.System != "" {
		params.System = []sdkanthropic.TextBlockParam{
			{Text: req.System},
		}
	}

	// Convert temperature
	if req.Temperature != 0 {
		params.Temperature = sdkanthropic.Float(req.Temperature)
	}

	// Convert TopP
	if req.TopP != 0 {
		params.TopP = sdkanthropic.Float(req.TopP)
	}

	// Convert TopK
	if req.TopK != 0 {
		params.TopK = sdkanthropic.Int(int64(req.TopK))
	}

	// Convert stop sequences
	if len(req.StopSequences) > 0 {
		params.StopSequences = req.StopSequences
	}

	// Convert tools
	if len(req.Tools) > 0 {
		sdkTools := make([]sdkanthropic.ToolUnionParam, len(req.Tools))
		for i, tool := range req.Tools {
			sdkTools[i] = convertToolToSDK(tool)
		}
		params.Tools = sdkTools
	}

	// Convert tool choice
	if req.ToolChoice != nil {
		params.ToolChoice = convertToolChoiceToSDK(req.ToolChoice)
	}

	// Convert thinking/reasoning
	if req.Thinking != nil {
		switch req.Thinking.Type {
		case "adaptive":
			params.Thinking = sdkanthropic.ThinkingConfigParamUnion{
				OfAdaptive: &sdkanthropic.ThinkingConfigAdaptiveParam{},
			}
		case "enabled":
			budgetTokens := int64(req.Thinking.BudgetTokens)
			if budgetTokens < 1024 {
				budgetTokens = 1024
			}
			params.Thinking = sdkanthropic.ThinkingConfigParamOfEnabled(budgetTokens)
		}
	}

	// Convert effort / output config
	if req.OutputConfig != nil && req.OutputConfig.Effort != "" {
		params.OutputConfig = sdkanthropic.OutputConfigParam{
			Effort: sdkanthropic.OutputConfigEffort(req.OutputConfig.Effort),
		}
	}

	return params
}

// convertToolToSDK converts an internal Tool to SDK ToolUnionParam
func convertToolToSDK(tool Tool) sdkanthropic.ToolUnionParam {
	// Extract properties and required from input schema
	var properties interface{}
	var required []string

	if props, ok := tool.InputSchema["properties"]; ok {
		properties = props
	}
	if req, ok := tool.InputSchema["required"]; ok {
		if reqSlice, ok := req.([]interface{}); ok {
			for _, r := range reqSlice {
				if s, ok := r.(string); ok {
					required = append(required, s)
				}
			}
		} else if reqStrSlice, ok := req.([]string); ok {
			required = reqStrSlice
		}
	}

	toolParam := sdkanthropic.ToolParam{
		Name:        tool.Name,
		Description: sdkanthropic.String(tool.Description),
		InputSchema: sdkanthropic.ToolInputSchemaParam{
			Properties: properties,
			Required:   required,
		},
	}

	return sdkanthropic.ToolUnionParam{OfTool: &toolParam}
}

// convertToolChoiceToSDK converts an internal tool choice to SDK ToolChoiceUnionParam
func convertToolChoiceToSDK(toolChoice interface{}) sdkanthropic.ToolChoiceUnionParam {
	switch tc := toolChoice.(type) {
	case map[string]string:
		return convertToolChoiceMapStringToSDK(tc)
	case map[string]interface{}:
		return convertToolChoiceMapInterfaceToSDK(tc)
	default:
		// Default to auto
		return sdkanthropic.ToolChoiceUnionParam{OfAuto: &sdkanthropic.ToolChoiceAutoParam{}}
	}
}

func convertToolChoiceMapStringToSDK(tc map[string]string) sdkanthropic.ToolChoiceUnionParam {
	switch tc["type"] {
	case "auto":
		return sdkanthropic.ToolChoiceUnionParam{OfAuto: &sdkanthropic.ToolChoiceAutoParam{}}
	case "any":
		return sdkanthropic.ToolChoiceUnionParam{OfAny: &sdkanthropic.ToolChoiceAnyParam{}}
	case "tool":
		return sdkanthropic.ToolChoiceParamOfTool(tc["name"])
	case "none":
		return sdkanthropic.ToolChoiceUnionParam{OfNone: &sdkanthropic.ToolChoiceNoneParam{}}
	default:
		return sdkanthropic.ToolChoiceUnionParam{OfAuto: &sdkanthropic.ToolChoiceAutoParam{}}
	}
}

func convertToolChoiceMapInterfaceToSDK(tc map[string]interface{}) sdkanthropic.ToolChoiceUnionParam {
	choiceType, _ := tc["type"].(string)
	switch choiceType {
	case "auto":
		return sdkanthropic.ToolChoiceUnionParam{OfAuto: &sdkanthropic.ToolChoiceAutoParam{}}
	case "any":
		return sdkanthropic.ToolChoiceUnionParam{OfAny: &sdkanthropic.ToolChoiceAnyParam{}}
	case "tool":
		name, _ := tc["name"].(string)
		return sdkanthropic.ToolChoiceParamOfTool(name)
	case "none":
		return sdkanthropic.ToolChoiceUnionParam{OfNone: &sdkanthropic.ToolChoiceNoneParam{}}
	default:
		return sdkanthropic.ToolChoiceUnionParam{OfAuto: &sdkanthropic.ToolChoiceAutoParam{}}
	}
}

// convertFromSDKMessage converts an SDK Message to internal CompletionResponse
func convertFromSDKMessage(msg *sdkanthropic.Message) *CompletionResponse {
	resp := &CompletionResponse{
		ID:         msg.ID,
		Type:       string(msg.Type),
		Role:       string(msg.Role),
		Model:      string(msg.Model),
		StopReason: string(msg.StopReason),
		Usage: Usage{
			InputTokens:              int(msg.Usage.InputTokens),
			OutputTokens:             int(msg.Usage.OutputTokens),
			CacheCreationInputTokens: int(msg.Usage.CacheCreationInputTokens),
			CacheReadInputTokens:     int(msg.Usage.CacheReadInputTokens),
		},
	}

	// Convert content blocks
	for _, block := range msg.Content {
		contentBlock := convertSDKContentBlock(block)
		resp.Content = append(resp.Content, contentBlock)
	}

	return resp
}

// convertSDKContentBlock converts an SDK ContentBlockUnion to internal ContentBlock
func convertSDKContentBlock(block sdkanthropic.ContentBlockUnion) ContentBlock {
	switch block.Type {
	case "text":
		return ContentBlock{
			Type: "text",
			Text: block.Text,
		}
	case "tool_use":
		// Parse input from json.RawMessage
		var inputMap map[string]interface{}
		if len(block.Input) > 0 {
			_ = json.Unmarshal(block.Input, &inputMap)
		}
		return ContentBlock{
			Type: "tool_use",
			ID:   block.ID,
			Name: block.Name,
			Input: inputMap,
			ToolUse: &ToolUse{
				ID:    block.ID,
				Name:  block.Name,
				Input: inputMap,
			},
		}
	case "thinking":
		return ContentBlock{
			Type: "thinking",
			Text: block.Thinking,
		}
	default:
		return ContentBlock{
			Type: block.Type,
			Text: block.Text,
		}
	}
}

// convertSDKStreamEvent converts an SDK MessageStreamEventUnion to internal StreamEvent.
// Returns nil if the event should be skipped.
// This function reads directly from the flat union struct fields rather than using
// the As*() methods, which require JSON.raw to be populated via UnmarshalJSON.
func convertSDKStreamEvent(
	event sdkanthropic.MessageStreamEventUnion,
	thinkingBlocks map[int64]bool,
	toolBlocks map[int64]*sdkToolBlockAccumulator,
) *interfaces.StreamEvent {
	streamEvent := &interfaces.StreamEvent{
		Timestamp: time.Now(),
		Metadata:  make(map[string]interface{}),
	}

	switch event.Type {
	case "message_start":
		streamEvent.Type = interfaces.StreamEventMessageStart
		streamEvent.Metadata["message_id"] = event.Message.ID
		streamEvent.Metadata["model"] = string(event.Message.Model)
		streamEvent.Metadata["role"] = string(event.Message.Role)

	case "content_block_start":
		switch event.ContentBlock.Type {
		case "thinking":
			thinkingBlocks[event.Index] = true
			streamEvent.Type = interfaces.StreamEventThinking
			streamEvent.Content = event.ContentBlock.Thinking
			streamEvent.Metadata["block_index"] = event.Index
			streamEvent.Metadata["block_type"] = "thinking"
		case "tool_use":
			thinkingBlocks[event.Index] = false
			toolBlocks[event.Index] = &sdkToolBlockAccumulator{
				ID:   event.ContentBlock.ID,
				Name: event.ContentBlock.Name,
			}
			// Don't emit event yet; wait for input to be fully accumulated
			return nil
		default: // "text"
			thinkingBlocks[event.Index] = false
			streamEvent.Type = interfaces.StreamEventContentDelta
			streamEvent.Content = event.ContentBlock.Text
			streamEvent.Metadata["block_index"] = event.Index
			streamEvent.Metadata["block_type"] = event.ContentBlock.Type
		}

	case "content_block_delta":
		switch event.Delta.Type {
		case "thinking_delta":
			streamEvent.Type = interfaces.StreamEventThinking
			streamEvent.Content = event.Delta.Thinking
			streamEvent.Metadata["block_index"] = event.Index
		case "input_json_delta":
			// Accumulate tool input
			if acc, ok := toolBlocks[event.Index]; ok {
				acc.InputJSON.WriteString(event.Delta.PartialJSON)
			}
			return nil
		default: // "text_delta"
			streamEvent.Type = interfaces.StreamEventContentDelta
			streamEvent.Content = event.Delta.Text
			streamEvent.Metadata["block_index"] = event.Index
		}
		streamEvent.Metadata["delta_type"] = event.Delta.Type

	case "content_block_stop":
		if acc, ok := toolBlocks[event.Index]; ok {
			// Emit complete tool call
			streamEvent.Type = interfaces.StreamEventToolUse
			streamEvent.ToolCall = &interfaces.ToolCall{
				ID:        acc.ID,
				Name:      acc.Name,
				Arguments: acc.InputJSON.String(),
			}
			streamEvent.Metadata["block_index"] = event.Index
			delete(toolBlocks, event.Index)
		} else {
			streamEvent.Type = interfaces.StreamEventContentComplete
			streamEvent.Metadata["block_index"] = event.Index
		}

	case "message_delta":
		streamEvent.Type = interfaces.StreamEventContentDelta
		streamEvent.Metadata["stop_reason"] = string(event.Delta.StopReason)
		streamEvent.Metadata["usage"] = map[string]interface{}{
			"output_tokens": event.Usage.OutputTokens,
		}

	case "message_stop":
		streamEvent.Type = interfaces.StreamEventMessageStop

	default:
		// Unknown event type, skip
		return nil
	}

	return streamEvent
}

// sdkToolBlockAccumulator tracks tool block input during streaming
type sdkToolBlockAccumulator struct {
	ID        string
	Name      string
	InputJSON strings.Builder
}
