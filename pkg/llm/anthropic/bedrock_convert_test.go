package anthropic

import (
	"encoding/json"
	"testing"

	sdkanthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
)

func TestConvertToSDKParams_BasicMessage(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 1024,
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
	}

	params := convertToSDKParams(req)

	assert.Equal(t, sdkanthropic.Model("claude-3-5-sonnet-latest"), params.Model)
	assert.Equal(t, int64(1024), params.MaxTokens)
	assert.Len(t, params.Messages, 1)
}

func TestConvertToSDKParams_MultipleMessages(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 2048,
		Messages: []Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
			{Role: "user", Content: "How are you?"},
		},
	}

	params := convertToSDKParams(req)

	assert.Len(t, params.Messages, 3)
}

func TestConvertToSDKParams_SystemMessage(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		System:    "You are a helpful assistant.",
	}

	params := convertToSDKParams(req)

	require.Len(t, params.System, 1)
	assert.Equal(t, "You are a helpful assistant.", params.System[0].Text)
}

func TestConvertToSDKParams_Temperature(t *testing.T) {
	req := &CompletionRequest{
		Model:       "claude-3-5-sonnet-latest",
		MaxTokens:   1024,
		Messages:    []Message{{Role: "user", Content: "Hello"}},
		Temperature: 0.5,
	}

	params := convertToSDKParams(req)

	assert.Equal(t, 0.5, params.Temperature.Value)
}

func TestConvertToSDKParams_TopPAndTopK(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		TopP:      0.9,
		TopK:      50,
	}

	params := convertToSDKParams(req)

	assert.Equal(t, 0.9, params.TopP.Value)
	assert.Equal(t, int64(50), params.TopK.Value)
}

func TestConvertToSDKParams_StopSequences(t *testing.T) {
	req := &CompletionRequest{
		Model:         "claude-3-5-sonnet-latest",
		MaxTokens:     1024,
		Messages:      []Message{{Role: "user", Content: "Hello"}},
		StopSequences: []string{"\n\nHuman:", "END"},
	}

	params := convertToSDKParams(req)

	assert.Equal(t, []string{"\n\nHuman:", "END"}, params.StopSequences)
}

func TestConvertToSDKParams_Tools(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		Tools: []Tool{
			{
				Name:        "get_weather",
				Description: "Get current weather",
				InputSchema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "City name",
						},
					},
					"required": []interface{}{"location"},
				},
			},
		},
	}

	params := convertToSDKParams(req)

	require.Len(t, params.Tools, 1)
	toolParam := params.Tools[0].OfTool
	require.NotNil(t, toolParam)
	assert.Equal(t, "get_weather", toolParam.Name)
	assert.Equal(t, "Get current weather", toolParam.Description.Value)
	assert.Equal(t, []string{"location"}, toolParam.InputSchema.Required)
}

func TestConvertToSDKParams_ToolChoiceAuto(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		ToolChoice: map[string]string{
			"type": "auto",
		},
	}

	params := convertToSDKParams(req)

	assert.NotNil(t, params.ToolChoice.OfAuto)
}

func TestConvertToSDKParams_ToolChoiceAny(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		ToolChoice: map[string]string{
			"type": "any",
		},
	}

	params := convertToSDKParams(req)

	assert.NotNil(t, params.ToolChoice.OfAny)
}

func TestConvertToSDKParams_ToolChoiceTool(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		ToolChoice: map[string]string{
			"type": "tool",
			"name": "get_weather",
		},
	}

	params := convertToSDKParams(req)

	require.NotNil(t, params.ToolChoice.OfTool)
	assert.Equal(t, "get_weather", params.ToolChoice.OfTool.Name)
}

func TestConvertToSDKParams_ToolChoiceMapInterface(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		ToolChoice: map[string]interface{}{
			"type": "auto",
		},
	}

	params := convertToSDKParams(req)

	assert.NotNil(t, params.ToolChoice.OfAuto)
}

func TestConvertToSDKParams_Thinking(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-7-sonnet-20250219",
		MaxTokens: 16000,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		Thinking: &ReasoningSpec{
			Type:         "enabled",
			BudgetTokens: 8000,
		},
	}

	params := convertToSDKParams(req)

	require.NotNil(t, params.Thinking.OfEnabled)
	assert.Equal(t, int64(8000), params.Thinking.OfEnabled.BudgetTokens)
}

func TestConvertToSDKParams_ThinkingMinBudget(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-7-sonnet-20250219",
		MaxTokens: 16000,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		Thinking: &ReasoningSpec{
			Type:         "enabled",
			BudgetTokens: 100, // Below 1024 minimum
		},
	}

	params := convertToSDKParams(req)

	require.NotNil(t, params.Thinking.OfEnabled)
	assert.Equal(t, int64(1024), params.Thinking.OfEnabled.BudgetTokens)
}

func TestConvertToSDKParams_RequiredStringSlice(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		Tools: []Tool{
			{
				Name:        "test_tool",
				Description: "Test",
				InputSchema: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
					"required":   []string{"field1", "field2"},
				},
			},
		},
	}

	params := convertToSDKParams(req)

	require.Len(t, params.Tools, 1)
	assert.Equal(t, []string{"field1", "field2"}, params.Tools[0].OfTool.InputSchema.Required)
}

func TestConvertFromSDKMessage_TextResponse(t *testing.T) {
	msg := &sdkanthropic.Message{
		ID:         "msg_123",
		Model:      "claude-3-5-sonnet-latest",
		Role:       "assistant",
		StopReason: sdkanthropic.StopReasonEndTurn,
	}
	msg.Content = []sdkanthropic.ContentBlockUnion{
		{Type: "text", Text: "Hello, world!"},
	}
	msg.Usage = sdkanthropic.Usage{
		InputTokens:  10,
		OutputTokens: 5,
	}

	resp := convertFromSDKMessage(msg)

	assert.Equal(t, "msg_123", resp.ID)
	assert.Equal(t, "assistant", resp.Role)
	assert.Equal(t, "end_turn", resp.StopReason)
	require.Len(t, resp.Content, 1)
	assert.Equal(t, "text", resp.Content[0].Type)
	assert.Equal(t, "Hello, world!", resp.Content[0].Text)
	assert.Equal(t, 10, resp.Usage.InputTokens)
	assert.Equal(t, 5, resp.Usage.OutputTokens)
}

func TestConvertFromSDKMessage_ToolUseResponse(t *testing.T) {
	msg := &sdkanthropic.Message{
		ID:         "msg_456",
		Model:      "claude-3-5-sonnet-latest",
		Role:       "assistant",
		StopReason: sdkanthropic.StopReasonToolUse,
	}
	msg.Content = []sdkanthropic.ContentBlockUnion{
		{
			Type:  "tool_use",
			ID:    "toolu_123",
			Name:  "get_weather",
			Input: json.RawMessage(`{"location":"London"}`),
		},
	}
	msg.Usage = sdkanthropic.Usage{
		InputTokens:  15,
		OutputTokens: 8,
	}

	resp := convertFromSDKMessage(msg)

	assert.Equal(t, "tool_use", resp.StopReason)
	require.Len(t, resp.Content, 1)
	block := resp.Content[0]
	assert.Equal(t, "tool_use", block.Type)
	assert.Equal(t, "toolu_123", block.ID)
	assert.Equal(t, "get_weather", block.Name)
	assert.Equal(t, "London", block.Input["location"])
	require.NotNil(t, block.ToolUse)
	assert.Equal(t, "toolu_123", block.ToolUse.ID)
	assert.Equal(t, "get_weather", block.ToolUse.Name)
}

func TestConvertFromSDKMessage_ThinkingBlock(t *testing.T) {
	msg := &sdkanthropic.Message{
		ID:         "msg_789",
		Model:      "claude-3-7-sonnet-20250219",
		Role:       "assistant",
		StopReason: sdkanthropic.StopReasonEndTurn,
	}
	msg.Content = []sdkanthropic.ContentBlockUnion{
		{Type: "thinking", Thinking: "Let me think about this..."},
		{Type: "text", Text: "Here's my answer"},
	}
	msg.Usage = sdkanthropic.Usage{
		InputTokens:  20,
		OutputTokens: 15,
	}

	resp := convertFromSDKMessage(msg)

	require.Len(t, resp.Content, 2)
	assert.Equal(t, "thinking", resp.Content[0].Type)
	assert.Equal(t, "Let me think about this...", resp.Content[0].Text)
	assert.Equal(t, "text", resp.Content[1].Type)
	assert.Equal(t, "Here's my answer", resp.Content[1].Text)
}

func TestConvertFromSDKMessage_MixedContent(t *testing.T) {
	msg := &sdkanthropic.Message{
		ID:         "msg_mixed",
		Model:      "claude-3-5-sonnet-latest",
		Role:       "assistant",
		StopReason: sdkanthropic.StopReasonToolUse,
	}
	msg.Content = []sdkanthropic.ContentBlockUnion{
		{Type: "text", Text: "I'll check the weather for you."},
		{
			Type:  "tool_use",
			ID:    "toolu_abc",
			Name:  "get_weather",
			Input: json.RawMessage(`{"city":"Paris"}`),
		},
	}
	msg.Usage = sdkanthropic.Usage{
		InputTokens:  25,
		OutputTokens: 12,
	}

	resp := convertFromSDKMessage(msg)

	require.Len(t, resp.Content, 2)
	assert.Equal(t, "text", resp.Content[0].Type)
	assert.Equal(t, "I'll check the weather for you.", resp.Content[0].Text)
	assert.Equal(t, "tool_use", resp.Content[1].Type)
	assert.Equal(t, "get_weather", resp.Content[1].Name)
}

func TestConvertFromSDKMessage_CacheUsage(t *testing.T) {
	msg := &sdkanthropic.Message{
		ID:         "msg_cache",
		Model:      "claude-3-5-sonnet-latest",
		Role:       "assistant",
		StopReason: sdkanthropic.StopReasonEndTurn,
	}
	msg.Content = []sdkanthropic.ContentBlockUnion{
		{Type: "text", Text: "Cached response"},
	}
	msg.Usage = sdkanthropic.Usage{
		InputTokens:              10,
		OutputTokens:             5,
		CacheCreationInputTokens: 100,
		CacheReadInputTokens:     200,
	}

	resp := convertFromSDKMessage(msg)

	assert.Equal(t, 10, resp.Usage.InputTokens)
	assert.Equal(t, 5, resp.Usage.OutputTokens)
	assert.Equal(t, 100, resp.Usage.CacheCreationInputTokens)
	assert.Equal(t, 200, resp.Usage.CacheReadInputTokens)
}

func TestConvertSDKStreamEvent_MessageStart(t *testing.T) {
	event := sdkanthropic.MessageStreamEventUnion{
		Type: "message_start",
	}
	// We can't easily create a fully populated MessageStartEvent via the flat union,
	// but we can test the type routing.
	thinkingBlocks := make(map[int64]bool)
	toolBlocks := make(map[int64]*sdkToolBlockAccumulator)

	result := convertSDKStreamEvent(event, thinkingBlocks, toolBlocks)

	require.NotNil(t, result)
	assert.Equal(t, interfaces.StreamEventMessageStart, result.Type)
}

func TestConvertSDKStreamEvent_MessageStop(t *testing.T) {
	event := sdkanthropic.MessageStreamEventUnion{
		Type: "message_stop",
	}
	thinkingBlocks := make(map[int64]bool)
	toolBlocks := make(map[int64]*sdkToolBlockAccumulator)

	result := convertSDKStreamEvent(event, thinkingBlocks, toolBlocks)

	require.NotNil(t, result)
	assert.Equal(t, interfaces.StreamEventMessageStop, result.Type)
}

func TestConvertSDKStreamEvent_ContentBlockStartText(t *testing.T) {
	event := sdkanthropic.MessageStreamEventUnion{
		Type:  "content_block_start",
		Index: 0,
		ContentBlock: sdkanthropic.ContentBlockStartEventContentBlockUnion{
			Type: "text",
			Text: "",
		},
	}
	thinkingBlocks := make(map[int64]bool)
	toolBlocks := make(map[int64]*sdkToolBlockAccumulator)

	result := convertSDKStreamEvent(event, thinkingBlocks, toolBlocks)

	require.NotNil(t, result)
	assert.Equal(t, interfaces.StreamEventContentDelta, result.Type)
	assert.False(t, thinkingBlocks[0])
}

func TestConvertSDKStreamEvent_ContentBlockStartThinking(t *testing.T) {
	event := sdkanthropic.MessageStreamEventUnion{
		Type:  "content_block_start",
		Index: 0,
		ContentBlock: sdkanthropic.ContentBlockStartEventContentBlockUnion{
			Type: "thinking",
		},
	}
	thinkingBlocks := make(map[int64]bool)
	toolBlocks := make(map[int64]*sdkToolBlockAccumulator)

	result := convertSDKStreamEvent(event, thinkingBlocks, toolBlocks)

	require.NotNil(t, result)
	assert.Equal(t, interfaces.StreamEventThinking, result.Type)
	assert.True(t, thinkingBlocks[0])
}

func TestConvertSDKStreamEvent_ContentBlockStartToolUse(t *testing.T) {
	event := sdkanthropic.MessageStreamEventUnion{
		Type:  "content_block_start",
		Index: 1,
		ContentBlock: sdkanthropic.ContentBlockStartEventContentBlockUnion{
			Type: "tool_use",
			ID:   "toolu_123",
			Name: "get_weather",
		},
	}
	thinkingBlocks := make(map[int64]bool)
	toolBlocks := make(map[int64]*sdkToolBlockAccumulator)

	result := convertSDKStreamEvent(event, thinkingBlocks, toolBlocks)

	// Should return nil (tool events are deferred until complete)
	assert.Nil(t, result)
	require.Contains(t, toolBlocks, int64(1))
	assert.Equal(t, "toolu_123", toolBlocks[1].ID)
	assert.Equal(t, "get_weather", toolBlocks[1].Name)
}

func TestConvertSDKStreamEvent_ContentBlockDeltaText(t *testing.T) {
	event := sdkanthropic.MessageStreamEventUnion{
		Type:  "content_block_delta",
		Index: 0,
		Delta: sdkanthropic.MessageStreamEventUnionDelta{
			Type: "text_delta",
			Text: "Hello",
		},
	}
	thinkingBlocks := make(map[int64]bool)
	toolBlocks := make(map[int64]*sdkToolBlockAccumulator)

	result := convertSDKStreamEvent(event, thinkingBlocks, toolBlocks)

	require.NotNil(t, result)
	assert.Equal(t, interfaces.StreamEventContentDelta, result.Type)
	assert.Equal(t, "Hello", result.Content)
}

func TestConvertSDKStreamEvent_ContentBlockDeltaInputJSON(t *testing.T) {
	event := sdkanthropic.MessageStreamEventUnion{
		Type:  "content_block_delta",
		Index: 1,
		Delta: sdkanthropic.MessageStreamEventUnionDelta{
			Type:        "input_json_delta",
			PartialJSON: `{"location":`,
		},
	}
	thinkingBlocks := make(map[int64]bool)
	toolBlocks := make(map[int64]*sdkToolBlockAccumulator)
	toolBlocks[1] = &sdkToolBlockAccumulator{
		ID:   "toolu_123",
		Name: "get_weather",
	}

	result := convertSDKStreamEvent(event, thinkingBlocks, toolBlocks)

	// Should return nil (accumulating)
	assert.Nil(t, result)
	assert.Equal(t, `{"location":`, toolBlocks[1].InputJSON.String())
}

func TestConvertSDKStreamEvent_ContentBlockStopTool(t *testing.T) {
	thinkingBlocks := make(map[int64]bool)
	toolBlocks := make(map[int64]*sdkToolBlockAccumulator)
	toolBlocks[1] = &sdkToolBlockAccumulator{
		ID:   "toolu_123",
		Name: "get_weather",
	}
	toolBlocks[1].InputJSON.WriteString(`{"location":"London"}`)

	event := sdkanthropic.MessageStreamEventUnion{
		Type:  "content_block_stop",
		Index: 1,
	}

	result := convertSDKStreamEvent(event, thinkingBlocks, toolBlocks)

	require.NotNil(t, result)
	assert.Equal(t, interfaces.StreamEventToolUse, result.Type)
	require.NotNil(t, result.ToolCall)
	assert.Equal(t, "toolu_123", result.ToolCall.ID)
	assert.Equal(t, "get_weather", result.ToolCall.Name)
	assert.Equal(t, `{"location":"London"}`, result.ToolCall.Arguments)
	// Verify tool block was cleaned up
	assert.NotContains(t, toolBlocks, int64(1))
}

func TestConvertSDKStreamEvent_ContentBlockStopText(t *testing.T) {
	event := sdkanthropic.MessageStreamEventUnion{
		Type:  "content_block_stop",
		Index: 0,
	}
	thinkingBlocks := make(map[int64]bool)
	toolBlocks := make(map[int64]*sdkToolBlockAccumulator)

	result := convertSDKStreamEvent(event, thinkingBlocks, toolBlocks)

	require.NotNil(t, result)
	assert.Equal(t, interfaces.StreamEventContentComplete, result.Type)
}

func TestConvertSDKStreamEvent_UnknownType(t *testing.T) {
	event := sdkanthropic.MessageStreamEventUnion{
		Type: "unknown_event",
	}
	thinkingBlocks := make(map[int64]bool)
	toolBlocks := make(map[int64]*sdkToolBlockAccumulator)

	result := convertSDKStreamEvent(event, thinkingBlocks, toolBlocks)

	assert.Nil(t, result)
}

func TestConvertToolToSDK(t *testing.T) {
	tool := Tool{
		Name:        "calculator",
		Description: "Perform calculations",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"expression": map[string]interface{}{
					"type":        "string",
					"description": "Math expression",
				},
			},
			"required": []interface{}{"expression"},
		},
	}

	result := convertToolToSDK(tool)

	require.NotNil(t, result.OfTool)
	assert.Equal(t, "calculator", result.OfTool.Name)
	assert.Equal(t, "Perform calculations", result.OfTool.Description.Value)
	assert.Equal(t, []string{"expression"}, result.OfTool.InputSchema.Required)
}

func TestConvertToolChoiceToSDK_None(t *testing.T) {
	result := convertToolChoiceToSDK(map[string]string{"type": "none"})
	assert.NotNil(t, result.OfNone)
}

func TestConvertToolChoiceToSDK_UnknownType(t *testing.T) {
	// Unknown type defaults to auto
	result := convertToolChoiceToSDK("some_string")
	assert.NotNil(t, result.OfAuto)
}

func TestConvertToSDKParams_NoTemperature(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		// Temperature is 0 (zero value)
	}

	params := convertToSDKParams(req)

	// Temperature should not be set when it's 0
	assert.Equal(t, float64(0), params.Temperature.Value)
}

func TestConvertToSDKParams_AdaptiveThinking(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-opus-4-6",
		MaxTokens: 16000,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		Thinking: &ReasoningSpec{
			Type: "adaptive",
		},
	}

	// Adaptive thinking is NOT handled in convertToSDKParams (SDK v1.4.0 lacks OfAdaptive);
	// it's injected via extraRequestOptions using option.WithJSONSet.
	params := convertToSDKParams(req)
	assert.Nil(t, params.Thinking.OfEnabled) // Should NOT be set as "enabled"

	opts := extraRequestOptions(req)
	assert.Len(t, opts, 1) // One option for adaptive thinking
}

func TestConvertToSDKParams_Effort(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-opus-4-6",
		MaxTokens: 16000,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		OutputConfig: &OutputConfig{
			Effort: "medium",
		},
	}

	// Effort is handled via extraRequestOptions (SDK v1.4.0 lacks OutputConfigParam).
	opts := extraRequestOptions(req)
	assert.Len(t, opts, 1) // One option for effort
}

func TestConvertToSDKParams_AdaptiveThinkingWithEffort(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-opus-4-6",
		MaxTokens: 16000,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		Thinking: &ReasoningSpec{
			Type: "adaptive",
		},
		OutputConfig: &OutputConfig{
			Effort: "max",
		},
	}

	// Both adaptive thinking and effort are handled via extraRequestOptions.
	params := convertToSDKParams(req)
	assert.Nil(t, params.Thinking.OfEnabled)

	opts := extraRequestOptions(req)
	assert.Len(t, opts, 2) // One for adaptive thinking, one for effort
}

func TestExtraRequestOptions_NoOptions(t *testing.T) {
	req := &CompletionRequest{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 1024,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
	}

	opts := extraRequestOptions(req)
	assert.Len(t, opts, 0)
}

func TestExtraRequestOptions_EnabledThinkingNotIncluded(t *testing.T) {
	// "enabled" thinking is handled natively by convertToSDKParams, NOT extraRequestOptions
	req := &CompletionRequest{
		Model:     "claude-3-7-sonnet-20250219",
		MaxTokens: 16000,
		Messages:  []Message{{Role: "user", Content: "Hello"}},
		Thinking: &ReasoningSpec{
			Type:         "enabled",
			BudgetTokens: 8000,
		},
	}

	opts := extraRequestOptions(req)
	assert.Len(t, opts, 0) // "enabled" is NOT handled here
}

func TestSupportsAdaptiveThinking(t *testing.T) {
	tests := []struct {
		model    string
		expected bool
	}{
		// Opus 4.6 variants — should support adaptive thinking
		{"claude-opus-4-6", true},
		{"anthropic.claude-opus-4-6-v1:0", true},
		{"us.anthropic.claude-opus-4-6-v1:0", true},
		{"eu.anthropic.claude-opus-4-6-v1:0", true},
		{"claude-opus-4-6@latest", true},

		// Older models — should NOT support adaptive thinking
		{"claude-3-7-sonnet-20250219", false},
		{"claude-sonnet-4-20250514", false},
		{"claude-opus-4-20250514", false},
		{"claude-opus-4-1-20250805", false},
		{"claude-opus-4-5-20251101", false},
		{"claude-3-5-sonnet-latest", false},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			assert.Equal(t, tt.expected, SupportsAdaptiveThinking(tt.model))
		})
	}
}

func TestSupportsThinking_IncludesOpus46(t *testing.T) {
	// Opus 4.6 should also appear in SupportsThinking (it's a superset)
	assert.True(t, SupportsThinking("claude-opus-4-6"))
	assert.True(t, SupportsThinking("anthropic.claude-opus-4-6-v1:0"))
	assert.True(t, SupportsThinking("us.anthropic.claude-opus-4-6-v1:0"))
}
