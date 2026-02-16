//go:build integration

package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Run with: go test -tags=integration -run TestBedrock -v ./pkg/llm/anthropic/...
// Requires the "pde" AWS profile to be configured:
//   aws configure list --profile pde

const (
	bedrockTestProfile = "pde"
	bedrockTestRegion  = "us-east-1"
	// Use a smaller, faster model for integration tests
	bedrockTestModel = "us." + BedrockClaudeSonnet45
)

func newBedrockTestClient(t *testing.T) *AnthropicClient {
	t.Helper()

	ctx := context.Background()
	awsCfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion(bedrockTestRegion),
		config.WithSharedConfigProfile(bedrockTestProfile),
	)
	require.NoError(t, err, "Failed to load AWS config with profile %q", bedrockTestProfile)

	client := NewClient("",
		WithModel(bedrockTestModel),
		WithBedrockAWSConfig(awsCfg),
	)
	require.NotNil(t, client.BedrockConfig, "BedrockConfig should be set")
	require.True(t, client.BedrockConfig.Enabled, "Bedrock should be enabled")

	return client
}

// --- Non-streaming tests ---

func TestBedrockIntegration_Generate(t *testing.T) {
	client := newBedrockTestClient(t)
	ctx := context.Background()

	resp, err := client.Generate(ctx, "Say exactly: hello world")
	require.NoError(t, err)
	assert.NotEmpty(t, resp)
	t.Logf("Response: %s", resp)
}

func TestBedrockIntegration_GenerateWithSystemMessage(t *testing.T) {
	client := newBedrockTestClient(t)
	ctx := context.Background()

	resp, err := client.Generate(ctx, "What are you?",
		interfaces.WithSystemMessage("You are a pirate. Always respond with pirate language."),
	)
	require.NoError(t, err)
	assert.NotEmpty(t, resp)
	t.Logf("Response: %s", resp)
}

func TestBedrockIntegration_Chat(t *testing.T) {
	client := newBedrockTestClient(t)
	ctx := context.Background()

	resp, err := client.Generate(ctx, "What is 2+2? Reply with just the number.")
	require.NoError(t, err)
	assert.Contains(t, resp, "4")
	t.Logf("Response: %s", resp)
}

// --- Streaming tests ---

func TestBedrockIntegration_GenerateStream(t *testing.T) {
	client := newBedrockTestClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	eventChan, err := client.GenerateStream(ctx, "Count from 1 to 5, one number per line.")
	require.NoError(t, err)

	var fullResponse strings.Builder
	var eventTypes []interfaces.StreamEventType
	eventCount := 0

	for event := range eventChan {
		eventCount++
		eventTypes = append(eventTypes, event.Type)

		switch event.Type {
		case interfaces.StreamEventContentDelta:
			fullResponse.WriteString(event.Content)
			t.Logf("Delta: %q", event.Content)
		case interfaces.StreamEventError:
			t.Fatalf("Stream error: %v", event.Error)
		case interfaces.StreamEventMessageStart:
			t.Log("Message started")
		case interfaces.StreamEventMessageStop:
			t.Log("Message stopped")
		case interfaces.StreamEventThinking:
			t.Logf("Thinking: %q", event.Content)
		}
	}

	resp := fullResponse.String()
	t.Logf("Full response (%d events): %s", eventCount, resp)
	assert.NotEmpty(t, resp, "Should have received content")
	assert.Greater(t, eventCount, 0, "Should have received events")
}

func TestBedrockIntegration_GenerateStreamLong(t *testing.T) {
	client := newBedrockTestClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	eventChan, err := client.GenerateStream(ctx, "Write a short paragraph about Go programming language.")
	require.NoError(t, err)

	var fullResponse strings.Builder
	var gotStart, gotStop bool
	deltaCount := 0

	for event := range eventChan {
		switch event.Type {
		case interfaces.StreamEventMessageStart:
			gotStart = true
		case interfaces.StreamEventContentDelta:
			deltaCount++
			fullResponse.WriteString(event.Content)
		case interfaces.StreamEventMessageStop:
			gotStop = true
		case interfaces.StreamEventError:
			t.Fatalf("Stream error: %v", event.Error)
		}
	}

	resp := fullResponse.String()
	t.Logf("Response (%d deltas): %s", deltaCount, resp)

	assert.True(t, gotStart, "Should have received message_start")
	assert.True(t, gotStop, "Should have received message_stop")
	assert.NotEmpty(t, resp, "Should have received content")
	assert.Greater(t, deltaCount, 1, "Should have received multiple deltas")
}

// --- Non-streaming with tools ---

func TestBedrockIntegration_GenerateWithTools(t *testing.T) {
	client := newBedrockTestClient(t)
	ctx := context.Background()

	// Create a simple mock tool
	tools := []interfaces.Tool{
		&mockTool{
			name:        "get_weather",
			description: "Get the current weather for a location",
			parameters: map[string]interfaces.ParameterSpec{
				"location": {
					Type:        "string",
					Description: "The city name",
					Required:    true,
				},
			},
			executeFn: func(ctx context.Context, args string) (string, error) {
				return `{"temperature": "72F", "conditions": "sunny"}`, nil
			},
		},
	}

	resp, err := client.GenerateWithTools(ctx, "What's the weather in San Francisco?", tools)
	require.NoError(t, err)
	assert.NotEmpty(t, resp)
	t.Logf("Response: %s", resp)
}

// --- Direct SDK call test (bypasses client layer) ---

func TestBedrockIntegration_DirectSDKNonStreaming(t *testing.T) {
	ctx := context.Background()

	awsCfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion(bedrockTestRegion),
		config.WithSharedConfigProfile(bedrockTestProfile),
	)
	require.NoError(t, err)

	bc, err := NewBedrockConfigWithAWSConfig(ctx, awsCfg)
	require.NoError(t, err)

	req := &CompletionRequest{
		Model:     bedrockTestModel,
		MaxTokens: 256,
		Messages: []Message{
			{Role: "user", Content: "Say hello."},
		},
	}

	msg, err := bc.InvokeModel(ctx, bedrockTestModel, req, nil)
	require.NoError(t, err)
	require.NotNil(t, msg)

	t.Logf("ID: %s", msg.ID)
	t.Logf("StopReason: %s", msg.StopReason)
	t.Logf("Usage: input=%d output=%d", msg.Usage.InputTokens, msg.Usage.OutputTokens)
	require.NotEmpty(t, msg.Content)
	t.Logf("Content[0].Type: %s", msg.Content[0].Type)
	t.Logf("Content[0].Text: %s", msg.Content[0].Text)
}

func TestBedrockIntegration_DirectSDKStreaming(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	awsCfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion(bedrockTestRegion),
		config.WithSharedConfigProfile(bedrockTestProfile),
	)
	require.NoError(t, err)

	bc, err := NewBedrockConfigWithAWSConfig(ctx, awsCfg)
	require.NoError(t, err)

	req := &CompletionRequest{
		Model:     bedrockTestModel,
		MaxTokens: 256,
		Messages: []Message{
			{Role: "user", Content: "Say hello."},
		},
	}

	output, err := bc.InvokeModelStream(ctx, bedrockTestModel, req, nil)
	require.NoError(t, err)
	require.NotNil(t, output)

	stream := output.GetStream()
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			t.Logf("Warning: failed to close stream: %v", closeErr)
		}
	}()

	var eventTypes []string
	var contentBuilder strings.Builder
	eventCount := 0

	for event := range stream.Events() {
		switch e := event.(type) {
		case *types.ResponseStreamMemberChunk:
			// Parse the raw JSON from the chunk
			var rawEvent struct {
				Type    string `json:"type"`
				Message struct {
					ID    string `json:"id"`
					Model string `json:"model"`
				} `json:"message"`
				Delta struct {
					Type       string `json:"type"`
					Text       string `json:"text"`
					StopReason string `json:"stop_reason"`
				} `json:"delta"`
			}
			if err := json.Unmarshal(e.Value.Bytes, &rawEvent); err != nil {
				t.Logf("Failed to parse chunk: %v", err)
				continue
			}

			eventCount++
			eventTypes = append(eventTypes, rawEvent.Type)
			t.Logf("Event %d: type=%s", eventCount, rawEvent.Type)

			switch rawEvent.Type {
			case "content_block_delta":
				if rawEvent.Delta.Type == "text_delta" {
					contentBuilder.WriteString(rawEvent.Delta.Text)
					t.Logf("  text_delta: %q", rawEvent.Delta.Text)
				}
			case "message_start":
				t.Logf("  message: id=%s model=%s", rawEvent.Message.ID, rawEvent.Message.Model)
			case "message_delta":
				t.Logf("  stop_reason=%s", rawEvent.Delta.StopReason)
			}
		default:
			t.Logf("Unknown stream event type: %T", e)
		}
	}

	if err := stream.Err(); err != nil {
		t.Fatalf("Stream error after %d events: %v\nEvent types seen: %v\nContent so far: %q",
			eventCount, err, eventTypes, contentBuilder.String())
	}

	content := contentBuilder.String()
	t.Logf("Total events: %d", eventCount)
	t.Logf("Event types: %v", eventTypes)
	t.Logf("Content: %s", content)

	assert.Greater(t, eventCount, 0, "Should have received events")
	assert.NotEmpty(t, content, "Should have received text content")
	assert.Contains(t, eventTypes, "message_start")
	assert.Contains(t, eventTypes, "message_stop")
}

// --- Streaming event conversion end-to-end test ---

func TestBedrockIntegration_StreamEventConversion(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	awsCfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion(bedrockTestRegion),
		config.WithSharedConfigProfile(bedrockTestProfile),
	)
	require.NoError(t, err)

	bc, err := NewBedrockConfigWithAWSConfig(ctx, awsCfg)
	require.NoError(t, err)

	req := &CompletionRequest{
		Model:     bedrockTestModel,
		MaxTokens: 256,
		Messages: []Message{
			{Role: "user", Content: "Say exactly: test"},
		},
	}

	output, err := bc.InvokeModelStream(ctx, bedrockTestModel, req, nil)
	require.NoError(t, err)

	stream := output.GetStream()
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			t.Logf("Warning: failed to close stream: %v", closeErr)
		}
	}()

	// Use the same conversion approach as executeBedrockStreaming
	client := NewClient("", WithModel(bedrockTestModel))
	thinkingBlocks := make(map[int]*thinkingBlockTracker)
	toolBlocks := make(map[int]struct {
		ID        string
		Name      string
		InputJSON strings.Builder
	})

	var events []interfaces.StreamEvent
	for event := range stream.Events() {
		switch e := event.(type) {
		case *types.ResponseStreamMemberChunk:
			var rawEvent map[string]interface{}
			if err := json.Unmarshal(e.Value.Bytes, &rawEvent); err != nil {
				t.Logf("Failed to parse chunk: %v", err)
				continue
			}

			eventType, ok := rawEvent["type"].(string)
			if !ok {
				continue
			}

			anthropicEvent := &AnthropicSSEEvent{
				Type: eventType,
				Data: e.Value.Bytes,
			}

			streamEvent, err := client.convertAnthropicEventToStreamEvent(anthropicEvent, thinkingBlocks, toolBlocks)
			if err != nil {
				t.Logf("Conversion error for %s: %v", eventType, err)
				continue
			}

			if streamEvent != nil {
				events = append(events, *streamEvent)
				t.Logf("Converted: type=%s content=%q", streamEvent.Type, truncate(streamEvent.Content, 80))
			}
		}
	}

	if err := stream.Err(); err != nil {
		t.Fatalf("Unexpected stream error: %v", err)
	}
	assert.NotEmpty(t, events, "Should have produced converted events")

	// Check we got the expected event flow
	var eventTypesList []interfaces.StreamEventType
	for _, e := range events {
		eventTypesList = append(eventTypesList, e.Type)
	}
	t.Logf("Event types: %v", eventTypesList)

	assert.Equal(t, interfaces.StreamEventMessageStart, eventTypesList[0], "Should start with message_start")
	assert.Equal(t, interfaces.StreamEventMessageStop, eventTypesList[len(eventTypesList)-1], "Should end with message_stop")
}

// --- Adaptive thinking streaming with Opus 4.6 ---

const bedrockTestModelOpus46 = "us.anthropic.claude-opus-4-6-v1"

func TestBedrockIntegration_AdaptiveStreamingOpus46(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	awsCfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion(bedrockTestRegion),
		config.WithSharedConfigProfile(bedrockTestProfile),
	)
	require.NoError(t, err)

	client := NewClient("",
		WithModel(bedrockTestModelOpus46),
		WithBedrockAWSConfig(awsCfg),
	)
	require.NotNil(t, client.BedrockConfig)
	require.True(t, client.BedrockConfig.Enabled)

	eventChan, err := client.GenerateStream(ctx, "What is ((2+2)*51/43) + 214/12 Reply briefly.",
		interfaces.WithReasoning(true),
	)
	require.NoError(t, err)

	var fullResponse strings.Builder
	var gotStart, gotStop bool
	var gotThinking bool
	deltaCount := 0

	for event := range eventChan {
		switch event.Type {
		case interfaces.StreamEventMessageStart:
			gotStart = true
			t.Log("Message started")
		case interfaces.StreamEventThinking:
			gotThinking = true
			t.Logf("Thinking: %q", truncate(event.Content, 120))
		case interfaces.StreamEventContentDelta:
			deltaCount++
			fullResponse.WriteString(event.Content)
			t.Logf("Delta: %q", event.Content)
		case interfaces.StreamEventMessageStop:
			gotStop = true
			t.Log("Message stopped")
		case interfaces.StreamEventError:
			t.Fatalf("Stream error: %v", event.Error)
		default:
			t.Logf("Other event: type=%s", event.Type)
		}
	}

	resp := fullResponse.String()
	t.Logf("Full response (%d deltas, thinking=%v): %s", deltaCount, gotThinking, resp)

	assert.True(t, gotStart, "Should have received message_start")
	assert.True(t, gotStop, "Should have received message_stop")
	assert.NotEmpty(t, resp, "Should have received content")
	assert.Greater(t, deltaCount, 0, "Should have received at least one delta")
	// Adaptive thinking may or may not produce thinking blocks depending on query complexity
	t.Logf("Got thinking blocks: %v", gotThinking)
}

// Test that verifies thinking blocks AND signatures are captured from Bedrock streaming
func TestBedrockIntegration_ThinkingSignatureCapture(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	awsCfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion(bedrockTestRegion),
		config.WithSharedConfigProfile(bedrockTestProfile),
	)
	require.NoError(t, err)

	bc, err := NewBedrockConfigWithAWSConfig(ctx, awsCfg)
	require.NoError(t, err)

	req := &CompletionRequest{
		Model:     bedrockTestModelOpus46,
		MaxTokens: 16000,
		Messages: []Message{
			{Role: "user", Content: "What is 17 * 23? Think step by step."},
		},
		Thinking: &ReasoningSpec{
			Type:         "enabled",
			BudgetTokens: 10000,
		},
		Temperature: 1.0,
	}

	output, err := bc.InvokeModelStream(ctx, bedrockTestModelOpus46, req, nil)
	require.NoError(t, err)

	stream := output.GetStream()
	defer stream.Close()

	client := NewClient("", WithModel(bedrockTestModelOpus46))
	thinkingBlocks := make(map[int]*thinkingBlockTracker)
	toolBlocks := make(map[int]struct {
		ID        string
		Name      string
		InputJSON strings.Builder
	})

	var eventTypes []string
	var thinkingText strings.Builder
	var contentText strings.Builder
	var gotSignatureDelta bool
	var completedThinkingText string
	var completedThinkingSignature string

	for event := range stream.Events() {
		switch e := event.(type) {
		case *types.ResponseStreamMemberChunk:
			// Log raw event type for debugging
			var rawEvent map[string]interface{}
			if err := json.Unmarshal(e.Value.Bytes, &rawEvent); err != nil {
				continue
			}
			eventType, _ := rawEvent["type"].(string)

			// Check for signature_delta in raw events
			if eventType == "content_block_delta" {
				if delta, ok := rawEvent["delta"].(map[string]interface{}); ok {
					deltaType, _ := delta["type"].(string)
					if deltaType == "signature_delta" {
						gotSignatureDelta = true
						t.Logf("Got signature_delta! length=%d", len(fmt.Sprintf("%v", delta["signature"])))
					}
				}
			}

			anthropicEvent := &AnthropicSSEEvent{
				Type: eventType,
				Data: e.Value.Bytes,
			}

			streamEvent, err := client.convertAnthropicEventToStreamEvent(anthropicEvent, thinkingBlocks, toolBlocks)
			if err != nil {
				t.Logf("Conversion error for %s: %v", eventType, err)
				continue
			}

			if streamEvent != nil {
				eventTypes = append(eventTypes, string(streamEvent.Type))

				switch streamEvent.Type {
				case interfaces.StreamEventThinking:
					thinkingText.WriteString(streamEvent.Content)
				case interfaces.StreamEventContentDelta:
					contentText.WriteString(streamEvent.Content)
				case interfaces.StreamEventContentComplete:
					// Check if this is a completed thinking block
					if bt, _ := streamEvent.Metadata["block_type"].(string); bt == "thinking" {
						completedThinkingText, _ = streamEvent.Metadata["thinking_text"].(string)
						completedThinkingSignature, _ = streamEvent.Metadata["thinking_signature"].(string)
						t.Logf("Thinking block complete: text_len=%d signature_len=%d",
							len(completedThinkingText), len(completedThinkingSignature))
					}
				}
			}
		}
	}

	require.NoError(t, stream.Err())

	t.Logf("Event types seen: %v", eventTypes)
	t.Logf("Thinking text length: %d", thinkingText.Len())
	t.Logf("Content text: %s", contentText.String())
	t.Logf("Got signature_delta events: %v", gotSignatureDelta)
	t.Logf("Completed thinking text length: %d", len(completedThinkingText))
	t.Logf("Completed thinking signature length: %d", len(completedThinkingSignature))

	assert.NotEmpty(t, thinkingText.String(), "Should have received thinking content")
	assert.NotEmpty(t, contentText.String(), "Should have received text content")
	assert.True(t, gotSignatureDelta, "Bedrock should send signature_delta events")
	assert.NotEmpty(t, completedThinkingSignature, "Should have captured thinking signature")
}

// Test streaming with thinking + tools (the agent loop scenario)
func TestBedrockIntegration_ThinkingWithToolsStreaming(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	awsCfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion(bedrockTestRegion),
		config.WithSharedConfigProfile(bedrockTestProfile),
	)
	require.NoError(t, err)

	client := NewClient("",
		WithModel(bedrockTestModelOpus46),
		WithBedrockAWSConfig(awsCfg),
	)

	tools := []interfaces.Tool{
		&mockTool{
			name:        "get_weather",
			description: "Get the current weather for a location",
			parameters: map[string]interfaces.ParameterSpec{
				"location": {
					Type:        "string",
					Description: "The city name",
					Required:    true,
				},
			},
			executeFn: func(ctx context.Context, args string) (string, error) {
				return `{"temperature": "72F", "conditions": "sunny"}`, nil
			},
		},
	}

	eventChan, err := client.GenerateWithToolsStream(ctx,
		"What's the weather in San Francisco? Use the tool.",
		tools,
		interfaces.WithReasoning(true, 10000),
	)
	require.NoError(t, err)

	var fullResponse strings.Builder
	var gotThinking, gotToolUse, gotToolResult bool
	thinkingCount := 0

	for event := range eventChan {
		switch event.Type {
		case interfaces.StreamEventThinking:
			if !gotThinking {
				t.Logf("First thinking chunk: %q", truncate(event.Content, 100))
			}
			gotThinking = true
			thinkingCount++
		case interfaces.StreamEventContentDelta:
			fullResponse.WriteString(event.Content)
		case interfaces.StreamEventToolUse:
			gotToolUse = true
			t.Logf("Tool call: %s(%s)", event.ToolCall.Name, truncate(event.ToolCall.Arguments, 100))
		case interfaces.StreamEventToolResult:
			gotToolResult = true
			t.Logf("Tool result: %s", truncate(event.Content, 100))
		case interfaces.StreamEventError:
			t.Fatalf("Stream error: %v", event.Error)
		}
	}

	resp := fullResponse.String()
	t.Logf("Response: %s", resp)
	t.Logf("Thinking chunks: %d, Tool use: %v, Tool result: %v", thinkingCount, gotToolUse, gotToolResult)

	assert.True(t, gotThinking, "Should have received thinking blocks")
	assert.True(t, gotToolUse, "Should have called the weather tool")
	assert.True(t, gotToolResult, "Should have received tool result")
	assert.NotEmpty(t, resp, "Should have a final response")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// --- mockTool for tool tests ---

type mockTool struct {
	name        string
	description string
	parameters  map[string]interfaces.ParameterSpec
	executeFn   func(ctx context.Context, args string) (string, error)
}

func (m *mockTool) Name() string                                    { return m.name }
func (m *mockTool) Description() string                             { return m.description }
func (m *mockTool) Parameters() map[string]interfaces.ParameterSpec { return m.parameters }
func (m *mockTool) Run(ctx context.Context, input string) (string, error) {
	return m.executeFn(ctx, input)
}
func (m *mockTool) Execute(ctx context.Context, args string) (string, error) {
	return m.executeFn(ctx, args)
}
func (m *mockTool) String() string { return fmt.Sprintf("mockTool(%s)", m.name) }

// --- Prompt caching tests ---

// bedrockGeneratePadding generates padding text to meet minimum cache token requirements.
// Anthropic requires at least 1024 tokens (Haiku) or 2048 tokens (larger models) for caching.
func bedrockGeneratePadding(words int) string {
	padding := ""
	for i := 0; i < words; i++ {
		padding += fmt.Sprintf("word%d ", i)
	}
	return padding
}

func TestBedrockIntegration_CacheSystemMessage(t *testing.T) {
	client := newBedrockTestClient(t)
	ctx := context.Background()

	// Long system message with timestamp to ensure fresh cache
	longSystemMessage := fmt.Sprintf(`You are an expert AI assistant. Session: %d
`, time.Now().UnixNano()) + bedrockGeneratePadding(2000)

	// First call - should create cache
	t.Log("Making first request (should create cache)...")
	resp1, err := client.GenerateDetailed(ctx, "What is 2+2? Answer briefly.",
		interfaces.WithSystemMessage(longSystemMessage),
		WithCacheSystemMessage(),
	)
	require.NoError(t, err)
	require.NotNil(t, resp1.Usage)

	t.Logf("First call - Cache creation: %d, Cache read: %d",
		resp1.Usage.CacheCreationInputTokens,
		resp1.Usage.CacheReadInputTokens)

	assert.Greater(t, resp1.Usage.CacheCreationInputTokens, 0, "First call should create cache")

	// Second call - should read from cache
	t.Log("Making second request (should read from cache)...")
	resp2, err := client.GenerateDetailed(ctx, "What is 3+3? Answer briefly.",
		interfaces.WithSystemMessage(longSystemMessage),
		WithCacheSystemMessage(),
	)
	require.NoError(t, err)
	require.NotNil(t, resp2.Usage)

	t.Logf("Second call - Cache creation: %d, Cache read: %d",
		resp2.Usage.CacheCreationInputTokens,
		resp2.Usage.CacheReadInputTokens)

	assert.Greater(t, resp2.Usage.CacheReadInputTokens, 0, "Second call should read from cache")
}

func TestBedrockIntegration_CacheStream(t *testing.T) {
	client := newBedrockTestClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Long system message with timestamp to ensure fresh cache
	longSystemMessage := fmt.Sprintf(`You are an expert AI assistant. Session: %d
`, time.Now().UnixNano()) + bedrockGeneratePadding(2000)

	// First call - should create cache (streaming)
	t.Log("Making first streaming request (should create cache)...")
	eventChan1, err := client.GenerateStream(ctx, "What is 2+2? Answer briefly.",
		interfaces.WithSystemMessage(longSystemMessage),
		WithCacheSystemMessage(),
	)
	require.NoError(t, err)

	// Drain the first stream
	for event := range eventChan1 {
		if event.Type == interfaces.StreamEventError {
			t.Fatalf("Stream error on first call: %v", event.Error)
		}
	}
	t.Log("First streaming request completed")

	// Second call - should read from cache (streaming)
	t.Log("Making second streaming request (should read from cache)...")
	eventChan2, err := client.GenerateStream(ctx, "What is 3+3? Answer briefly.",
		interfaces.WithSystemMessage(longSystemMessage),
		WithCacheSystemMessage(),
	)
	require.NoError(t, err)

	var fullResponse strings.Builder
	for event := range eventChan2 {
		if event.Type == interfaces.StreamEventContentDelta {
			fullResponse.WriteString(event.Content)
		}
		if event.Type == interfaces.StreamEventError {
			t.Fatalf("Stream error on second call: %v", event.Error)
		}
	}

	resp := fullResponse.String()
	t.Logf("Second streaming response: %s", resp)
	assert.NotEmpty(t, resp, "Should have received content from cached streaming request")
}
