package tools

import (
	"context"
	"testing"
	"time"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/stretchr/testify/assert"
)

// MockStreamingAgent implements the SubAgent interface with streaming support
type MockStreamingAgent struct {
	name        string
	description string
}

func (m *MockStreamingAgent) Run(ctx context.Context, input string) (string, error) {
	return "mock result: " + input, nil
}

func (m *MockStreamingAgent) RunDetailed(ctx context.Context, input string) (*interfaces.AgentResponse, error) {
	return &interfaces.AgentResponse{
		Content:   "mock result: " + input,
		AgentName: m.name,
		Model:     "mock-model",
	}, nil
}

func (m *MockStreamingAgent) GetName() string {
	return m.name
}

func (m *MockStreamingAgent) GetDescription() string {
	return m.description
}

func (m *MockStreamingAgent) RunStream(ctx context.Context, input string) (<-chan interfaces.AgentStreamEvent, error) {
	eventChan := make(chan interfaces.AgentStreamEvent, 10)
	
	go func() {
		defer close(eventChan)
		
		// Send a content event
		eventChan <- interfaces.AgentStreamEvent{
			Type:      interfaces.AgentEventContent,
			Content:   "streaming response for: " + input,
			Timestamp: time.Now(),
		}
		
		// Send completion event
		eventChan <- interfaces.AgentStreamEvent{
			Type:      interfaces.AgentEventComplete,
			Timestamp: time.Now(),
		}
	}()
	
	return eventChan, nil
}

func TestAgentToolSupportsStreaming(t *testing.T) {
	mockAgent := &MockStreamingAgent{
		name:        "TestAgent",
		description: "A test streaming agent",
	}
	
	agentTool := NewAgentTool(mockAgent)
	
	// AgentTool should support streaming
	assert.True(t, agentTool.SupportsStreaming(), "AgentTool should support streaming")
}

func TestAgentToolRunStream(t *testing.T) {
	mockAgent := &MockStreamingAgent{
		name:        "TestAgent",
		description: "A test streaming agent",
	}
	
	agentTool := NewAgentTool(mockAgent)
	
	ctx := context.Background()
	input := "test query"
	
	eventChan, err := agentTool.RunStream(ctx, input)
	assert.NoError(t, err, "RunStream should not error")
	assert.NotNil(t, eventChan, "Event channel should not be nil")
	
	// Collect events
	var events []interfaces.AgentStreamEvent
	for event := range eventChan {
		events = append(events, event)
	}
	
	// Should have at least 2 events (content + complete)
	assert.GreaterOrEqual(t, len(events), 2, "Should have at least 2 events")
	
	// First event should be content
	assert.Equal(t, interfaces.AgentEventContent, events[0].Type)
	assert.Contains(t, events[0].Content, input)
	
	// Check that subagent metadata is added
	assert.NotNil(t, events[0].Metadata)
	assert.Equal(t, "TestAgent", events[0].Metadata["subagent_name"])
	assert.Equal(t, 1, events[0].Metadata["subagent_depth"])
}

func TestAgentToolAsStreamingToolInterface(t *testing.T) {
	mockAgent := &MockStreamingAgent{
		name:        "TestAgent",
		description: "A test streaming agent",
	}
	
	agentTool := NewAgentTool(mockAgent)
	
	// Check if AgentTool implements StreamingTool interface
	var _ interfaces.StreamingTool = agentTool
	
	// Test through interface
	var tool interfaces.Tool = agentTool
	if streamingTool, ok := tool.(interfaces.StreamingTool); ok {
		assert.True(t, streamingTool.SupportsStreaming())
	} else {
		t.Fatal("AgentTool should implement StreamingTool interface")
	}
}

