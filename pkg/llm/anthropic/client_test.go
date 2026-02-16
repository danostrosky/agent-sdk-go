package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
)

func TestMessageFiltering(t *testing.T) {

	tests := []struct {
		name            string
		messages        []interfaces.Message
		expectedCount   int
		expectedContent []string
	}{
		{
			name: "Filter out messages with empty content",
			messages: []interfaces.Message{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: ""}, // Should be filtered out
				{Role: "user", Content: "   "},   // Should be filtered out (whitespace only)
				{Role: "assistant", Content: "World"},
			},
			expectedCount:   2,
			expectedContent: []string{"Hello", "World"},
		},
		{
			name: "Filter out messages with empty role",
			messages: []interfaces.Message{
				{Role: "", Content: "Should be filtered"},
				{Role: "user", Content: "Should stay"},
			},
			expectedCount:   1,
			expectedContent: []string{"Should stay"},
		},
		{
			name: "Keep valid messages only",
			messages: []interfaces.Message{
				{Role: "user", Content: "First message"},
				{Role: "assistant", Content: "Second message"},
				{Role: "user", Content: "Third message"},
			},
			expectedCount:   3,
			expectedContent: []string{"First message", "Second message", "Third message"},
		},
		{
			name: "Filter out all invalid messages",
			messages: []interfaces.Message{
				{Role: "", Content: ""},
				{Role: "user", Content: ""},
				{Role: "", Content: "Some content"},
				{Role: "assistant", Content: "   "},
			},
			expectedCount:   0,
			expectedContent: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Convert to Anthropic messages format
			anthropicMessages := make([]Message, len(tt.messages))
			for i, msg := range tt.messages {
				role := msg.Role
				if role == "system" {
					continue // System messages are handled separately
				}
				anthropicMessages[i] = Message{
					Role:    string(role),
					Content: msg.Content,
				}
			}

			// Apply the filtering logic from the actual code (uses HasContent())
			var filteredMessages []Message
			for _, msg := range anthropicMessages {
				if msg.Role != "" && msg.HasContent() {
					filteredMessages = append(filteredMessages, msg)
				}
			}

			// Check the count
			if len(filteredMessages) != tt.expectedCount {
				t.Errorf("Expected %d messages, got %d", tt.expectedCount, len(filteredMessages))
			}

			// Check the content
			for i, msg := range filteredMessages {
				if i < len(tt.expectedContent) && msg.Content != tt.expectedContent[i] {
					t.Errorf("Expected message %d content %q, got %q", i, tt.expectedContent[i], msg.Content)
				}
			}
		})
	}
}

func TestEmptyContentHandling(t *testing.T) {
	tests := []struct {
		name         string
		content      string
		shouldFilter bool
	}{
		{"Normal content", "Hello world", false},
		{"Empty string", "", true},
		{"Whitespace only", "   ", true},
		{"Tab only", "\t", true},
		{"Newline only", "\n", true},
		{"Mixed whitespace", " \t\n ", true},
		{"Content with whitespace", " Hello ", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := Message{
				Role:    "user",
				Content: tt.content,
			}

			// Apply filtering condition using HasContent()
			shouldKeep := msg.Role != "" && msg.HasContent()
			shouldFilter := !shouldKeep

			if shouldFilter != tt.shouldFilter {
				t.Errorf("Content %q: expected shouldFilter=%v, got shouldFilter=%v",
					tt.content, tt.shouldFilter, shouldFilter)
			}
		})
	}
}

func TestMessageMarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		message  Message
		expected string
	}{
		{
			name:     "String content",
			message:  Message{Role: "user", Content: "hello"},
			expected: `{"role":"user","content":"hello"}`,
		},
		{
			name: "ContentBlocks with tool_use",
			message: Message{
				Role: "assistant",
				ContentBlocks: []ContentBlock{
					{Type: "text", Text: "Let me call a tool"},
					{Type: "tool_use", ID: "toolu_123", Name: "search", Input: map[string]interface{}{"q": "test"}},
				},
			},
			expected: `{"role":"assistant","content":[{"type":"text","text":"Let me call a tool"},{"type":"tool_use","id":"toolu_123","name":"search","input":{"q":"test"}}]}`,
		},
		{
			name: "ContentBlocks with tool_result",
			message: Message{
				Role: "user",
				ContentBlocks: []ContentBlock{
					{Type: "tool_result", ToolUseID: "toolu_123", ToolResultContent: "result data"},
				},
			},
			expected: `{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_123","content":"result data"}]}`,
		},
		{
			name:     "Empty content string",
			message:  Message{Role: "user", Content: ""},
			expected: `{"role":"user","content":""}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.message)
			if err != nil {
				t.Fatalf("MarshalJSON failed: %v", err)
			}
			if string(data) != tt.expected {
				t.Errorf("Expected:\n%s\nGot:\n%s", tt.expected, string(data))
			}
		})
	}
}

func TestMessageUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name           string
		input          string
		expectedRole   string
		hasBlocks      bool
		hasContent     bool
		expectedText   string
		expectedBlocks int
	}{
		{
			name:         "String content",
			input:        `{"role":"user","content":"hello"}`,
			expectedRole: "user",
			hasContent:   true,
			expectedText: "hello",
		},
		{
			name:           "Array content with tool_use",
			input:          `{"role":"assistant","content":[{"type":"text","text":"hi"},{"type":"tool_use","id":"t1","name":"search"}]}`,
			expectedRole:   "assistant",
			hasBlocks:      true,
			expectedBlocks: 2,
		},
		{
			name:         "Empty content",
			input:        `{"role":"user","content":""}`,
			expectedRole: "user",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var msg Message
			err := json.Unmarshal([]byte(tt.input), &msg)
			if err != nil {
				t.Fatalf("UnmarshalJSON failed: %v", err)
			}
			if msg.Role != tt.expectedRole {
				t.Errorf("Expected role %q, got %q", tt.expectedRole, msg.Role)
			}
			if tt.hasBlocks && len(msg.ContentBlocks) != tt.expectedBlocks {
				t.Errorf("Expected %d blocks, got %d", tt.expectedBlocks, len(msg.ContentBlocks))
			}
			if tt.hasContent && msg.Content != tt.expectedText {
				t.Errorf("Expected content %q, got %q", tt.expectedText, msg.Content)
			}
		})
	}
}

func TestMessageHasContent(t *testing.T) {
	tests := []struct {
		name     string
		message  Message
		expected bool
	}{
		{"String content", Message{Content: "hello"}, true},
		{"Empty string", Message{Content: ""}, false},
		{"Whitespace only", Message{Content: "   "}, false},
		{"ContentBlocks", Message{ContentBlocks: []ContentBlock{{Type: "text", Text: "hi"}}}, true},
		{"Empty blocks", Message{ContentBlocks: []ContentBlock{}}, false},
		{"Both set", Message{Content: "hi", ContentBlocks: []ContentBlock{{Type: "text"}}}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.message.HasContent(); got != tt.expected {
				t.Errorf("HasContent() = %v, want %v", got, tt.expected)
			}
		})
	}
}
