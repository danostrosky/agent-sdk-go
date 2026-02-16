package anthropic

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/Ingenimax/agent-sdk-go/pkg/logging"
)

// messageHistoryBuilder builds Anthropic-compatible message history from memory and current prompt
type messageHistoryBuilder struct {
	logger logging.Logger
}

// newMessageHistoryBuilder creates a new message history builder
func newMessageHistoryBuilder(logger logging.Logger) *messageHistoryBuilder {
	return &messageHistoryBuilder{
		logger: logger,
	}
}

// buildMessages constructs Anthropic messages from memory and current prompt
// Returns messages ready for Anthropic API calls, preserving chronological order
func (b *messageHistoryBuilder) buildMessages(ctx context.Context, prompt string, params *interfaces.GenerateOptions) []Message {
	messages := []Message{}

	// Add memory messages
	if params.Memory != nil {
		memoryMessages, err := params.Memory.GetMessages(ctx)
		if err != nil {
			b.logger.Error(ctx, "Failed to retrieve memory messages", map[string]interface{}{
				"error": err.Error(),
			})
		} else {
			// Convert memory messages to Anthropic format, preserving chronological order
			for _, msg := range memoryMessages {
				anthropicMsg := b.convertMemoryMessage(msg)
				if anthropicMsg != nil {
					messages = append(messages, *anthropicMsg)
				}
			}
		}
	} else {
		// Only append current user message when memory is nil
		messages = append(messages, Message{
			Role:    "user",
			Content: prompt,
		})
	}

	return messages
}

// convertMemoryMessage converts a memory message to Anthropic format
func (b *messageHistoryBuilder) convertMemoryMessage(msg interfaces.Message) *Message {
	switch msg.Role {
	case interfaces.MessageRoleUser:
		return &Message{
			Role:    "user",
			Content: msg.Content,
		}

	case interfaces.MessageRoleAssistant:
		if len(msg.ToolCalls) > 0 {
			// Assistant message with tool calls — use ContentBlocks for proper API format
			var blocks []ContentBlock
			if msg.Content != "" {
				blocks = append(blocks, ContentBlock{Type: "text", Text: msg.Content})
			}
			for _, toolCall := range msg.ToolCalls {
				var args map[string]interface{}
				if err := json.Unmarshal([]byte(toolCall.Arguments), &args); err != nil {
					b.logger.Warn(context.Background(), "Failed to parse tool call arguments", map[string]interface{}{
						"error":     err.Error(),
						"arguments": toolCall.Arguments,
					})
					continue
				}
				if args == nil {
					args = map[string]interface{}{}
				}
				blocks = append(blocks, ContentBlock{
					Type:  "tool_use",
					ID:    toolCall.ID,
					Name:  toolCall.Name,
					Input: args,
				})
			}
			return &Message{Role: "assistant", ContentBlocks: blocks}
		} else if msg.Content != "" {
			// Regular assistant message
			return &Message{
				Role:    "assistant",
				Content: msg.Content,
			}
		}

	case interfaces.MessageRoleTool:
		// Tool messages in Anthropic are handled as tool_result content blocks
		if msg.ToolCallID != "" {
			return &Message{
				Role: "user",
				ContentBlocks: []ContentBlock{{
					Type:              "tool_result",
					ToolUseID:         msg.ToolCallID,
					ToolResultContent: msg.Content,
				}},
			}
		}

	case interfaces.MessageRoleSystem:
		return &Message{
			Role:    "user", // System instruction is handled separately, other system (like summarized) are passed as user messages
			Content: fmt.Sprintf("System: %s", msg.Content),
		}
	}

	return nil
}
