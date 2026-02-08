// Example: Braintrust tracing with Bedrock
//
// This demonstrates two layers of tracing:
//  1. SDK-level middleware that auto-captures Anthropic API request/response details
//  2. Application-level tracing via TracedLLM for agent-level observability
//
// Prerequisites:
//   - AWS credentials configured (e.g., via AWS_PROFILE or env vars)
//   - BRAINTRUST_API_KEY set
//   - BRAINTRUST_PROJECT set (optional, defaults to "default")
//
// Usage:
//
//	export BRAINTRUST_API_KEY=your_key
//	export BRAINTRUST_PROJECT=my-project
//	export AWS_REGION=us-east-1
//	go run ./examples/tracing/braintrust/
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/Ingenimax/agent-sdk-go/pkg/agent"
	"github.com/Ingenimax/agent-sdk-go/pkg/llm/anthropic"
	"github.com/Ingenimax/agent-sdk-go/pkg/logging"
	"github.com/Ingenimax/agent-sdk-go/pkg/memory"
	"github.com/Ingenimax/agent-sdk-go/pkg/tracing"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
)

func main() {
	logger := logging.New()
	ctx := context.Background()

	// 1. Create the Braintrust tracer
	btTracer, err := tracing.NewBraintrustTracer(tracing.BraintrustConfig{
		Enabled: true,
		APIKey:  os.Getenv("BRAINTRUST_API_KEY"),
		Project: getEnvOr("BRAINTRUST_PROJECT", "agent-sdk-go"),
	})
	if err != nil {
		logger.Error(ctx, "Failed to create Braintrust tracer", map[string]interface{}{"error": err.Error()})
		os.Exit(1)
	}
	defer func() { _ = btTracer.Shutdown() }()

	// 2. Load AWS config
	awsCfg, err := awsconfig.LoadDefaultConfig(ctx)
	if err != nil {
		logger.Error(ctx, "Failed to load AWS config", map[string]interface{}{"error": err.Error()})
		os.Exit(1)
	}

	// 3. Create Bedrock client with SDK-level Braintrust tracing middleware
	llm := anthropic.NewClient("",
		anthropic.WithModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
		anthropic.WithBedrockSDKOptions(
			tracing.BraintrustAnthropicMiddleware(btTracer.TracerProvider()),
		),
		anthropic.WithBedrockAWSConfig(awsCfg),
	)

	// 4. Wrap with application-level tracing
	tracedLLM := tracing.NewTracedLLM(llm, btTracer)

	// 5. Create an agent
	a, err := agent.NewAgent(
		agent.WithLLM(tracedLLM),
		agent.WithTracer(btTracer),
		agent.WithMemory(memory.NewConversationBuffer()),
		agent.WithSystemPrompt("You are a helpful AI assistant."),
	)
	if err != nil {
		logger.Error(ctx, "Failed to create agent", map[string]interface{}{"error": err.Error()})
		os.Exit(1)
	}

	fmt.Println("Agent with Braintrust tracing ready! Type 'exit' to quit.")
	fmt.Println("Traces will appear in your Braintrust dashboard.")
	fmt.Println()

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("You: ")
		query, _ := reader.ReadString('\n')
		query = strings.TrimSpace(query)
		if query == "" {
			continue
		}
		if query == "exit" {
			break
		}

		response, err := a.Run(ctx, query)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}
		fmt.Printf("Agent: %s\n\n", response)
	}

	fmt.Println("Flushing traces...")
	_ = btTracer.Flush()
	fmt.Println("Done! Check your Braintrust dashboard.")
}

func getEnvOr(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}
