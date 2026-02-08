// Example: Braintrust evaluation
//
// This demonstrates running LLM evaluations using the Braintrust platform.
// Test cases are defined with inputs and expected outputs, then scored
// against actual LLM responses.
//
// Prerequisites:
//   - BRAINTRUST_API_KEY set
//   - OPENAI_API_KEY set (or use any other LLM provider)
//
// Usage:
//
//	export BRAINTRUST_API_KEY=your_key
//	export OPENAI_API_KEY=your_key
//	go run ./examples/evaluation/braintrust/
package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/Ingenimax/agent-sdk-go/pkg/evaluation"
	"github.com/Ingenimax/agent-sdk-go/pkg/llm/openai"
	"github.com/Ingenimax/agent-sdk-go/pkg/logging"
	"github.com/Ingenimax/agent-sdk-go/pkg/tracing"
)

func main() {
	logger := logging.New()
	ctx := context.Background()

	// 1. Create the Braintrust tracer
	btTracer, err := tracing.NewBraintrustTracer(tracing.BraintrustConfig{
		Enabled: true,
		APIKey:  os.Getenv("BRAINTRUST_API_KEY"),
		Project: getEnvOr("BRAINTRUST_PROJECT", "agent-sdk-go-eval"),
	})
	if err != nil {
		logger.Error(ctx, "Failed to create Braintrust tracer", map[string]interface{}{"error": err.Error()})
		os.Exit(1)
	}
	defer func() { _ = btTracer.Shutdown() }()

	// 2. Create an LLM client
	llm := openai.NewClient(os.Getenv("OPENAI_API_KEY"),
		openai.WithModel("gpt-4o-mini"),
	)

	// 3. Create an evaluator
	evaluator, err := evaluation.NewEvaluator(btTracer)
	if err != nil {
		logger.Error(ctx, "Failed to create evaluator", map[string]interface{}{"error": err.Error()})
		os.Exit(1)
	}

	// 4. Define test cases
	cases := []evaluation.EvalCase{
		{Input: "What is 2+2?", Expected: "4"},
		{Input: "What is the capital of France?", Expected: "Paris"},
		{Input: "Is the sky blue?", Expected: "yes"},
	}

	// 5. Define scorers
	scorers := map[string]evaluation.ScoreFunc{
		"contains_expected": func(_ context.Context, _, output, expected string) (float64, error) {
			if strings.Contains(strings.ToLower(output), strings.ToLower(expected)) {
				return 1.0, nil
			}
			return 0.0, nil
		},
	}

	// 6. Run evaluation
	fmt.Println("Running evaluation...")
	result, err := evaluator.Evaluate(ctx, llm, "qa-basics", cases, scorers)
	if err != nil {
		logger.Error(ctx, "Evaluation failed", map[string]interface{}{"error": err.Error()})
		os.Exit(1)
	}

	fmt.Println("Evaluation complete!")
	if result.ExperimentURL != "" {
		fmt.Printf("View results: %s\n", result.ExperimentURL)
	}
	fmt.Println("Check your Braintrust dashboard for detailed results.")
}

func getEnvOr(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}
