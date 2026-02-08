// Package evaluation provides an AI evaluation framework built on Braintrust.
// It wraps the Braintrust eval SDK to provide a convenient integration with
// agent-sdk-go's LLM interfaces.
package evaluation

import (
	"context"
	"fmt"

	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/Ingenimax/agent-sdk-go/pkg/tracing"
	"github.com/braintrustdata/braintrust-sdk-go"
	"github.com/braintrustdata/braintrust-sdk-go/eval"
)

// Evaluator runs evaluation cases against an LLM using Braintrust.
type Evaluator struct {
	tracer *tracing.BraintrustTracer
}

// NewEvaluator creates a new Evaluator backed by a BraintrustTracer.
// The tracer must be enabled and have an active client.
func NewEvaluator(tracer *tracing.BraintrustTracer) (*Evaluator, error) {
	if tracer == nil {
		return nil, fmt.Errorf("braintrust tracer is required")
	}
	if tracer.Client() == nil {
		return nil, fmt.Errorf("braintrust tracer must have an active client (is tracing enabled?)")
	}
	return &Evaluator{tracer: tracer}, nil
}

// EvalCase represents a single evaluation test case.
type EvalCase struct {
	Input    string
	Expected string
	Tags     []string
}

// ScoreFunc is a function that scores an LLM output against the expected result.
// It returns a score between 0.0 and 1.0.
type ScoreFunc func(ctx context.Context, input, output, expected string) (float64, error)

// EvalResult holds the result of an evaluation run.
type EvalResult struct {
	ExperimentURL string
}

// Evaluate runs a set of test cases against an LLM and reports results to Braintrust.
//
// The scorer functions receive (ctx, input, output, expected) and return a score in [0, 1].
//
// Example:
//
//	result, err := evaluator.Evaluate(ctx, myLLM, "qa-eval", cases,
//	    map[string]evaluation.ScoreFunc{
//	        "exact_match": func(_ context.Context, _, output, expected string) (float64, error) {
//	            if output == expected { return 1.0, nil }
//	            return 0.0, nil
//	        },
//	    },
//	)
func (e *Evaluator) Evaluate(
	ctx context.Context,
	llm interfaces.LLM,
	experiment string,
	cases []EvalCase,
	scorers map[string]ScoreFunc,
) (*EvalResult, error) {
	if len(cases) == 0 {
		return nil, fmt.Errorf("at least one eval case is required")
	}
	if len(scorers) == 0 {
		return nil, fmt.Errorf("at least one scorer is required")
	}

	// Convert cases to Braintrust eval format
	btCases := make([]eval.Case[string, string], len(cases))
	for i, c := range cases {
		btCases[i] = eval.Case[string, string]{
			Input:    c.Input,
			Expected: c.Expected,
			Tags:     c.Tags,
		}
	}

	// Convert scorers
	btScorers := make([]eval.Scorer[string, string], 0, len(scorers))
	for name, fn := range scorers {
		scoreFn := fn // capture
		btScorers = append(btScorers, eval.NewScorer(
			name,
			func(ctx context.Context, result eval.TaskResult[string, string]) (eval.Scores, error) {
				score, err := scoreFn(ctx, result.Input, result.Output, result.Expected)
				if err != nil {
					return nil, err
				}
				return eval.S(score), nil
			},
		))
	}

	// Create the task function that calls the LLM
	task := eval.T(func(ctx context.Context, input string) (string, error) {
		return llm.Generate(ctx, input)
	})

	// Run evaluation via Braintrust
	btEvaluator := braintrust.NewEvaluator[string, string](e.tracer.Client())
	_, err := btEvaluator.Run(ctx, eval.Opts[string, string]{
		Experiment: experiment,
		Dataset:    eval.NewDataset(btCases),
		Task:       task,
		Scorers:    btScorers,
	})
	if err != nil {
		return nil, fmt.Errorf("evaluation failed: %w", err)
	}

	return &EvalResult{}, nil
}
