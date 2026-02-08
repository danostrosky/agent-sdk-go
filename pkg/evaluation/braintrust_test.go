package evaluation

import (
	"context"
	"testing"

	"github.com/Ingenimax/agent-sdk-go/pkg/tracing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewEvaluator_NilTracer(t *testing.T) {
	_, err := NewEvaluator(nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "tracer is required")
}

func TestNewEvaluator_DisabledTracer(t *testing.T) {
	tracer, err := tracing.NewBraintrustTracer(tracing.BraintrustConfig{
		Enabled: false,
	})
	require.NoError(t, err)

	_, err = NewEvaluator(tracer)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "active client")
}

func TestEvaluate_EmptyCases(t *testing.T) {
	// We can't create a real evaluator without a Braintrust API key,
	// but we can test validation
	tracer, _ := tracing.NewBraintrustTracer(tracing.BraintrustConfig{Enabled: false})
	eval := &Evaluator{tracer: tracer}

	_, err := eval.Evaluate(context.TODO(), nil, "test", nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least one eval case")
}

func TestEvaluate_EmptyScorers(t *testing.T) {
	tracer, _ := tracing.NewBraintrustTracer(tracing.BraintrustConfig{Enabled: false})
	eval := &Evaluator{tracer: tracer}

	cases := []EvalCase{{Input: "hello", Expected: "world"}}
	_, err := eval.Evaluate(context.TODO(), nil, "test", cases, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least one scorer")
}
