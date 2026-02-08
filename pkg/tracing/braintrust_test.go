package tracing

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBraintrustTracer_Disabled(t *testing.T) {
	tracer, err := NewBraintrustTracer(BraintrustConfig{
		Enabled: false,
	})
	require.NoError(t, err)
	assert.NotNil(t, tracer)
	assert.Nil(t, tracer.TracerProvider())
	assert.Nil(t, tracer.Client())

	// Disabled tracer should return no-op spans
	ctx, span := tracer.StartSpan(context.Background(), "test")
	assert.NotNil(t, ctx)
	assert.NotNil(t, span)
	span.End() // should not panic

	ctx2, span2 := tracer.StartTraceSession(context.Background(), "session-1")
	assert.NotNil(t, ctx2)
	assert.NotNil(t, span2)
	span2.End()

	// Flush/Shutdown should be no-ops
	assert.NoError(t, tracer.Flush())
	assert.NoError(t, tracer.Shutdown())
}

func TestBraintrustTracer_MissingAPIKey(t *testing.T) {
	_, err := NewBraintrustTracer(BraintrustConfig{
		Enabled: true,
		APIKey:  "",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "API key is required")
}

func TestBraintrustConfig_FromEnv(t *testing.T) {
	// When no custom config is provided, it should read from the global config
	// which defaults to disabled
	tracer, err := NewBraintrustTracer()
	require.NoError(t, err)
	assert.NotNil(t, tracer)
	// Default config has Enabled=false
	assert.Nil(t, tracer.TracerProvider())
}

func TestBraintrustSpan_Methods(t *testing.T) {
	// Test span methods don't panic with a disabled tracer
	tracer, err := NewBraintrustTracer(BraintrustConfig{Enabled: false})
	require.NoError(t, err)

	ctx, span := tracer.StartSpan(context.Background(), "test-span")
	assert.NotNil(t, ctx)
	assert.NotNil(t, span)

	// These should not panic
	span.SetAttribute("key", "value")
	span.AddEvent("event", map[string]interface{}{"attr": "val"})
	span.RecordError(nil)
	span.End()
}
