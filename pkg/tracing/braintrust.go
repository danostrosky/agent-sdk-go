package tracing

import (
	"context"
	"fmt"
	"time"

	"github.com/Ingenimax/agent-sdk-go/pkg/config"
	"github.com/Ingenimax/agent-sdk-go/pkg/interfaces"
	"github.com/Ingenimax/agent-sdk-go/pkg/multitenancy"
	"github.com/braintrustdata/braintrust-sdk-go"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// BraintrustConfig contains configuration for Braintrust tracing
type BraintrustConfig struct {
	// Enabled determines whether Braintrust tracing is enabled
	Enabled bool

	// APIKey is the Braintrust API key
	APIKey string

	// Project is the Braintrust project name
	Project string

	// APIURL is the optional Braintrust API URL override
	APIURL string
}

// BraintrustTracer implements tracing using Braintrust via OpenTelemetry
type BraintrustTracer struct {
	tracerProvider *sdktrace.TracerProvider
	tracer         trace.Tracer
	client         *braintrust.Client
	enabled        bool
	config         BraintrustConfig
}

// BraintrustSpan wraps an OTEL span to implement the interfaces.Span interface
type BraintrustSpan struct {
	span trace.Span
}

// End implements interfaces.Span
func (s *BraintrustSpan) End() {
	s.span.End()
}

// AddEvent implements interfaces.Span
func (s *BraintrustSpan) AddEvent(name string, attributes map[string]interface{}) {
	attrs := make([]attribute.KeyValue, 0, len(attributes))
	for k, v := range attributes {
		attrs = append(attrs, attribute.String(k, fmt.Sprintf("%v", v)))
	}
	s.span.AddEvent(name, trace.WithAttributes(attrs...))
}

// SetAttribute implements interfaces.Span
func (s *BraintrustSpan) SetAttribute(key string, value interface{}) {
	s.span.SetAttributes(attribute.String(key, fmt.Sprintf("%v", value)))
}

// RecordError implements interfaces.Span
func (s *BraintrustSpan) RecordError(err error) {
	s.span.RecordError(err)
}

// NewBraintrustTracer creates a new Braintrust tracer
func NewBraintrustTracer(customConfig ...BraintrustConfig) (*BraintrustTracer, error) {
	cfg := config.Get()

	var tracerConfig BraintrustConfig
	if len(customConfig) > 0 {
		tracerConfig = customConfig[0]
	} else {
		tracerConfig = BraintrustConfig{
			Enabled: cfg.Tracing.Braintrust.Enabled,
			APIKey:  cfg.Tracing.Braintrust.APIKey,
			Project: cfg.Tracing.Braintrust.Project,
			APIURL:  cfg.Tracing.Braintrust.APIURL,
		}
	}

	if !tracerConfig.Enabled {
		return &BraintrustTracer{
			enabled: false,
		}, nil
	}

	if tracerConfig.APIKey == "" {
		return nil, fmt.Errorf("braintrust API key is required")
	}

	// Create a dedicated TracerProvider for Braintrust (does not set global provider)
	tp := sdktrace.NewTracerProvider()

	// Build Braintrust client options
	opts := []braintrust.Option{
		braintrust.WithAPIKey(tracerConfig.APIKey),
	}
	if tracerConfig.Project != "" {
		opts = append(opts, braintrust.WithProject(tracerConfig.Project))
	}
	if tracerConfig.APIURL != "" {
		opts = append(opts, braintrust.WithAPIURL(tracerConfig.APIURL))
	}

	// Create the Braintrust client which registers its span processor on the TracerProvider
	btClient, err := braintrust.New(tp, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create braintrust client: %w", err)
	}

	tracer := tp.Tracer("agent-sdk-go")

	return &BraintrustTracer{
		tracerProvider: tp,
		tracer:         tracer,
		client:         btClient,
		enabled:        true,
		config:         tracerConfig,
	}, nil
}

// StartSpan implements interfaces.Tracer
func (t *BraintrustTracer) StartSpan(ctx context.Context, name string) (context.Context, interfaces.Span) {
	if !t.enabled {
		return ctx, &BraintrustSpan{span: trace.SpanFromContext(ctx)}
	}

	orgID, _ := multitenancy.GetOrgID(ctx)

	attrs := []attribute.KeyValue{}
	if orgID != "" {
		attrs = append(attrs, attribute.String("org_id", orgID))
	}

	ctx, span := t.tracer.Start(ctx, name, trace.WithAttributes(attrs...))
	return ctx, &BraintrustSpan{span: span}
}

// StartTraceSession implements interfaces.Tracer
func (t *BraintrustTracer) StartTraceSession(ctx context.Context, contextID string) (context.Context, interfaces.Span) {
	if !t.enabled {
		return ctx, &BraintrustSpan{span: trace.SpanFromContext(ctx)}
	}

	orgID, _ := multitenancy.GetOrgID(ctx)

	attrs := []attribute.KeyValue{
		attribute.String("trace.session_id", contextID),
	}
	if orgID != "" {
		attrs = append(attrs, attribute.String("org_id", orgID))
	}

	ctx, span := t.tracer.Start(ctx, "trace-session", trace.WithAttributes(attrs...))
	return ctx, &BraintrustSpan{span: span}
}

// TracerProvider returns the OTEL TracerProvider used by Braintrust.
// Use this to create SDK-level middleware for auto-instrumenting Anthropic API calls.
func (t *BraintrustTracer) TracerProvider() *sdktrace.TracerProvider {
	return t.tracerProvider
}

// Client returns the underlying Braintrust client.
// Use this for evaluation workflows.
func (t *BraintrustTracer) Client() *braintrust.Client {
	return t.client
}

// Flush flushes all pending spans
func (t *BraintrustTracer) Flush() error {
	if !t.enabled || t.tracerProvider == nil {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return t.tracerProvider.ForceFlush(ctx)
}

// Shutdown shuts down the tracer provider
func (t *BraintrustTracer) Shutdown() error {
	if !t.enabled || t.tracerProvider == nil {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return t.tracerProvider.Shutdown(ctx)
}
