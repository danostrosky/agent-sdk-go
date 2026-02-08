package tracing

import (
	traceanthropic "github.com/braintrustdata/braintrust-sdk-go/trace/contrib/anthropic"
	"go.opentelemetry.io/otel/sdk/trace"

	"github.com/anthropics/anthropic-sdk-go/option"
)

// BraintrustAnthropicMiddleware creates an option.RequestOption that instruments
// Anthropic SDK calls with Braintrust tracing. Pass the TracerProvider from a
// BraintrustTracer to automatically capture API request/response details.
//
// Usage:
//
//	btTracer, _ := tracing.NewBraintrustTracer(tracing.BraintrustConfig{...})
//	llm := anthropic.NewClient("",
//	    anthropic.WithBedrockSDKOptions(
//	        tracing.BraintrustAnthropicMiddleware(btTracer.TracerProvider()),
//	    ),
//	    anthropic.WithBedrockAWSConfig(awsCfg),
//	)
func BraintrustAnthropicMiddleware(tp *trace.TracerProvider) option.RequestOption {
	return option.WithMiddleware(
		traceanthropic.NewMiddleware(traceanthropic.WithTracerProvider(tp)),
	)
}
