package anthropic

import (
	"context"
	"fmt"

	"github.com/Ingenimax/agent-sdk-go/pkg/logging"
	sdkanthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
	"github.com/anthropics/anthropic-sdk-go/packages/ssestream"
	"github.com/aws/aws-sdk-go-v2/aws"
)

// BedrockConfig contains configuration for AWS Bedrock
// This mirrors the VertexConfig structure for consistency
type BedrockConfig struct {
	Enabled bool
	Region  string

	// Internal fields
	awsConfig aws.Config
	sdkClient *sdkanthropic.Client
	logger    logging.Logger
}

// NewBedrockConfigWithAWSConfig creates a new BedrockConfig from an existing AWS config
// This is the primary way to configure Bedrock - users configure credentials and settings
// through the aws.Config itself using config.LoadDefaultConfig() or other AWS SDK methods
func NewBedrockConfigWithAWSConfig(ctx context.Context, awsConfig aws.Config) (*BedrockConfig, error) {
	if awsConfig.Region == "" {
		return nil, fmt.Errorf("region is required in AWS config")
	}

	// Create Anthropic SDK client with Bedrock middleware
	sdkClient := sdkanthropic.NewClient(
		bedrock.WithConfig(awsConfig),
	)

	bedrockConfig := &BedrockConfig{
		Enabled:   true,
		Region:    awsConfig.Region,
		awsConfig: awsConfig,
		sdkClient: &sdkClient,
		logger:    logging.New(),
	}

	bedrockConfig.logger.Info(ctx, "Configured AWS Bedrock with official SDK", map[string]interface{}{
		"region": awsConfig.Region,
	})

	return bedrockConfig, nil
}

// InvokeModel invokes a Bedrock model using the official Anthropic SDK (non-streaming)
func (bc *BedrockConfig) InvokeModel(ctx context.Context, modelID string, params sdkanthropic.MessageNewParams) (*sdkanthropic.Message, error) {
	if !bc.Enabled {
		return nil, fmt.Errorf("bedrock is not enabled")
	}

	// Set the model on the params
	params.Model = sdkanthropic.Model(modelID)

	bc.logger.Debug(ctx, "Invoking Bedrock model via SDK", map[string]interface{}{
		"modelID": modelID,
		"region":  bc.Region,
	})

	msg, err := bc.sdkClient.Messages.New(ctx, params)
	if err != nil {
		bc.logger.Error(ctx, "Failed to invoke Bedrock model", map[string]interface{}{
			"error":   err.Error(),
			"modelID": modelID,
			"region":  bc.Region,
		})
		return nil, fmt.Errorf("failed to invoke Bedrock model: %w", err)
	}

	bc.logger.Debug(ctx, "Successfully received response from Bedrock", map[string]interface{}{
		"modelID":      modelID,
		"stopReason":   msg.StopReason,
		"inputTokens":  msg.Usage.InputTokens,
		"outputTokens": msg.Usage.OutputTokens,
	})

	return msg, nil
}

// InvokeModelStream invokes a Bedrock model with streaming using the official Anthropic SDK
func (bc *BedrockConfig) InvokeModelStream(ctx context.Context, modelID string, params sdkanthropic.MessageNewParams) (*ssestream.Stream[sdkanthropic.MessageStreamEventUnion], error) {
	if !bc.Enabled {
		return nil, fmt.Errorf("bedrock is not enabled")
	}

	// Set the model on the params
	params.Model = sdkanthropic.Model(modelID)

	bc.logger.Debug(ctx, "Invoking Bedrock model with streaming via SDK", map[string]interface{}{
		"modelID": modelID,
		"region":  bc.Region,
	})

	stream := bc.sdkClient.Messages.NewStreaming(ctx, params)

	// Check for immediate errors
	if stream.Err() != nil {
		bc.logger.Error(ctx, "Failed to invoke Bedrock model with streaming", map[string]interface{}{
			"error":   stream.Err().Error(),
			"modelID": modelID,
			"region":  bc.Region,
		})
		return nil, fmt.Errorf("failed to invoke Bedrock model with streaming: %w", stream.Err())
	}

	return stream, nil
}
