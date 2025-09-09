# torch::gru

## Overview
Creates a Gated Recurrent Unit (GRU) layer for processing sequential data. GRU is a simpler alternative to LSTM that combines the forget and input gates into a single update gate, making it computationally more efficient while maintaining similar performance for many sequence modeling tasks.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::gru -input_size SIZE -hidden_size SIZE ?-num_layers N? ?-bias BOOL? ?-batch_first BOOL? ?-dropout RATE? ?-bidirectional BOOL?
torch::gru -inputSize SIZE -hiddenSize SIZE ?-numLayers N? ?-bias BOOL? ?-batchFirst BOOL? ?-dropout RATE? ?-bidirectional BOOL?
```

### Positional Syntax (Legacy)
```tcl
torch::gru INPUT_SIZE HIDDEN_SIZE ?NUM_LAYERS? ?BIAS? ?BATCH_FIRST? ?DROPOUT? ?BIDIRECTIONAL?
```

### camelCase Alias
```tcl
torch::Gru -input_size SIZE -hidden_size SIZE ?-num_layers N? ?-bias BOOL? ?-batch_first BOOL? ?-dropout RATE? ?-bidirectional BOOL?
torch::Gru -inputSize SIZE -hiddenSize SIZE ?-numLayers N? ?-bias BOOL? ?-batchFirst BOOL? ?-dropout RATE? ?-bidirectional BOOL?
```

## Parameters

### Required Parameters
- **`-input_size`** or **`-inputSize`**: Size of input features
  - Type: Integer
  - Range: > 0
  - The number of expected features in the input tensor

- **`-hidden_size`** or **`-hiddenSize`**: Size of hidden state
  - Type: Integer
  - Range: > 0
  - The number of features in the hidden state

### Optional Parameters
- **`-num_layers`** or **`-numLayers`**: Number of recurrent layers
  - Type: Integer
  - Default: `1`
  - Range: > 0
  - Stacked GRU layers for deeper networks

- **`-bias`**: Whether to use bias parameters
  - Type: Boolean
  - Default: `true`
  - If `false`, the layer will not use bias parameters

- **`-batch_first`** or **`-batchFirst`**: Input/output tensor format
  - Type: Boolean
  - Default: `false`
  - If `true`, input/output tensors are `(batch, seq, feature)` instead of `(seq, batch, feature)`

- **`-dropout`**: Dropout probability
  - Type: Float
  - Default: `0.0`
  - Range: [0.0, 1.0]
  - Dropout applied to outputs of each GRU layer except the last layer

- **`-bidirectional`**: Bidirectional GRU
  - Type: Boolean
  - Default: `false`
  - If `true`, creates a bidirectional GRU

## Return Value
Returns a GRU layer handle that can be used with `torch::layer_forward` or other layer operations.

## Examples

### Basic GRU Creation
```tcl
# Create a basic GRU layer
set gru [torch::gru -input_size 10 -hidden_size 20]
puts "GRU layer created: $gru"

# Same using positional syntax
set gru_pos [torch::gru 10 20]
puts "GRU layer (positional): $gru_pos"
```

### GRU with Multiple Layers
```tcl
# Create a 3-layer GRU
set deep_gru [torch::gru -inputSize 50 -hiddenSize 100 -numLayers 3]
puts "Deep GRU created: $deep_gru"

# Same using positional syntax
set deep_gru_pos [torch::gru 50 100 3]
puts "Deep GRU (positional): $deep_gru_pos"
```

### GRU with Dropout
```tcl
# Create GRU with dropout for regularization
set gru_dropout [torch::gru -input_size 128 -hidden_size 256 -num_layers 2 -dropout 0.3]
puts "GRU with dropout: $gru_dropout"

# Same using positional syntax
set gru_dropout_pos [torch::gru 128 256 2 1 0 0.3]
puts "GRU with dropout (positional): $gru_dropout_pos"
```

### Bidirectional GRU
```tcl
# Create bidirectional GRU
set bi_gru [torch::gru -inputSize 64 -hiddenSize 128 -bidirectional 1]
puts "Bidirectional GRU: $bi_gru"

# Bidirectional GRU with multiple layers
set bi_deep_gru [torch::gru -inputSize 32 -hiddenSize 64 -numLayers 2 -bidirectional 1 -dropout 0.2]
puts "Bidirectional deep GRU: $bi_deep_gru"
```

### Batch-First GRU
```tcl
# Create GRU with batch-first input format
set batch_first_gru [torch::gru -input_size 100 -hidden_size 200 -batch_first 1]
puts "Batch-first GRU: $batch_first_gru"

# Same using camelCase
set batch_first_gru2 [torch::gru -inputSize 100 -hiddenSize 200 -batchFirst 1]
puts "Batch-first GRU (camelCase): $batch_first_gru2"
```

### Using camelCase Alias
```tcl
# Using camelCase alias
set gru [torch::Gru -inputSize 40 -hiddenSize 80 -numLayers 2]
puts "GRU (camelCase alias): $gru"

# Mixed parameter styles
set gru_mixed [torch::Gru -input_size 40 -hiddenSize 80 -num_layers 2 -batchFirst 1]
puts "GRU (mixed style): $gru_mixed"
```

### Complex GRU Configuration
```tcl
# All parameters specified
set complex_gru [torch::gru -inputSize 256 -hiddenSize 512 -numLayers 4 -bias 1 -batchFirst 1 -dropout 0.5 -bidirectional 1]
puts "Complex GRU: $complex_gru"

# Same using positional syntax
set complex_gru_pos [torch::gru 256 512 4 1 1 0.5 1]
puts "Complex GRU (positional): $complex_gru_pos"
```

### Parameter Variations
```tcl
# All equivalent ways to create the same GRU
set gru1 [torch::gru -input_size 10 -hidden_size 20 -num_layers 2]
set gru2 [torch::gru -inputSize 10 -hiddenSize 20 -numLayers 2]
set gru3 [torch::Gru -input_size 10 -hidden_size 20 -num_layers 2]
set gru4 [torch::Gru -inputSize 10 -hiddenSize 20 -numLayers 2]
set gru5 [torch::gru 10 20 2]
```

## GRU Architecture Explained

### How GRU Works
The GRU uses two gates to control information flow:
- **Update Gate**: Decides how much of the previous hidden state to keep
- **Reset Gate**: Determines how much of the previous hidden state to forget

### Mathematical Formulation
For each time step:
```
r_t = σ(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)  # Reset gate
z_t = σ(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)  # Update gate
n_t = tanh(W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  # New gate
h_t = (1 - z_t) * n_t + z_t * h_{t-1}  # New hidden state
```

Where:
- `σ` is the sigmoid function
- `*` denotes element-wise multiplication
- `@` denotes matrix multiplication
- `W_*` are weight matrices
- `b_*` are bias vectors

### Input/Output Shapes
```tcl
# For batch_first=False (default):
# Input: (seq_len, batch, input_size)
# Output: (seq_len, batch, num_directions * hidden_size)
# Hidden: (num_layers * num_directions, batch, hidden_size)

# For batch_first=True:
# Input: (batch, seq_len, input_size)
# Output: (batch, seq_len, num_directions * hidden_size)
# Hidden: (num_layers * num_directions, batch, hidden_size)
```

## Common Use Cases

### Natural Language Processing
```tcl
# Text classification
set text_gru [torch::gru -inputSize 300 -hiddenSize 128 -numLayers 2 -dropout 0.3 -batchFirst 1]

# Sentiment analysis
set sentiment_gru [torch::gru -inputSize 100 -hiddenSize 64 -bidirectional 1]

# Language modeling
set lm_gru [torch::gru -inputSize 512 -hiddenSize 512 -numLayers 3 -dropout 0.2]
```

### Speech Recognition
```tcl
# Acoustic modeling
set acoustic_gru [torch::gru -inputSize 40 -hiddenSize 256 -numLayers 4 -dropout 0.3 -bidirectional 1]

# Feature extraction
set feature_gru [torch::gru -inputSize 13 -hiddenSize 128 -numLayers 2 -bidirectional 1]
```

### Time Series Analysis
```tcl
# Stock price prediction
set stock_gru [torch::gru -inputSize 5 -hiddenSize 50 -numLayers 2 -dropout 0.1]

# Weather forecasting
set weather_gru [torch::gru -inputSize 10 -hiddenSize 64 -numLayers 3 -bidirectional 1]
```

### Encoder-Decoder Architectures
```tcl
# Encoder
set encoder_gru [torch::gru -inputSize 256 -hiddenSize 512 -numLayers 2 -dropout 0.2 -bidirectional 1]

# Decoder
set decoder_gru [torch::gru -inputSize 256 -hiddenSize 512 -numLayers 2 -dropout 0.2]
```

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::gru} msg
puts $msg  ;# "Required parameters missing or invalid"

# Insufficient parameters
catch {torch::gru 10} msg
puts $msg  ;# "Usage: torch::gru input_size hidden_size ..."

# Invalid input_size
catch {torch::gru -input_size 0 -hidden_size 20} msg
puts $msg  ;# "Required parameters missing or invalid"

# Invalid hidden_size
catch {torch::gru -input_size 10 -hidden_size -1} msg
puts $msg  ;# "Required parameters missing or invalid"

# Invalid num_layers
catch {torch::gru -input_size 10 -hidden_size 20 -num_layers 0} msg
puts $msg  ;# "Required parameters missing or invalid"

# Invalid dropout
catch {torch::gru -input_size 10 -hidden_size 20 -dropout -0.1} msg
puts $msg  ;# "Required parameters missing or invalid"

# Non-numeric parameters
catch {torch::gru -input_size "not_a_number" -hidden_size 20} msg
puts $msg  ;# "Invalid inputSize value"

# Non-boolean parameters
catch {torch::gru -input_size 10 -hidden_size 20 -bias "not_a_boolean"} msg
puts $msg  ;# "Invalid bias value"

# Unknown parameter
catch {torch::gru -input_size 10 -hidden_size 20 -unknown_param value} msg
puts $msg  ;# "Unknown parameter: -unknown_param"
```

## Performance Considerations

### Memory Usage
- **Single-layer GRU**: ~3 × input_size × hidden_size parameters
- **Multi-layer GRU**: Each additional layer adds ~3 × hidden_size² parameters
- **Bidirectional GRU**: Doubles the number of parameters

### Speed vs. Accuracy Trade-offs
- **Fewer layers**: Faster training/inference, may underfit complex patterns
- **More layers**: Better representation learning, slower and may overfit
- **Dropout**: Reduces overfitting but increases training time

### Typical Configurations
```tcl
# Fast but simple
set fast_gru [torch::gru -inputSize 64 -hiddenSize 64 -numLayers 1]

# Balanced
set balanced_gru [torch::gru -inputSize 128 -hiddenSize 128 -numLayers 2 -dropout 0.2]

# High capacity
set complex_gru [torch::gru -inputSize 256 -hiddenSize 512 -numLayers 4 -dropout 0.3 -bidirectional 1]
```

## Comparison with LSTM

### GRU Advantages
- **Simpler architecture**: Only 2 gates vs. 3 in LSTM
- **Faster training**: Fewer parameters to update
- **Less memory usage**: Smaller parameter footprint
- **Good performance**: Often matches LSTM performance

### When to Use GRU
- Limited computational resources
- Shorter sequences
- Simpler sequence patterns
- When training time is critical

### When to Use LSTM
- Complex long-term dependencies
- Very long sequences
- When you need fine-grained control over memory

## Migration Guide

### From Positional to Named Syntax

```tcl
# Old positional syntax
set gru [torch::gru 10 20]
set gru [torch::gru 10 20 2 1 1 0.2 1]

# New named syntax
set gru [torch::gru -input_size 10 -hidden_size 20]
set gru [torch::gru -inputSize 10 -hiddenSize 20 -numLayers 2 -bias 1 -batchFirst 1 -dropout 0.2 -bidirectional 1]
```

### Parameter Mapping

| Position | Named | Alternative | Default |
|----------|-------|-------------|---------|
| 1 | `-input_size` | `-inputSize` | (required) |
| 2 | `-hidden_size` | `-hiddenSize` | (required) |
| 3 | `-num_layers` | `-numLayers` | `1` |
| 4 | `-bias` | N/A | `true` |
| 5 | `-batch_first` | `-batchFirst` | `false` |
| 6 | `-dropout` | N/A | `0.0` |
| 7 | `-bidirectional` | N/A | `false` |

## Best Practices

### Architecture Design
```tcl
# Start simple and add complexity
set simple_gru [torch::gru -inputSize 50 -hiddenSize 100]

# Add layers for more capacity
set deeper_gru [torch::gru -inputSize 50 -hiddenSize 100 -numLayers 2]

# Add dropout for regularization
set regularized_gru [torch::gru -inputSize 50 -hiddenSize 100 -numLayers 2 -dropout 0.3]

# Use bidirectional for better context
set bidirectional_gru [torch::gru -inputSize 50 -hiddenSize 100 -numLayers 2 -dropout 0.3 -bidirectional 1]
```

### Hyperparameter Tuning
- **Hidden size**: Start with input_size, then try 2x or 4x
- **Number of layers**: Start with 1-2, rarely need more than 4
- **Dropout**: 0.2-0.5 for regularization
- **Bidirectional**: Use when you have access to full sequences

### Training Tips
- Use gradient clipping to prevent exploding gradients
- Consider learning rate scheduling
- Monitor validation loss to detect overfitting
- Use batch normalization between layers if needed

## See Also

- [torch::lstm](lstm.md) - Long Short-Term Memory networks
- [torch::rnn_tanh](rnn_tanh.md) - Simple RNN with tanh activation
- [torch::rnn_relu](rnn_relu.md) - Simple RNN with ReLU activation
- [torch::layer_forward](layer_forward.md) - Forward pass through layers
- [torch::optimizer_adam](optimizer_adam.md) - Adam optimizer for training

## Implementation Notes

- Based on PyTorch's `torch.nn.GRU`
- Supports all standard tensor data types (float32, float64, etc.)
- Automatic gradient support for training
- Efficient CUDA implementation when available
- Thread-safe for concurrent usage
- Compatible with PyTorch's pre-trained GRU models

## Advanced Usage

### Custom Initialization
```tcl
# Create GRU and initialize weights
set gru [torch::gru -inputSize 100 -hiddenSize 200 -numLayers 2]
# Use torch::layer_init_weights or similar for custom initialization
```

### Mixed Precision Training
```tcl
# GRU layers work with mixed precision training
set gru [torch::gru -inputSize 256 -hiddenSize 512 -numLayers 3 -dropout 0.1]
# Use appropriate loss scaling during training
```

### Gradient Clipping
```tcl
# After creating GRU, apply gradient clipping during training
set gru [torch::gru -inputSize 128 -hiddenSize 256 -numLayers 2]
# torch::nn_utils_clip_grad_norm_ $gru 1.0  # Clip gradients
``` 