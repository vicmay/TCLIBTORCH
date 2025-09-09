# torch::rnn_relu

Creates an RNN (Recurrent Neural Network) layer with ReLU activation function. RNN processes sequential data by maintaining hidden states that are updated at each time step.

## Syntax

### Current Syntax (Positional Parameters)
```tcl
torch::rnn_relu input_size hidden_size ?num_layers? ?bias? ?batch_first? ?dropout? ?bidirectional?
```

### New Syntax (Named Parameters)
```tcl
torch::rnn_relu -inputSize INT -hiddenSize INT [-numLayers INT] [-bias BOOL] [-batchFirst BOOL] [-dropout DOUBLE] [-bidirectional BOOL]
```

### CamelCase Alias
```tcl
torch::rnnRelu input_size hidden_size ?num_layers? ?bias? ?batch_first? ?dropout? ?bidirectional?
torch::rnnRelu -inputSize INT -hiddenSize INT [-numLayers INT] [-bias BOOL] [-batchFirst BOOL] [-dropout DOUBLE] [-bidirectional BOOL]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input_size` (positional) | int | Yes | - | Size of the input features |
| `-inputSize` | int | Yes | - | Size of the input features |
| `hidden_size` (positional) | int | Yes | - | Size of the hidden state |
| `-hiddenSize` | int | Yes | - | Size of the hidden state |
| `num_layers` (positional) | int | No | 1 | Number of recurrent layers |
| `-numLayers` | int | No | 1 | Number of recurrent layers |
| `bias` (positional) | bool | No | true | Whether to use bias parameters |
| `-bias` | bool | No | true | Whether to use bias parameters |
| `batch_first` (positional) | bool | No | false | Whether input/output tensors have batch dimension first |
| `-batchFirst` | bool | No | false | Whether input/output tensors have batch dimension first |
| `dropout` (positional) | double | No | 0.0 | Dropout probability (0.0 to 1.0) |
| `-dropout` | double | No | 0.0 | Dropout probability (0.0 to 1.0) |
| `bidirectional` (positional) | bool | No | false | Whether to use bidirectional RNN |
| `-bidirectional` | bool | No | false | Whether to use bidirectional RNN |

## Returns

Returns a handle to the created RNN module that can be used for forward passes and training.

## Description

The `torch::rnn_relu` function creates a multi-layer RNN with ReLU activation. Key features:

- **ReLU Activation**: Uses ReLU instead of traditional tanh or sigmoid
- **Multi-layer Support**: Can stack multiple RNN layers
- **Bidirectional Option**: Can process sequences in both directions
- **Dropout Regularization**: Applies dropout between layers
- **Flexible Input Format**: Supports both batch-first and sequence-first formats

### Mathematical Formula

For each time step t, the RNN with ReLU activation computes:
```
h_t = ReLU(W_ih * x_t + b_ih + W_hh * h_(t-1) + b_hh)
```

Where:
- `x_t` is the input at time step t
- `h_t` is the hidden state at time step t
- `W_ih`, `W_hh` are weight matrices
- `b_ih`, `b_hh` are bias vectors (if bias=true)

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a simple RNN with ReLU activation
set rnn [torch::rnn_relu 100 50]
puts "Created RNN: $rnn"

# RNN with custom parameters
set rnn [torch::rnn_relu 128 256 3 true false 0.1 false]
puts "Multi-layer RNN: $rnn"
```

### Named Parameters Syntax
```tcl
# Basic RNN with named parameters
set rnn [torch::rnn_relu -inputSize 100 -hiddenSize 50]
puts "Created RNN: $rnn"

# Advanced RNN configuration
set rnn [torch::rnn_relu \
    -inputSize 128 \
    -hiddenSize 256 \
    -numLayers 3 \
    -bias true \
    -batchFirst false \
    -dropout 0.1 \
    -bidirectional false]
puts "Advanced RNN: $rnn"
```

### CamelCase Alias
```tcl
# Using camelCase alias with positional syntax
set rnn [torch::rnnRelu 64 128 2]

# Using camelCase alias with named parameters
set rnn [torch::rnnRelu -inputSize 64 -hiddenSize 128 -numLayers 2]
```

### Bidirectional RNN
```tcl
# Create bidirectional RNN for better sequence modeling
set bi_rnn [torch::rnn_relu \
    -inputSize 100 \
    -hiddenSize 50 \
    -numLayers 2 \
    -bidirectional true \
    -dropout 0.2]
puts "Bidirectional RNN: $bi_rnn"
```

### Batch-First Format
```tcl
# RNN optimized for batch-first input format
set rnn [torch::rnn_relu \
    -inputSize 256 \
    -hiddenSize 512 \
    -batchFirst true \
    -numLayers 1]
puts "Batch-first RNN: $rnn"
```

### Complete Neural Network Example
```tcl
proc create_sequence_classifier {vocab_size embed_dim hidden_dim num_classes} {
    # Create embedding layer (not shown - would need separate implementation)
    # set embedding [torch::embedding $vocab_size $embed_dim]
    
    # Create RNN with ReLU activation
    set rnn [torch::rnn_relu \
        -inputSize $embed_dim \
        -hiddenSize $hidden_dim \
        -numLayers 2 \
        -bidirectional true \
        -dropout 0.3 \
        -batchFirst true]
    
    # Create classification layer (not shown - would need separate implementation)
    # set classifier [torch::linear [expr $hidden_dim * 2] $num_classes]  # *2 for bidirectional
    
    return [list $rnn]
}

# Usage
set vocab_size 10000
set embed_dim 128
set hidden_dim 256
set num_classes 5

set model_components [create_sequence_classifier $vocab_size $embed_dim $hidden_dim $num_classes]
set rnn [lindex $model_components 0]
puts "Sequence classifier RNN: $rnn"
```

## Input/Output Shapes

### Standard Format (batch_first=false)
- **Input**: `(seq_len, batch, input_size)`
- **Output**: `(seq_len, batch, hidden_size * num_directions)`
- **Hidden**: `(num_layers * num_directions, batch, hidden_size)`

### Batch-First Format (batch_first=true)
- **Input**: `(batch, seq_len, input_size)`
- **Output**: `(batch, seq_len, hidden_size * num_directions)`
- **Hidden**: `(num_layers * num_directions, batch, hidden_size)`

Where `num_directions = 1` for unidirectional, `2` for bidirectional.

## Performance Considerations

### Dropout
- Only applied between layers (requires `num_layers > 1`)
- Warning will be issued if `dropout > 0` and `num_layers = 1`
- Helps prevent overfitting in deep RNNs

### Memory Usage
- Memory scales with: `sequence_length × batch_size × hidden_size`
- Bidirectional RNNs use approximately 2x memory
- Consider gradient checkpointing for very long sequences

### ReLU vs Traditional Activations
- **ReLU Advantages**: Faster computation, no vanishing gradient problem
- **ReLU Disadvantages**: Can suffer from "dying ReLU" problem
- **Use Cases**: Better for shorter sequences, when gradient flow is important

## Error Handling

The function will raise an error if:
- `input_size` or `hidden_size` is not positive
- `num_layers` is not positive
- `dropout` is negative
- Invalid parameter types are provided
- Unknown parameters are specified (named syntax only)

### Error Examples
```tcl
# Error: Missing required parameters
catch {torch::rnn_relu 10} error
puts "Error: $error"
# Output: Usage: torch::rnn_relu input_size hidden_size...

# Error: Invalid input size
catch {torch::rnn_relu -inputSize -5 -hiddenSize 10} error
puts "Error: $error"
# Output: Required parameters missing or invalid: inputSize, hiddenSize must be positive

# Error: Unknown parameter
catch {torch::rnn_relu -inputSize 10 -hiddenSize 20 -invalidParam 5} error
puts "Error: $error"
# Output: Unknown parameter: -invalidParam

# Error: Negative dropout
catch {torch::rnn_relu -inputSize 10 -hiddenSize 20 -dropout -0.1} error
puts "Error: $error"
# Output: Required parameters missing or invalid...
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set rnn [torch::rnn_relu 128 256 3 true false 0.1 true]

# New named parameter syntax
set rnn [torch::rnn_relu \
    -inputSize 128 \
    -hiddenSize 256 \
    -numLayers 3 \
    -bias true \
    -batchFirst false \
    -dropout 0.1 \
    -bidirectional true]

# Both syntaxes work - choose based on preference
```

### Adopting CamelCase

```tcl
# Traditional snake_case
set rnn [torch::rnn_relu 64 128 2]

# Modern camelCase alias
set rnn [torch::rnnRelu 64 128 2]

# Both work identically
```

## Comparison with Other RNN Types

| Feature | torch::rnn_relu | torch::rnn_tanh | torch::lstm | torch::gru |
|---------|-----------------|-----------------|-------------|------------|
| Activation | ReLU | tanh | Gated | Gated |
| Memory Complexity | Low | Low | High | Medium |
| Gradient Flow | Excellent | Good | Excellent | Good |
| Long Sequences | Poor | Good | Excellent | Good |
| Training Speed | Fast | Medium | Slow | Medium |

## Compatibility

- ✅ **Backward Compatible**: All existing code using positional syntax continues to work
- ✅ **New Features**: Named parameters provide better readability and flexibility
- ✅ **Modern Style**: CamelCase aliases follow modern TCL conventions
- ✅ **Performance**: No performance penalty for using either syntax

## See Also

- [`torch::rnn_tanh`](rnn_tanh.md) - RNN with tanh activation
- [`torch::lstm`](lstm.md) - Long Short-Term Memory networks
- [`torch::gru`](gru.md) - Gated Recurrent Unit networks
- [`torch::linear`](linear.md) - Linear transformation layers
- [Recurrent Networks Guide](../guides/recurrent.md) - Complete guide to RNN usage

## Version History

- **v2.0**: Added dual syntax support (named parameters + camelCase aliases)
- **v1.0**: Initial implementation with positional parameters 