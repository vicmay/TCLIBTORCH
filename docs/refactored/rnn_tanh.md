# torch::rnn_tanh

Creates a Recurrent Neural Network (RNN) module with tanh activation function.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::rnn_tanh -inputSize <int> -hiddenSize <int> [-numLayers <int>] [-bias <bool>] [-batchFirst <bool>] [-dropout <float>] [-bidirectional <bool>]
```

### Positional Syntax (Legacy)
```tcl
torch::rnn_tanh <input_size> <hidden_size> [<num_layers>] [<bias>] [<batch_first>] [<dropout>] [<bidirectional>]
```

### CamelCase Alias
```tcl
torch::rnnTanh -inputSize <int> -hiddenSize <int> [options...]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `inputSize`/`input_size` | integer | Yes | - | Number of input features |
| `hiddenSize`/`hidden_size` | integer | Yes | - | Number of hidden units |
| `numLayers`/`num_layers` | integer | No | 1 | Number of stacked RNN layers |
| `bias` | boolean | No | true | Whether to use bias parameters |
| `batchFirst`/`batch_first` | boolean | No | false | If true, input shape is (batch, seq, feature) |
| `dropout` | float | No | 0.0 | Dropout probability (0.0-1.0) |
| `bidirectional` | boolean | No | false | Whether to use bidirectional RNN |

## Returns

Returns a string handle to the created RNN module that can be used with other PyTorch operations.

## Description

The RNN (Recurrent Neural Network) with tanh activation is a fundamental sequence modeling architecture. Each RNN layer applies the tanh activation function to the hidden state at each time step, providing non-linear transformations that help capture complex temporal dependencies.

### Mathematical Formula

For a single layer RNN with tanh activation:
```
h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
```

Where:
- `x_t` is the input at time step t
- `h_t` is the hidden state at time step t
- `W_ih`, `W_hh` are weight matrices
- `b_ih`, `b_hh` are bias vectors (if enabled)

### Multi-layer RNNs

When `numLayers > 1`, layers are stacked vertically:
- Layer 1 takes sequence input, outputs hidden sequence
- Layer 2 takes Layer 1's output as input, and so on
- Final layer produces the output sequence

### Bidirectional RNNs

When `bidirectional=true`:
- Forward RNN processes sequence left-to-right
- Backward RNN processes sequence right-to-left
- Hidden size is doubled (concat forward and backward)
- Provides richer representation by seeing future context

## Examples

### Basic Usage - Named Parameters
```tcl
# Create simple RNN with tanh activation
set rnn [torch::rnn_tanh -inputSize 10 -hiddenSize 20]

# Create with multiple layers
set rnn [torch::rnn_tanh -inputSize 64 -hiddenSize 128 -numLayers 3]
```

### Advanced Configuration
```tcl
# Bidirectional RNN with dropout
set rnn [torch::rnn_tanh \
    -inputSize 100 \
    -hiddenSize 256 \
    -numLayers 2 \
    -bidirectional true \
    -dropout 0.3 \
    -batchFirst true]

# Without bias parameters
set rnn [torch::rnn_tanh -inputSize 50 -hiddenSize 100 -bias false]
```

### Legacy Positional Syntax
```tcl
# Basic RNN
set rnn [torch::rnn_tanh 10 20]

# With all parameters
set rnn [torch::rnn_tanh 64 128 3 true false 0.2 true]
```

### CamelCase Alias
```tcl
# Same functionality with camelCase
set rnn [torch::rnnTanh -inputSize 10 -hiddenSize 20 -numLayers 2]
```

## Input/Output Shapes

### Input Tensor Shapes
- **batchFirst=false**: `(seq_len, batch, input_size)`
- **batchFirst=true**: `(batch, seq_len, input_size)`

### Output Tensor Shapes
- **Unidirectional**: `(seq_len, batch, hidden_size)` or `(batch, seq_len, hidden_size)`
- **Bidirectional**: `(seq_len, batch, 2 * hidden_size)` or `(batch, seq_len, 2 * hidden_size)`

### Hidden State Shapes
- **Unidirectional**: `(num_layers, batch, hidden_size)`
- **Bidirectional**: `(2 * num_layers, batch, hidden_size)`

## Use Cases

1. **Sequence Classification**: Text sentiment analysis, document classification
2. **Language Modeling**: Word prediction, text generation
3. **Time Series Prediction**: Stock prices, weather forecasting
4. **Speech Recognition**: Audio to text conversion
5. **Machine Translation**: Sequence-to-sequence translation

## Performance Notes

- **Tanh vs ReLU**: Tanh activation provides smoother gradients but may suffer from vanishing gradients in deep networks
- **Dropout**: Applied between layers (not within recurrent connections)
- **Bidirectional**: Doubles computation and memory usage
- **Batch First**: Can be more efficient for certain operations

## Error Handling

The function validates all parameters and provides descriptive error messages:

```tcl
# Missing required parameters
catch {torch::rnn_tanh -inputSize 10} error
# Error: Required parameters missing or invalid

# Invalid parameter values
catch {torch::rnn_tanh -inputSize 0 -hiddenSize 20} error
# Error: Required parameters missing or invalid

# Unknown parameters
catch {torch::rnn_tanh -inputSize 10 -hiddenSize 20 -invalid 5} error
# Error: Unknown parameter: -invalid
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set rnn [torch::rnn_tanh 64 128 2 true false 0.1 false]

# New named parameter syntax
set rnn [torch::rnn_tanh \
    -inputSize 64 \
    -hiddenSize 128 \
    -numLayers 2 \
    -bias true \
    -batchFirst false \
    -dropout 0.1 \
    -bidirectional false]
```

### Benefits of Named Parameters
- **Clarity**: Parameter names make code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Easy to modify specific parameters
- **Error Prevention**: Less likely to mix up parameter positions

## Comparison with Other RNN Types

| Type | Activation | Gating | Complexity | Use Case |
|------|------------|--------|------------|----------|
| RNN-Tanh | tanh | None | Simple | Basic sequences |
| RNN-ReLU | ReLU | None | Simple | Avoiding vanishing gradients |
| LSTM | Sigmoid/Tanh | Input/Forget/Output | Complex | Long sequences |
| GRU | Sigmoid/Tanh | Reset/Update | Medium | Balanced performance |

## See Also

- [`torch::rnn_relu`](rnn_relu.md) - RNN with ReLU activation
- [`torch::lstm`](lstm.md) - Long Short-Term Memory networks
- [`torch::gru`](gru.md) - Gated Recurrent Unit networks
- [`torch::linear`](linear.md) - Fully connected layers

## Version Information

- **Introduced**: LibTorch TCL Extension v1.0
- **Dual Syntax Support**: v2.0
- **CamelCase Alias**: v2.0 