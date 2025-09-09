# torch::linearLayer / torch::linear

Creates a linear (fully connected) layer that applies a linear transformation to incoming data: `y = xA^T + b`.

## Syntax

### Legacy Syntax (Positional Parameters)
```tcl
torch::linearLayer in_features out_features ?bias?
```

### Modern Syntax (Named Parameters)
```tcl
torch::linearLayer -inFeatures <int> -outFeatures <int> ?-bias <bool>?
torch::linear -inFeatures <int> -outFeatures <int> ?-bias <bool>?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `in_features` / `-inFeatures` | integer | Yes | - | Size of each input sample |
| `out_features` / `-outFeatures` | integer | Yes | - | Size of each output sample |
| `bias` / `-bias` | boolean | No | `true` | Whether to include learnable bias parameters |

## Returns

Returns a string handle representing the linear layer module that can be used with `torch::layer_forward` for forward passes.

## Examples

### Creating Basic Linear Layers

```tcl
# Legacy syntax
set layer1 [torch::linearLayer 784 128]
set layer2 [torch::linearLayer 128 64 true]
set layer3 [torch::linearLayer 64 10 false]

# Modern syntax with named parameters
set layer1 [torch::linearLayer -inFeatures 784 -outFeatures 128]
set layer2 [torch::linearLayer -inFeatures 128 -outFeatures 64 -bias true]
set layer3 [torch::linearLayer -inFeatures 64 -outFeatures 10 -bias false]

# camelCase alias (recommended)
set layer1 [torch::linear -inFeatures 784 -outFeatures 128]
set layer2 [torch::linear -inFeatures 128 -outFeatures 64 -bias true]
set layer3 [torch::linear -inFeatures 64 -outFeatures 10 -bias false]
```

### Forward Pass Examples

```tcl
# Create layer and perform forward pass
set linear [torch::linear -inFeatures 100 -outFeatures 50]
set input [torch::randn -shape {32 100}]
set output [torch::layer_forward $linear $input]

# Check output shape
puts [torch::tensor_shape $output]  ;# Output: 32 50

# Different batch sizes
set single_input [torch::randn -shape {1 100}]
set single_output [torch::layer_forward $linear $single_input]
puts [torch::tensor_shape $single_output]  ;# Output: 1 50
```

### Neural Network Layer Building

```tcl
# Build a simple feedforward network
proc create_mlp {input_size hidden_size output_size} {
    set layer1 [torch::linear -inFeatures $input_size -outFeatures $hidden_size]
    set layer2 [torch::linear -inFeatures $hidden_size -outFeatures $output_size -bias false]
    return [list $layer1 $layer2]
}

# Create 784 -> 256 -> 10 network
set layers [create_mlp 784 256 10]
set layer1 [lindex $layers 0]
set layer2 [lindex $layers 1]

# Forward pass through network
set input [torch::randn -shape {16 784}]
set hidden [torch::layer_forward $layer1 $input]
set output [torch::layer_forward $layer2 $hidden]
puts [torch::tensor_shape $output]  ;# Output: 16 10
```

### Common Layer Configurations

```tcl
# Classification head (with bias)
set classifier [torch::linear -inFeatures 512 -outFeatures 1000]

# Feature projection (without bias)
set projection [torch::linear -inFeatures 2048 -outFeatures 256 -bias false]

# Embedding layer
set embedding [torch::linear -inFeatures 10000 -outFeatures 300]

# Single neuron output
set regressor [torch::linear -inFeatures 64 -outFeatures 1]
```

### Parameter Variations

```tcl
# Different ways to specify bias parameter
set with_bias1 [torch::linearLayer 100 50 true]
set with_bias2 [torch::linearLayer 100 50 1]
set with_bias3 [torch::linear -inFeatures 100 -outFeatures 50 -bias true]

set without_bias1 [torch::linearLayer 100 50 false]
set without_bias2 [torch::linearLayer 100 50 0]
set without_bias3 [torch::linear -inFeatures 100 -outFeatures 50 -bias false]
```

## Mathematical Details

The linear layer performs the transformation:

```
y = xA^T + b
```

Where:
- `x` is the input tensor of shape `(batch_size, in_features)`
- `A` is the weight matrix of shape `(out_features, in_features)`
- `b` is the bias vector of shape `(out_features)` (if bias=true)
- `y` is the output tensor of shape `(batch_size, out_features)`

### Weight Initialization

The layer weights are initialized using PyTorch's default initialization:
- Weights: Uniform distribution `U(-sqrt(k), sqrt(k))` where `k = 1/in_features`
- Bias (if enabled): Uniform distribution `U(-sqrt(k), sqrt(k))` where `k = 1/in_features`

## Common Use Cases

### 1. Classification Networks
```tcl
# MNIST classifier
set fc1 [torch::linear -inFeatures 784 -outFeatures 128]
set fc2 [torch::linear -inFeatures 128 -outFeatures 64]
set fc3 [torch::linear -inFeatures 64 -outFeatures 10]
```

### 2. Feature Extractors
```tcl
# Dimensionality reduction
set feature_extractor [torch::linear -inFeatures 2048 -outFeatures 512]
```

### 3. Regression Models
```tcl
# Single output regression
set regressor [torch::linear -inFeatures 20 -outFeatures 1]
```

### 4. Attention Mechanisms
```tcl
# Query, Key, Value projections
set query_proj [torch::linear -inFeatures 512 -outFeatures 64 -bias false]
set key_proj [torch::linear -inFeatures 512 -outFeatures 64 -bias false]
set value_proj [torch::linear -inFeatures 512 -outFeatures 64 -bias false]
```

## Error Handling

```tcl
# Invalid parameters will raise errors
catch {torch::linear -inFeatures -10 -outFeatures 5} error_msg
puts $error_msg  ;# "Required parameters missing or invalid: inFeatures and outFeatures must be positive"

catch {torch::linear -inFeatures 10} error_msg
puts $error_msg  ;# "Required parameters missing or invalid: inFeatures and outFeatures must be positive"

catch {torch::linear -inFeatures 10 -outFeatures 5 -bias invalid} error_msg
puts $error_msg  ;# "Invalid value for -bias parameter (should be boolean)"
```

## Performance Considerations

1. **Layer Size**: Larger layers require more memory and computation
   - `784 x 128` layer has 100,352 parameters (784*128 + 128 bias)
   - Memory usage: ~400KB for float32 weights

2. **Bias Parameter**: Disabling bias reduces parameters by `out_features`
   - Use `-bias false` for layers followed by normalization

3. **Batch Processing**: Larger batch sizes improve GPU utilization
   - Prefer batch processing over single samples

## Related Commands

- `torch::layer_forward` - Perform forward pass through the layer
- `torch::conv2d` - 2D convolution layers
- `torch::dropout` - Dropout layers for regularization
- `torch::batchnorm2d` - Batch normalization layers

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# OLD (Legacy)
set layer [torch::linearLayer 100 50 true]

# NEW (Modern - equivalent)
set layer [torch::linear -inFeatures 100 -outFeatures 50 -bias true]
```

### Parameter Mapping

| Legacy Position | Modern Parameter | Description |
|----------------|------------------|-------------|
| 1st argument | `-inFeatures` | Input feature dimensions |
| 2nd argument | `-outFeatures` | Output feature dimensions |
| 3rd argument | `-bias` | Include bias (optional, default: true) |

### Benefits of Modern Syntax

1. **Self-documenting**: Parameter names make intent clear
2. **Flexible ordering**: Parameters can be specified in any order
3. **IDE support**: Better autocomplete and error checking
4. **Extensible**: Easy to add new parameters in future
5. **Consistent**: Matches modern PyTorch and other frameworks

## Version History

- **v1.0**: Initial implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters and camelCase alias
- **Current**: Both syntaxes supported, camelCase (`torch::linear`) recommended

## Notes

- Both `torch::linearLayer` and `torch::linear` refer to the same implementation
- The camelCase alias `torch::linear` is recommended for new code
- Legacy positional syntax remains fully supported for backward compatibility
- Layer handles can be reused for multiple forward passes
- Weights are trainable parameters that can be updated during training 