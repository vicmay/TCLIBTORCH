# torch::transformer_decoder

Creates a transformer decoder with multiple layers, implementing self-attention, cross-attention, and feed-forward networks.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::transformer_decoder tgt memory d_model nhead num_layers dim_feedforward
```

### Named Parameters (New)
```tcl
torch::transformer_decoder -tgt tgt -memory memory -dModel d_model -nhead nhead -numLayers num_layers -dimFeedforward dim_feedforward
```

### CamelCase Alias
```tcl
torch::transformerDecoder tgt memory d_model nhead num_layers dim_feedforward
torch::transformerDecoder -tgt tgt -memory memory -dModel d_model -nhead nhead -numLayers num_layers -dimFeedforward dim_feedforward
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `tgt` / `-tgt` | tensor | Target tensor (decoder input) | Yes |
| `memory` / `-memory` | tensor | Memory tensor (encoder output) | Yes |
| `d_model` / `-dModel` | int | Dimension of the model | Yes |
| `nhead` / `-nhead` | int | Number of attention heads | Yes |
| `num_layers` / `-numLayers` | int | Number of transformer layers | Yes |
| `dim_feedforward` / `-dimFeedforward` | int | Dimension of feed-forward network | Yes |

## Return Value

Returns a tensor handle representing the output of the transformer decoder.

## Description

The `torch::transformer_decoder` command implements a multi-layer transformer decoder architecture. Each layer consists of:

1. **Self-Attention**: Allows the decoder to attend to previous positions in the sequence
2. **Cross-Attention**: Allows the decoder to attend to the encoder output (memory)
3. **Feed-Forward Network**: A two-layer linear transformation with ReLU activation
4. **Layer Normalization**: Applied after each sub-layer
5. **Residual Connections**: Added around each sub-layer

The implementation automatically handles tensor shape adjustments to match the specified `d_model` dimension.

## Examples

### Basic Usage with Positional Parameters
```tcl
# Create input tensors
set tgt [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set memory [torch::tensor_create -data {5.0 6.0 7.0 8.0} -shape {2 2} -dtype float32]

# Create transformer decoder
set result [torch::transformer_decoder $tgt $memory 2 1 2 4]

# Get output shape
set shape [torch::tensor_shape $result]
puts "Output shape: $shape"

# Cleanup
torch::tensor_delete $tgt
torch::tensor_delete $memory
torch::tensor_delete $result
```

### Using Named Parameters
```tcl
# Create input tensors
set tgt [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set memory [torch::tensor_create -data {5.0 6.0 7.0 8.0} -shape {2 2} -dtype float32]

# Create transformer decoder with named parameters
set result [torch::transformer_decoder -tgt $tgt -memory $memory -dModel 4 -nhead 2 -numLayers 1 -dimFeedforward 8]

# Get output shape
set shape [torch::tensor_shape $result]
puts "Output shape: $shape"

# Cleanup
torch::tensor_delete $tgt
torch::tensor_delete $memory
torch::tensor_delete $result
```

### Using CamelCase Alias
```tcl
# Create input tensors
set tgt [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set memory [torch::tensor_create -data {5.0 6.0 7.0 8.0} -shape {2 2} -dtype float32]

# Create transformer decoder using camelCase alias
set result [torch::transformerDecoder $tgt $memory 2 1 2 4]

# Get output shape
set shape [torch::tensor_shape $result]
puts "Output shape: $shape"

# Cleanup
torch::tensor_delete $tgt
torch::tensor_delete $memory
torch::tensor_delete $result
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
torch::transformer_decoder $tgt $memory 512 8 6 2048
```

**New (Named):**
```tcl
torch::transformer_decoder -tgt $tgt -memory $memory -dModel 512 -nhead 8 -numLayers 6 -dimFeedforward 2048
```

### Using CamelCase Alias

**Old (snake_case):**
```tcl
torch::transformer_decoder $tgt $memory 512 8 6 2048
```

**New (camelCase):**
```tcl
torch::transformerDecoder $tgt $memory 512 8 6 2048
```

## Error Handling

The command provides clear error messages for various error conditions:

- **Missing parameters**: "Usage: torch::transformer_decoder tgt memory d_model nhead num_layers dim_feedforward"
- **Invalid parameter values**: "Invalid parameters: all tensors must be defined, dModel, nhead, numLayers, and dimFeedforward must be positive"
- **Unknown named parameters**: "Unknown parameter: -unknown"
- **Missing parameter values**: "Missing value for parameter"

## Notes

- The implementation is simplified and uses identity transformations for attention mechanisms
- Input tensors are automatically padded or truncated to match the `d_model` dimension
- All parameters must be positive integers
- The output tensor maintains the same batch and sequence dimensions as the input, with the last dimension adjusted to `d_model`
- This is a simplified implementation for demonstration purposes; production transformer decoders would include more sophisticated attention mechanisms

## Related Commands

- `torch::transformer_encoder` - Creates a transformer encoder
- `torch::transformer_encoder_layer` - Creates a single transformer encoder layer
- `torch::transformer_decoder_layer` - Creates a single transformer decoder layer
- `torch::multi_head_attention` - Implements multi-head attention mechanism 