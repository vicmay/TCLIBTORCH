# torch::transformer_encoder / torch::transformerEncoder

Transformer Encoder (Simplified)

---

## Overview

Runs a stack of transformer encoder layers on the input tensor. This is a simplified implementation for demonstration and testing purposes.

---

## Syntax

### Positional (Backward Compatible)
```tcl
torch::transformer_encoder src d_model nhead num_layers dim_feedforward
```

### Named Parameters (Modern)
```tcl
torch::transformer_encoder -src SRC -dModel D_MODEL -nhead NHEAD -numLayers NUM_LAYERS -dimFeedforward DIM_FEEDFORWARD
torch::transformerEncoder -src SRC -dModel D_MODEL -nhead NHEAD -numLayers NUM_LAYERS -dimFeedforward DIM_FEEDFORWARD
```

---

## Parameters
| Name            | Type         | Required | Description                                 |
|-----------------|--------------|----------|---------------------------------------------|
| src             | Tensor       | Yes      | Input tensor (shape: [*, d_model])          |
| d_model / dModel| Integer      | Yes      | Embedding dimension                         |
| nhead           | Integer      | Yes      | Number of attention heads (unused, for API) |
| num_layers / numLayers | Integer| Yes      | Number of encoder layers                    |
| dim_feedforward / dimFeedforward | Integer | Yes | Feedforward layer dimension                |

---

## Return Value
A tensor of the same shape as `src` (last dimension = d_model).

---

## Examples

### Positional Syntax
```tcl
set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set result [torch::transformer_encoder $src 4 2 2 8]
```

### Named Parameter Syntax
```tcl
set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set result [torch::transformer_encoder -src $src -dModel 4 -nhead 2 -numLayers 2 -dimFeedforward 8]
```

### CamelCase Alias
```tcl
set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set result [torch::transformerEncoder -src $src -dModel 4 -nhead 2 -numLayers 2 -dimFeedforward 8]
```

---

## Migration Guide
- **Old code using positional syntax will continue to work.**
- **New code should use named parameters and camelCase for clarity and maintainability.**

| Old (positional) | New (named/camelCase) |
|------------------|----------------------|
| torch::transformer_encoder $src 4 2 2 8 | torch::transformerEncoder -src $src -dModel 4 -nhead 2 -numLayers 2 -dimFeedforward 8 |

---

## Error Handling
- Missing or invalid parameters will produce a clear error message.
- Both syntaxes are validated for required arguments.

---

## See Also
- [torch::transformer_decoder](transformer_decoder.md)
- [torch::transformer_encoder_layer](transformer_encoder_layer.md) 