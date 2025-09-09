# torch::transformer_encoder_layer / torch::transformerEncoderLayer

Transformer Encoder Layer (Simplified)

---

## Overview

Runs a single transformer encoder layer on the input tensor. This is a simplified implementation for demonstration and testing purposes.

---

## Syntax

### Positional (Backward Compatible)
```tcl
torch::transformer_encoder_layer src d_model nhead dim_feedforward dropout
```

### Named Parameters (Modern)
```tcl
torch::transformer_encoder_layer -src SRC -dModel D_MODEL -nhead NHEAD -dimFeedforward DIM_FEEDFORWARD -dropout DROPOUT
torch::transformerEncoderLayer -src SRC -dModel D_MODEL -nhead NHEAD -dimFeedforward DIM_FEEDFORWARD -dropout DROPOUT
```

---

## Parameters
| Name            | Type         | Required | Description                                 |
|-----------------|--------------|----------|---------------------------------------------|
| src             | Tensor       | Yes      | Input tensor (shape: [*, d_model])          |
| d_model / dModel| Integer      | Yes      | Embedding dimension                         |
| nhead           | Integer      | Yes      | Number of attention heads (unused, for API) |
| dim_feedforward / dimFeedforward | Integer | Yes | Feedforward layer dimension                |
| dropout         | Double       | Yes      | Dropout probability (0.0 - 1.0)             |

---

## Return Value
A tensor of the same shape as `src` (last dimension = d_model).

---

## Examples

### Positional Syntax
```tcl
set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set result [torch::transformer_encoder_layer $src 4 2 8 0.1]
```

### Named Parameter Syntax
```tcl
set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set result [torch::transformer_encoder_layer -src $src -dModel 4 -nhead 2 -dimFeedforward 8 -dropout 0.1]
```

### CamelCase Alias
```tcl
set src [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set result [torch::transformerEncoderLayer -src $src -dModel 4 -nhead 2 -dimFeedforward 8 -dropout 0.1]
```

---

## Migration Guide
- **Old code using positional syntax will continue to work.**
- **New code should use named parameters and camelCase for clarity and maintainability.**

| Old (positional) | New (named/camelCase) |
|------------------|----------------------|
| torch::transformer_encoder_layer $src 4 2 8 0.1 | torch::transformerEncoderLayer -src $src -dModel 4 -nhead 2 -dimFeedforward 8 -dropout 0.1 |

---

## Error Handling
- Missing or invalid parameters will produce a clear error message.
- Both syntaxes are validated for required arguments.

---

## See Also
- [torch::transformer_encoder](transformer_encoder.md)
- [torch::transformer_decoder_layer](transformer_decoder_layer.md) 