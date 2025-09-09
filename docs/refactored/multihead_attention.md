# torch::multihead_attention

Multi-head attention mechanism implementation for Transformer architectures.

## Syntax

### Current (Positional Parameters)
```tcl
torch::multihead_attention query key value embed_dim num_heads
```

### New (Named Parameters) 
```tcl
torch::multihead_attention -query TENSOR -key TENSOR -value TENSOR -embedDim INT -numHeads INT
```

### CamelCase Alias
```tcl
torch::multiheadAttention ...  # Same syntax as above
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-query` | Tensor | Yes | - | Input query tensor |
| `-key` | Tensor | Yes | - | Input key tensor |
| `-value` | Tensor | Yes | - | Input value tensor |
| `-embedDim` | Integer | Yes | - | Embedding dimension (must be divisible by num_heads) |
| `-numHeads` | Integer | Yes | - | Number of attention heads |

## Description

Implements multi-head attention as described in "Attention Is All You Need" (Vaswani et al., 2017). The operation performs scaled dot-product attention across multiple heads in parallel, then concatenates and projects the results.

### Mathematical Operation

1. Split query, key, and value tensors into multiple heads
2. For each head: Attention(Q, K, V) = softmax(QK^T / √d_k)V
3. Concatenate all head outputs
4. Return final output tensor

### Input Shape Requirements
- **query**: `[seq_len, batch_size, embed_dim]`
- **key**: `[seq_len, batch_size, embed_dim]`  
- **value**: `[seq_len, batch_size, embed_dim]`

### Output Shape
- Returns: `[seq_len, batch_size, embed_dim]`

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create test tensors
set query [torch::tensor_randn -shape {10 2 512}]
set key [torch::tensor_randn -shape {10 2 512}]
set value [torch::tensor_randn -shape {10 2 512}]

# Multi-head attention with 8 heads
set output [torch::multihead_attention $query $key $value 512 8]
```

### Named Parameter Syntax
```tcl
# Same operation with named parameters
set output [torch::multihead_attention \
    -query $query \
    -key $key \
    -value $value \
    -embedDim 512 \
    -numHeads 8]
```

### CamelCase Alias
```tcl
# Using camelCase alias
set output [torch::multiheadAttention \
    -query $query \
    -key $key \
    -value $value \
    -embedDim 512 \
    -numHeads 8]
```

### Self-Attention Example
```tcl
# Self-attention (query, key, value are the same)
set x [torch::tensor_randn -shape {20 4 256}]

set self_attn [torch::multihead_attention \
    -query $x \
    -key $x \
    -value $x \
    -embedDim 256 \
    -numHeads 4]
```

### Cross-Attention Example
```tcl
# Cross-attention (decoder attending to encoder)
set decoder_input [torch::tensor_randn -shape {15 4 256}]
set encoder_output [torch::tensor_randn -shape {20 4 256}]

set cross_attn [torch::multihead_attention \
    -query $decoder_input \
    -key $encoder_output \
    -value $encoder_output \
    -embedDim 256 \
    -numHeads 8]
```

### Different Number of Heads
```tcl
# Single head attention
set single_head [torch::multihead_attention \
    -query $query \
    -key $key \
    -value $value \
    -embedDim 512 \
    -numHeads 1]

# Many heads (embed_dim must be divisible by num_heads)
set many_heads [torch::multihead_attention \
    -query $query \
    -key $key \
    -value $value \
    -embedDim 512 \
    -numHeads 16]
```

## Error Handling

### Common Errors

1. **Missing Required Parameters**
```tcl
# Error: missing value parameter
catch {torch::multihead_attention -query $q -key $k -embedDim 256 -numHeads 4} error
# Result: "Required parameters missing: query, key, value, embedDim, numHeads"
```

2. **Unknown Parameters**
```tcl
# Error: typo in parameter name
catch {torch::multihead_attention -query $q -key $k -value $v -embedDim 256 -heads 4} error
# Result: "Unknown parameter: -heads"
```

3. **Invalid Head Count**
```tcl
# Error: embed_dim not divisible by num_heads
catch {torch::multihead_attention -query $q -key $k -value $v -embedDim 257 -numHeads 4} error
# Note: This would cause mathematical issues in the head_dim calculation
```

## Performance Considerations

### Memory Usage
- Memory usage scales with `seq_len²` due to attention matrix
- Multiple heads process in parallel
- Large sequences may require gradient checkpointing

### Computational Complexity
- Time complexity: O(seq_len² × embed_dim)
- Space complexity: O(seq_len² × num_heads)

### Optimization Tips
```tcl
# Use appropriate head counts (powers of 2 often work well)
set heads 8  ;# Common choices: 1, 2, 4, 8, 16, 32

# Ensure embed_dim is divisible by num_heads
set embed_dim 512
set head_dim [expr {$embed_dim / $heads}]  ;# Should be integer
```

## Migration Guide

### From Positional to Named Parameters

**Old Syntax:**
```tcl
set result [torch::multihead_attention $query $key $value 512 8]
```

**New Syntax:**
```tcl
set result [torch::multihead_attention \
    -query $query \
    -key $key \
    -value $value \
    -embedDim 512 \
    -numHeads 8]
```

### Benefits of New Syntax
- **Self-documenting**: Parameter names make code more readable
- **Flexible ordering**: Parameters can be specified in any order
- **Error prevention**: Less likely to pass parameters in wrong order
- **Future-proof**: Easy to add optional parameters

### Backward Compatibility
- Old positional syntax continues to work unchanged
- No breaking changes to existing code
- Both syntaxes produce identical results

## Implementation Notes

### Attention Mechanism Details
1. **Query/Key/Value Projection**: Input tensors are reshaped for multi-head processing
2. **Scaled Dot-Product**: Uses scaling factor of 1/√(head_dim) for numerical stability
3. **Softmax Attention**: Applied along the key dimension
4. **Output Projection**: Concatenated heads are returned in original embedding dimension

### Mathematical Formulation
```
head_dim = embed_dim / num_heads
scale = 1 / sqrt(head_dim)

For each head h:
  Q_h = query.view(seq_len, batch, num_heads, head_dim)[h]
  K_h = key.view(seq_len, batch, num_heads, head_dim)[h]  
  V_h = value.view(seq_len, batch, num_heads, head_dim)[h]
  
  Attention_h = softmax(Q_h @ K_h^T * scale) @ V_h

Output = concat(Attention_1, ..., Attention_h).view(seq_len, batch, embed_dim)
```

## See Also
- [`torch::scaled_dot_product_attention`](scaled_dot_product_attention.md) - Single-head attention
- [`torch::transformer_encoder_layer`](transformer_encoder_layer.md) - Complete transformer encoder layer
- [`torch::transformer_decoder_layer`](transformer_decoder_layer.md) - Complete transformer decoder layer
- [`torch::positional_encoding`](positional_encoding.md) - Positional encoding for transformers

## References
- Vaswani, A., et al. (2017). "Attention Is All You Need." *arXiv preprint arXiv:1706.03762*
- PyTorch MultiheadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html 