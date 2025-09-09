# torch::scaled_dot_product_attention

Compute scaled dot-product attention as described in the "Attention Is All You Need" paper.

## Syntax

```tcl
# Positional syntax
torch::scaled_dot_product_attention query key value

# Named parameter syntax
torch::scaled_dot_product_attention -query query -key key -value value
```

The command also supports a camelCase alias: `torch::scaledDotProductAttention`

## Arguments

* `query` (positional) or `-query query` (named): Query tensor of shape `[batch_size, num_heads, seq_len_q, d_k]`
* `key` (positional) or `-key key` (named): Key tensor of shape `[batch_size, num_heads, seq_len_k, d_k]`
* `value` (positional) or `-value value` (named): Value tensor of shape `[batch_size, num_heads, seq_len_v, d_v]`

## Return Value

Returns a tensor of shape `[batch_size, num_heads, seq_len_q, d_v]` containing the attention output.

## Examples

```tcl
# Create test tensors
set query [torch::tensor_create {{{1.0 2.0} {3.0 4.0}} {{5.0 6.0} {7.0 8.0}}}]
set key [torch::tensor_create {{{1.0 0.0} {0.0 1.0}} {{1.0 0.0} {0.0 1.0}}}]
set value [torch::tensor_create {{{1.0 0.0} {0.0 1.0}} {{1.0 0.0} {0.0 1.0}}}]

# Using positional syntax
set result [torch::scaled_dot_product_attention $query $key $value]

# Using named parameter syntax
set result [torch::scaled_dot_product_attention -query $query -key $key -value $value]

# Using camelCase alias
set result [torch::scaledDotProductAttention -query $query -key $key -value $value]
```

## Error Conditions

* If required parameters (query, key, value) are missing
* If any of the tensors is invalid
* If an unknown parameter is provided in named syntax

## See Also

* `torch::multihead_attention` - Multi-head attention layer
* `torch::transformer_encoder` - Transformer encoder layer
* `torch::transformer_decoder` - Transformer decoder layer 