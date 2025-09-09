# torch::sparse_embedding / torch::sparseEmbedding

Creates a sparse embedding layer that maps indices to dense vectors. The sparse version uses sparse gradients during training, which can be more memory efficient when dealing with large embedding tables where only a small subset of embeddings are accessed at each step.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::sparse_embedding input num_embeddings embedding_dim padding_idx
```

### Named Parameter Syntax
```tcl
torch::sparse_embedding -input tensor -num_embeddings int -embedding_dim int -padding_idx int
```

### CamelCase Alias
```tcl
torch::sparseEmbedding -input tensor -num_embeddings int -embedding_dim int -padding_idx int
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| input | tensor | Input tensor containing indices into the embedding matrix. Must be of type long/int64. |
| num_embeddings | int | Size of the dictionary of embeddings (number of rows in the embedding matrix). |
| embedding_dim | int | Size of each embedding vector (number of columns in the embedding matrix). |
| padding_idx | int | If specified, the entries at this index in the embedding matrix will be filled with zeros. |

## Return Value

Returns a tensor of shape `(*, embedding_dim)` where `*` is the input shape. For example, if input is of shape `(N)`, the output will be of shape `(N, embedding_dim)`.

## Examples

### Basic Usage with Positional Syntax
```tcl
# Create input indices tensor
set indices [torch::tensor_create {0 1 2} -dtype long]

# Create sparse embedding with 10 embeddings of size 5
set result [torch::sparse_embedding $indices 10 5 -1]
```

### Using Named Parameters
```tcl
# Create input indices tensor
set indices [torch::tensor_create {0 1 2} -dtype long]

# Create sparse embedding with named parameters
set result [torch::sparse_embedding \
    -input $indices \
    -num_embeddings 10 \
    -embedding_dim 5 \
    -padding_idx -1
]
```

### Using Padding Index
```tcl
# Create input indices tensor with padding
set indices [torch::tensor_create {0 1 2 0} -dtype long]

# Create sparse embedding with padding at index 0
set result [torch::sparse_embedding \
    -input $indices \
    -num_embeddings 10 \
    -embedding_dim 5 \
    -padding_idx 0
]
```

### Using CamelCase Alias
```tcl
set indices [torch::tensor_create {0 1 2} -dtype long]
set result [torch::sparseEmbedding \
    -input $indices \
    -num_embeddings 10 \
    -embedding_dim 5 \
    -padding_idx -1
]
```

## Error Conditions

The command will return an error in the following cases:
- Invalid input tensor or tensor not found in storage
- num_embeddings is less than or equal to 0
- embedding_dim is less than or equal to 0
- padding_idx is outside the valid range [-num_embeddings, num_embeddings)
- Missing required parameters
- Unknown parameters provided
- Input tensor is not of type long/int64

## See Also

- [torch::embedding](embedding.md) - Dense embedding layer
- [torch::embedding_bag](embedding_bag.md) - Computes sums, means, or maxes of embeddings 