# torch::positional_encoding

Creates a positional encoding tensor for transformer models. The positional encoding is used to inject sequence order information into transformer models since they don't inherently have a notion of position.

## Syntax

### Legacy Syntax (Positional Parameters)
```tcl
torch::positional_encoding seq_len d_model dropout
```

### Modern Syntax (Named Parameters)
```tcl
torch::positional_encoding -seqLen seq_len -dModel d_model -dropout dropout
```

### CamelCase Alias
```tcl
torch::positionalEncoding -seqLen seq_len -dModel d_model -dropout dropout
```

## Parameters

- `seq_len` or `-seqLen` (int)
  - The length of the sequence (number of positions)
  - Must be a positive integer

- `d_model` or `-dModel` (int)
  - The number of expected features in the encoder/decoder inputs
  - Must be a positive even integer (since the implementation uses sin/cos pairs)

- `dropout` or `-dropout` (float)
  - The dropout value to apply to the positional encoding
  - Must be in range [0.0, 1.0]

## Return Value

Returns a tensor of shape `[seq_len, d_model]` containing the positional encodings.

## Examples

### Using Legacy Syntax
```tcl
# Create positional encoding for sequence length 10 and 512 features
set pe [torch::positional_encoding 10 512 0.1]
```

### Using Modern Syntax
```tcl
# Same operation with named parameters
set pe [torch::positional_encoding -seqLen 10 -dModel 512 -dropout 0.1]
```

### Using CamelCase Alias
```tcl
# Same operation with camelCase alias
set pe [torch::positionalEncoding -seqLen 10 -dModel 512 -dropout 0.1]
```

## Implementation Details

The positional encoding is computed using sine and cosine functions of different frequencies:

- For even indices: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- For odd indices: PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
- pos is the position in the sequence (0 to seq_len-1)
- i is the dimension index (0 to d_model/2-1)

After computing the positional encoding matrix, dropout is applied with the specified probability.

## Error Conditions

- Returns error if any required parameter is missing
- Returns error if seq_len is not a positive integer
- Returns error if d_model is not a positive even integer
- Returns error if dropout is not in range [0.0, 1.0]

## See Also

- [torch::transformer_encoder_layer](transformer_encoder_layer.md)
- [torch::transformer_decoder_layer](transformer_decoder_layer.md)
- [torch::multihead_attention](multihead_attention.md) 