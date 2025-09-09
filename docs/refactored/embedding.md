# torch::embedding

## Overview

The embedding layer (`torch::embedding`) is a fundamental component in neural networks that maps discrete indices to dense vector representations. It's commonly used in natural language processing for word embeddings, in recommender systems for item embeddings, and in any scenario where categorical data needs to be converted to continuous representations.

## Mathematical Foundation

An embedding layer is essentially a lookup table that maps integer indices to dense vectors:

```
Embedding: {0, 1, 2, ..., num_embeddings-1} → ℝ^embedding_dim
```

Given an index `i`, the embedding layer returns the `i`-th row of the embedding matrix `W ∈ ℝ^{num_embeddings × embedding_dim}`.

### Key Properties
- **Learnable parameters**: The embedding matrix is typically learned during training
- **Differentiable**: Gradients can flow back through embedding lookups
- **Memory efficient**: Only accessed embeddings contribute to computation
- **Sparse gradients**: Only embeddings for selected indices receive gradient updates

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::embedding input num_embeddings embedding_dim [padding_idx]
```

### Named Parameter Syntax
```tcl
torch::embedding -input tensor -num_embeddings int -embedding_dim int [-padding_idx int]
torch::embedding -tensor tensor -num_embeddings int -embedding_dim int [-padding_idx int]
```

### CamelCase Alias
```tcl
torch::Embedding input num_embeddings embedding_dim [padding_idx]
torch::Embedding -input tensor -num_embeddings int -embedding_dim int [-padding_idx int]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input`/`tensor` | tensor | Yes | - | Integer indices tensor |
| `num_embeddings` | int | Yes | - | Size of vocabulary (number of embeddings) |
| `embedding_dim` | int | Yes | - | Dimension of each embedding vector |
| `padding_idx` | int | No | -1 | Index to zero out (typically for padding tokens) |

### Parameter Details

#### input/tensor
- **Type**: Integer tensor (typically int64)
- **Shape**: Any shape (N₁, N₂, ..., Nₖ)
- **Values**: Must be in range [0, num_embeddings-1]
- **Description**: Indices to look up in the embedding table

#### num_embeddings
- **Type**: Positive integer
- **Range**: > 0
- **Description**: Total number of unique embeddings (vocabulary size)
- **Example**: 10000 for a vocabulary of 10,000 words

#### embedding_dim
- **Type**: Positive integer
- **Range**: > 0
- **Description**: Dimension of each embedding vector
- **Common values**: 64, 128, 256, 300, 512, 768

#### padding_idx
- **Type**: Integer
- **Range**: [0, num_embeddings-1] or -1 (disabled)
- **Description**: Index that should always return zero vector
- **Use case**: Padding tokens in variable-length sequences

## Output

Returns a tensor with shape `(*input.shape, embedding_dim)` where each index in the input is replaced by its corresponding embedding vector.

### Shape Transformation
- **Input shape**: (N₁, N₂, ..., Nₖ)
- **Output shape**: (N₁, N₂, ..., Nₖ, embedding_dim)

## Basic Examples

### 1. Simple Word Embedding
```tcl
# Create word indices for "hello world"
# Vocabulary: {0: <pad>, 1: hello, 2: world, 3: <unk>}
set word_ids [torch::tensor_create -data {1 2} -dtype int64 -shape {2}]

# Create embeddings: vocab_size=4, embedding_dim=100
set embeddings [torch::embedding $word_ids 4 100]

# Output shape: [2, 100] - two words, each with 100-dim embedding
puts "Embedding shape: [torch::tensor_shape $embeddings]"
```

### 2. Sentence Batch Processing
```tcl
# Batch of sentences (batch_size=3, max_length=5)
set sentences [torch::tensor_create -data {
    1 2 3 0 0
    4 5 6 7 0  
    8 9 0 0 0
} -dtype int64 -shape {3 5}]

# Create embeddings with padding
set embeddings [torch::embedding $sentences 10 64 0]  # padding_idx=0

# Output shape: [3, 5, 64] - 3 sentences, 5 tokens each, 64-dim embeddings
puts "Batch embedding shape: [torch::tensor_shape $embeddings]"
```

### 3. Named Parameter Syntax
```tcl
# Equivalent operations using named parameters
set indices [torch::tensor_create -data {0 1 2 3} -dtype int64 -shape {4}]

# Method 1: -input
set emb1 [torch::embedding -input $indices -num_embeddings 5 -embedding_dim 32]

# Method 2: -tensor  
set emb2 [torch::embedding -tensor $indices -num_embeddings 5 -embedding_dim 32]

# Method 3: with padding
set emb3 [torch::embedding -input $indices -num_embeddings 5 -embedding_dim 32 -padding_idx 0]
```

## Advanced Examples

### 1. Variable-Length Sequences with Padding
```tcl
# Sequences of different lengths padded to max length
set vocab_size 1000
set embedding_dim 128
set padding_idx 0

# Batch with padding (0 = padding token)
set padded_batch [torch::tensor_create -data {
    15 23 45 67 89 12  0  0  # sequence length 6
    34 56 78  0  0  0  0  0  # sequence length 3  
    91 23 45 67 12 34 78 90  # sequence length 8
} -dtype int64 -shape {3 8}]

set embeddings [torch::embedding $padded_batch $vocab_size $embedding_dim $padding_idx]
# Shape: [3, 8, 128] with padding positions having zero vectors
```

### 2. Multi-dimensional Index Tensors
```tcl
# 3D index tensor (e.g., for hierarchical data)
set indices_3d [torch::tensor_create -data {
    1 2 3
    4 5 6
    7 8 9
    10 11 12
} -dtype int64 -shape {2 2 3}]

set embeddings [torch::embedding $indices_3d 15 64]
# Output shape: [2, 2, 3, 64]
```

### 3. Large Vocabulary Embedding
```tcl
# Large vocabulary (e.g., for language modeling)
set text_indices [torch::tensor_create -data {1001 2034 5678 9876 3421} -dtype int64 -shape {5}]

# Large vocabulary with high-dimensional embeddings
set embeddings [torch::embedding $text_indices 50000 768]  # BERT-like dimensions
# Output shape: [5, 768]
```

## Natural Language Processing Examples

### 1. Word Embeddings
```tcl
# Vocabulary mapping
# 0: <pad>, 1: the, 2: cat, 3: sat, 4: on, 5: mat
set sentence [torch::tensor_create -data {1 2 3 4 1 5} -dtype int64 -shape {6}]

set word_embeddings [torch::embedding $sentence 6 300 0]  # 300-dim word vectors
# "the cat sat on the mat" → [6, 300] tensor
```

### 2. Character-Level Embeddings
```tcl
# Character vocabulary (a-z + space + special tokens)
set char_ids [torch::tensor_create -data {8 5 12 12 15 27 23 15 18 12 4} -dtype int64 -shape {11}]

set char_embeddings [torch::embedding $char_ids 30 50]  # 30 chars, 50-dim each
# "hello world" → [11, 50] tensor
```

### 3. Multilingual Embeddings
```tcl
# Shared vocabulary across languages
set multilingual_ids [torch::tensor_create -data {
    1234 5678 9012  # English: "hello world today"
    3456 7890 1234  # Spanish: "hola mundo hoy"  
    5678 2345 6789  # French: "bonjour monde aujourd"
} -dtype int64 -shape {3 3}]

set embeddings [torch::embedding $multilingual_ids 100000 512]
# Shape: [3, 3, 512] - multilingual sentence embeddings
```

## Recommender System Examples

### 1. User and Item Embeddings
```tcl
# User IDs
set user_ids [torch::tensor_create -data {42 137 891} -dtype int64 -shape {3}]
set user_embeddings [torch::embedding $user_ids 10000 64]  # 10K users, 64-dim

# Item IDs  
set item_ids [torch::tensor_create -data {1023 5047 7789} -dtype int64 -shape {3}]
set item_embeddings [torch::embedding $item_ids 50000 64]  # 50K items, 64-dim

# Both have shape [3, 64] for similarity computation
```

### 2. Categorical Feature Embeddings
```tcl
# Multiple categorical features
set category_ids [torch::tensor_create -data {5 12 3} -dtype int64 -shape {3}]
set brand_ids [torch::tensor_create -data {89 23 156} -dtype int64 -shape {3}]

set category_emb [torch::embedding $category_ids 20 16]   # 20 categories
set brand_emb [torch::embedding $brand_ids 500 32]       # 500 brands

# Concatenate embeddings: [3, 48] total
```

## Computer Vision Examples

### 1. Positional Embeddings
```tcl
# Position indices for transformer vision model
set positions [torch::tensor_create -data {0 1 2 3 4 5 6 7} -dtype int64 -shape {8}]

set pos_embeddings [torch::embedding $positions 196 768]  # 14x14 patches, 768-dim
# For 224x224 image with 16x16 patches
```

### 2. Class Token Embeddings
```tcl
# Special tokens for classification
set class_tokens [torch::tensor_create -data {0} -dtype int64 -shape {1}]  # CLS token
set class_embedding [torch::embedding $class_tokens 1 768]

# Shape: [1, 768] - single classification token embedding
```

## Performance Considerations

### Memory Usage
```tcl
# Memory calculation: num_embeddings × embedding_dim × 4 bytes (float32)
# Example: 50,000 vocab × 512 dim × 4 bytes = ~100MB
set large_vocab_emb [torch::embedding $indices 50000 512]
```

### Optimization Tips

1. **Choose appropriate dimensions**: Balance between expressiveness and memory
   ```tcl
   # Smaller for simple tasks
   set small_emb [torch::embedding $indices 1000 64]
   
   # Larger for complex tasks  
   set large_emb [torch::embedding $indices 100000 768]
   ```

2. **Use padding efficiently**: Minimize padding tokens in batches
   ```tcl
   # Good: minimal padding
   set efficient_batch [torch::tensor_create -data {1 2 3 0} -dtype int64 -shape {4}]
   
   # Less efficient: excessive padding
   set inefficient_batch [torch::tensor_create -data {1 2 0 0 0 0 0 0} -dtype int64 -shape {8}]
   ```

3. **Vocabulary size considerations**: Use appropriate vocabulary sizes
   ```tcl
   # Subword tokenization (smaller vocab)
   set subword_emb [torch::embedding $indices 32000 512]
   
   # Word-level tokenization (larger vocab)
   set word_emb [torch::embedding $indices 100000 300]
   ```

## Error Handling

### Common Errors and Solutions

#### 1. Index Out of Range
```tcl
# Error: index 10 >= vocabulary size 5
set bad_indices [torch::tensor_create -data {0 1 10} -dtype int64 -shape {3}]
catch {torch::embedding $bad_indices 5 64} error
# Solution: Ensure all indices < num_embeddings
```

#### 2. Invalid Dimensions
```tcl
# Error: non-positive embedding dimension
catch {torch::embedding $indices 100 0} error
# Solution: Use positive embedding_dim

# Error: non-positive vocabulary size
catch {torch::embedding $indices 0 64} error  
# Solution: Use positive num_embeddings
```

#### 3. Invalid Tensor Types
```tcl
# Error: float indices (should be integers)
set float_indices [torch::tensor_create -data {1.5 2.3} -dtype float32 -shape {2}]
catch {torch::embedding $float_indices 5 64} error
# Solution: Use integer indices (int64 recommended)
```

## Migration Guide

### From Positional to Named Parameters

**Before:**
```tcl
set embeddings [torch::embedding $indices 1000 128 0]
```

**After:**
```tcl
set embeddings [torch::embedding -input $indices -num_embeddings 1000 -embedding_dim 128 -padding_idx 0]
```

### Benefits of Named Parameters
- **Clarity**: Purpose of each parameter is explicit
- **Flexibility**: Parameters can be specified in any order
- **Safety**: Reduces risk of parameter order mistakes
- **Extensibility**: Easy to add new parameters in future

## Best Practices

### 1. Vocabulary Design
```tcl
# Reserve special indices
# 0: <pad> (padding)
# 1: <unk> (unknown/out-of-vocabulary)  
# 2: <start> (sequence start)
# 3: <end> (sequence end)
# 4+: actual vocabulary
set special_tokens 4
set vocab_size [expr $actual_vocab_size + $special_tokens]
```

### 2. Embedding Dimensions
```tcl
# Common dimension choices
set small_emb 64      # For simple tasks
set medium_emb 128    # General purpose
set large_emb 256     # Complex tasks
set xlarge_emb 512    # Very complex tasks
set pretrained_emb 768 # BERT-like models
```

### 3. Padding Strategy
```tcl
# Always use index 0 for padding
set padding_idx 0

# Consistent padding in batches
proc pad_sequence {sequences max_length} {
    # Implementation to pad sequences to max_length
    # Always use padding_idx for padding
}
```

### 4. Testing Embeddings
```tcl
# Verify embedding shapes
proc test_embedding {indices vocab_size emb_dim} {
    set emb [torch::embedding $indices $vocab_size $emb_dim]
    set input_shape [torch::tensor_shape $indices]
    set output_shape [torch::tensor_shape $emb]
    
    # Expected: output_shape = input_shape + [emb_dim]
    puts "Input: $input_shape → Output: $output_shape"
}
```

## Integration Examples

### 1. With Attention Mechanisms
```tcl
# Token embeddings + positional embeddings
set token_ids [torch::tensor_create -data {1 5 3 8} -dtype int64 -shape {4}]
set positions [torch::tensor_create -data {0 1 2 3} -dtype int64 -shape {4}]

set token_emb [torch::embedding $token_ids 1000 512]
set pos_emb [torch::embedding $positions 100 512]

# Combine embeddings (element-wise addition)
set combined_emb [torch::tensor_add $token_emb $pos_emb]
```

### 2. With RNN/LSTM
```tcl
# Sequence of token embeddings for RNN processing
set sequence [torch::tensor_create -data {12 45 67 23 89} -dtype int64 -shape {5}]
set embeddings [torch::embedding $sequence 2000 256]

# Shape: [5, 256] - ready for RNN input (sequence_length, embedding_dim)
```

### 3. With Convolutional Layers
```tcl
# Character-level embeddings for CNN
set chars [torch::tensor_create -data {
    1 5 12 12 15 27 23 15 18 12 4 0 0 0 0 0
} -dtype int64 -shape {16}]  # Fixed-length character sequence

set char_emb [torch::embedding $chars 30 16]  # 30 chars, 16-dim each
# Shape: [16, 16] - ready for 1D convolution
```

## Comparison with Alternatives

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| One-hot encoding | Small vocabularies | Simple, interpretable | Memory intensive, no learned relationships |
| Dense embeddings | Large vocabularies | Memory efficient, learnable | Requires training |
| Pre-trained embeddings | Transfer learning | Rich representations | May not fit domain |
| Factorized embeddings | Very large vocabularies | Memory efficient | More complex |

## Implementation Details

### Embedding Matrix Creation
The implementation creates a random embedding matrix using `torch::randn({num_embeddings, embedding_dim})` and optionally zeros out the padding index.

### Lookup Operation
Uses PyTorch's `torch::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse)` function with default values for the last two parameters.

### Memory Layout
- **Weight matrix**: Shape [num_embeddings, embedding_dim]
- **Lookup result**: Preserves input shape with embedding dimension appended

## See Also

- `torch::tensor_create` - Creating input index tensors
- `torch::tensor_add` - Combining embeddings (e.g., token + positional)
- `torch::tensor_matmul` - Computing similarities between embeddings
- `torch::tensor_sum` - Pooling embeddings
- `torch::tensor_mean` - Average pooling embeddings
- `torch::embedding_bag` - Efficient embedding aggregation 