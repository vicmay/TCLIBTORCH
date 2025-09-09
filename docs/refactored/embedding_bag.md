# torch::embedding_bag

## Overview

Embedding bag (`torch::embedding_bag`) is an efficient aggregation operation that combines multiple embeddings into a single representation. It's commonly used in recommender systems, natural language processing, and any application where you need to aggregate variable-length sequences of embeddings using sum, mean, or max operations.

Unlike regular embedding lookups that return individual vectors, embedding bag operations group indices into "bags" and aggregate the corresponding embeddings, making it highly efficient for sparse feature scenarios.

## Mathematical Foundation

The embedding bag operation performs the following steps:

1. **Lookup**: For each index `i` in the input, retrieve embedding vector `E[i]` from the weight matrix
2. **Grouping**: Group embeddings according to the offsets tensor to form bags
3. **Aggregation**: Apply the specified aggregation function (sum, mean, or max) within each bag

### Aggregation Modes

| Mode | Value | Operation | Formula |
|------|-------|-----------|---------|
| Sum | 0 | Element-wise sum | `bag_result = Σ E[i]` for i in bag |
| Mean | 1 | Element-wise average | `bag_result = (1/n) Σ E[i]` for i in bag |
| Max | 2 | Element-wise maximum | `bag_result = max(E[i])` for i in bag |

### Bag Structure

Bags are defined by the `offsets` tensor which specifies the start position of each bag:
- `offsets[0]` = start of bag 0
- `offsets[1]` = start of bag 1 (end of bag 0)
- `offsets[k]` = start of bag k (end of bag k-1)
- The last bag extends to the end of the input tensor

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::embedding_bag input weight offsets mode [per_sample_weights]
```

### Named Parameter Syntax
```tcl
torch::embedding_bag -input tensor -weight tensor -offsets tensor -mode int [-per_sample_weights tensor]
```

### CamelCase Alias
```tcl
torch::embeddingBag input weight offsets mode [per_sample_weights]
torch::embeddingBag -input tensor -weight tensor -offsets tensor -mode int [-per_sample_weights tensor]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input` | tensor | Yes | - | Integer indices tensor (int64) |
| `weight` | tensor | Yes | - | Embedding weight matrix (float) |
| `offsets` | tensor | Yes | - | Bag start positions (int64) |
| `mode` | int | Yes | - | Aggregation mode: 0=sum, 1=mean, 2=max |
| `per_sample_weights` | tensor | No | - | Optional weights for each index |

### Parameter Details

#### input
- **Type**: Integer tensor (int64)
- **Shape**: (N,) - 1D tensor of indices
- **Values**: Must be in range [0, weight.size(0)-1]
- **Description**: Flat tensor of embedding indices to look up

#### weight
- **Type**: Float tensor
- **Shape**: (num_embeddings, embedding_dim)
- **Description**: Pre-trained or learnable embedding matrix

#### offsets
- **Type**: Integer tensor (int64)
- **Shape**: (num_bags,) - 1D tensor
- **Values**: Monotonically increasing indices
- **Description**: Start position of each bag in the input tensor

#### mode
- **Type**: Integer
- **Values**: 0 (sum), 1 (mean), 2 (max)
- **Description**: Aggregation function to apply within each bag

#### per_sample_weights (Optional)
- **Type**: Float tensor
- **Shape**: Same as input tensor
- **Description**: Individual weights for each embedding lookup (only for sum/mean modes)

## Output

Returns a tensor with shape `(num_bags, embedding_dim)` where each row represents the aggregated embedding for one bag.

### Shape Transformation
- **Input**: (N,) indices + (num_bags,) offsets + (num_embeddings, embedding_dim) weight
- **Output**: (num_bags, embedding_dim)

## Basic Examples

### 1. Simple Sum Aggregation
```tcl
# Create embeddings for 4 words, aggregate into 2 bags
set indices [torch::tensor_create -data {0 1 2 1} -dtype int64 -shape {4}]
set weight [torch::tensor_create -data {
    1.0 2.0
    3.0 4.0  
    5.0 6.0
} -dtype float32 -shape {3 2}]

# Two bags: [0,1] and [2,1] 
set offsets [torch::tensor_create -data {0 2} -dtype int64 -shape {2}]

# Sum aggregation (mode 0)
set result [torch::embedding_bag $indices $weight $offsets 0]
# Shape: [2, 2] - two aggregated embeddings

puts "Result shape: [torch::tensor_shape $result]"
# Bag 0: sum of embeddings 0 and 1 → [1+3, 2+4] = [4, 6]
# Bag 1: sum of embeddings 2 and 1 → [5+3, 6+4] = [8, 10]
```

### 2. Mean Aggregation for Variable-Length Sequences
```tcl
# Document representation: average word embeddings per sentence
set word_indices [torch::tensor_create -data {5 12 8 3 15 9} -dtype int64 -shape {6}]
set word_embeddings [torch::tensor_create -data [string repeat "1.0 2.0 3.0 " 20] -dtype float32 -shape {20 3}]

# Three sentences: lengths 2, 3, 1
set sentence_offsets [torch::tensor_create -data {0 2 5} -dtype int64 -shape {3}]

# Mean aggregation (mode 1) 
set sentence_embeddings [torch::embedding_bag $word_indices $word_embeddings $sentence_offsets 1]
# Shape: [3, 3] - three sentence embeddings

puts "Sentence embeddings shape: [torch::tensor_shape $sentence_embeddings]"
```

### 3. Named Parameter Syntax
```tcl
# Equivalent using named parameters
set indices [torch::tensor_create -data {0 1 2 3} -dtype int64 -shape {4}]
set weight [torch::tensor_create -data [string repeat "0.5 " 20] -dtype float32 -shape {5 4}]
set offsets [torch::tensor_create -data {0 2} -dtype int64 -shape {2}]

set result [torch::embedding_bag -input $indices -weight $weight -offsets $offsets -mode 2]
# Max aggregation with named parameters
```

## Advanced Examples

### 1. Recommender System: User-Item Interactions
```tcl
# User interaction history - items they've interacted with
set user_items [torch::tensor_create -data {101 205 78 156 92 301 45} -dtype int64 -shape {7}]

# Pre-trained item embeddings (1000 items, 64-dim embeddings)
set item_embeddings [torch::tensor_create -data [string repeat "0.1 " 64000] -dtype float32 -shape {1000 64}]

# Three users with different numbers of interactions: 3, 2, 2 items
set user_offsets [torch::tensor_create -data {0 3 5} -dtype int64 -shape {3}]

# Mean aggregation to get user profiles
set user_profiles [torch::embedding_bag $user_items $item_embeddings $user_offsets 1]
# Shape: [3, 64] - profile for each user based on their interaction history

puts "User profiles shape: [torch::tensor_shape $user_profiles]"
```

### 2. Text Classification: Document Embeddings
```tcl
# Multiple documents represented as bag of words
set word_ids [torch::tensor_create -data {
    45 123 67 890 234 12 678 345 789 23 456 78 901 234 567
} -dtype int64 -shape {15}]

# Pre-trained word embeddings (vocabulary=1000, dim=300)
set word_vectors [torch::tensor_create -data [string repeat "0.01 " 300000] -dtype float32 -shape {1000 300}]

# Four documents with lengths: 4, 5, 3, 3 words
set doc_offsets [torch::tensor_create -data {0 4 9 12} -dtype int64 -shape {4}]

# Sum aggregation for document vectors (common in text classification)
set doc_embeddings [torch::embedding_bag $word_ids $word_vectors $doc_offsets 0]
# Shape: [4, 300] - one vector per document

puts "Document embeddings shape: [torch::tensor_shape $doc_embeddings]"
```

## Error Handling

### Common Errors and Solutions

#### 1. Index Out of Bounds
```tcl
# Error: indices exceed weight matrix size
set bad_indices [torch::tensor_create -data {0 1 100} -dtype int64 -shape {3}]
set small_weight [torch::tensor_create -data {1.0 2.0} -dtype float32 -shape {1 2}]
set offsets [torch::tensor_create -data {0} -dtype int64 -shape {1}]

catch {torch::embedding_bag $bad_indices $small_weight $offsets 0} error
# Solution: Ensure all indices < weight.size(0)
```

#### 2. Invalid Aggregation Mode
```tcl
# Error: invalid mode value
set indices [torch::tensor_create -data {0 1} -dtype int64 -shape {2}]
set weight [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -shape {2 2}]
set offsets [torch::tensor_create -data {0} -dtype int64 -shape {1}]

catch {torch::embedding_bag $indices $weight $offsets 5} error  # Invalid mode
# Solution: Use mode 0 (sum), 1 (mean), or 2 (max)
```

## Migration Guide

### From Positional to Named Parameters

**Before:**
```tcl
set result [torch::embedding_bag $indices $weight $offsets 1]
```

**After:**
```tcl
set result [torch::embedding_bag -input $indices -weight $weight -offsets $offsets -mode 1]
```

### Benefits of Named Parameters
- **Clarity**: Each parameter's purpose is explicit
- **Flexibility**: Parameters can be specified in any order
- **Safety**: Reduces risk of parameter confusion
- **Extensibility**: Easy to add new parameters in future versions

## Best Practices

### 1. Bag Size Considerations
```tcl
# Avoid extremely unbalanced bag sizes
# Good: similar bag sizes for batch processing
set balanced_offsets [torch::tensor_create -data {0 3 6 9} -dtype int64 -shape {4}]

# Less efficient: highly variable bag sizes
set unbalanced_offsets [torch::tensor_create -data {0 1 50 51} -dtype int64 -shape {4}]
```

### 2. Aggregation Mode Selection
```tcl
# Sum: when magnitude matters (e.g., term frequency)
set tf_aggregation [torch::embedding_bag $indices $weight $offsets 0]

# Mean: when normalization is important (e.g., user preferences)
set preference_aggregation [torch::embedding_bag $indices $weight $offsets 1]

# Max: when selecting strongest features (e.g., max pooling)
set feature_selection [torch::embedding_bag $indices $weight $offsets 2]
```

## See Also

- `torch::embedding` - Individual embedding lookups
- `torch::tensor_sum` - Manual sum aggregation
- `torch::tensor_mean` - Manual mean aggregation  
- `torch::tensor_max` - Manual max aggregation
- `torch::tensor_create` - Creating input tensors
- `torch::tensor_matmul` - Computing similarities between embeddings 