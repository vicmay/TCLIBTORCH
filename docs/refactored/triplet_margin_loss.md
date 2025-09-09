# torch::triplet_margin_loss

**Triplet Margin Loss**

Computes the triplet margin loss between anchor, positive, and negative tensors. This loss function is widely used in metric learning, face recognition, and similarity learning tasks to learn embeddings where similar items are closer together and dissimilar items are farther apart.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::triplet_margin_loss -anchor tensor -positive tensor -negative tensor ?-margin double? ?-p double? ?-reduction string?
torch::tripletMarginLoss -anchor tensor -positive tensor -negative tensor ?-margin double? ?-p double? ?-reduction string?
```

### Positional Syntax (Legacy)
```tcl
torch::triplet_margin_loss anchor positive negative ?margin? ?p? ?reduction?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **anchor** | tensor | - | Anchor tensor (reference embeddings) |
| **positive** | tensor | - | Positive tensor (similar to anchor) |
| **negative** | tensor | - | Negative tensor (dissimilar to anchor) |
| **margin** | double | 1.0 | Margin value for separation between positive and negative pairs |
| **p** | double | 2.0 | The norm degree for pairwise distance (1=L1, 2=L2) |
| **reduction** | string | "mean" | Reduction mode: "none", "mean", or "sum" |

## Returns

Returns a tensor handle containing the computed triplet margin loss.

- If `reduction="none"`: Returns tensor with same batch size as input
- If `reduction="mean"`: Returns scalar tensor with mean loss
- If `reduction="sum"`: Returns scalar tensor with sum of losses

## Mathematical Details

The triplet margin loss is computed as:

```
triplet_loss = max(0, ||anchor - positive||_p - ||anchor - negative||_p + margin)
```

Where:
- `||·||_p` denotes the p-norm distance
- `anchor`, `positive`, `negative` are embedding vectors
- `margin` is the minimum desired separation

### Key Properties:
- **Metric Learning**: Learns embeddings that preserve semantic relationships
- **Triplet-based**: Uses three samples: anchor, positive (similar), negative (dissimilar)
- **Margin-based**: Enforces minimum separation between positive and negative pairs
- **Distance-aware**: Uses configurable p-norm for distance computation
- **Zero loss optimum**: Loss is zero when negatives are sufficiently far from positives

### Learning Objective:
The loss encourages:
1. **Attraction**: Anchor and positive embeddings to be close
2. **Repulsion**: Anchor and negative embeddings to be far apart
3. **Margin enforcement**: Negative distance > positive distance + margin

## Examples

### Named Parameter Syntax

#### Basic Usage
```tcl
# Create anchor tensor (reference embedding)
set anchor [torch::tensorCreate -data {1.0 0.5 -0.2 0.8} -shape {1 4}]

# Create positive tensor (similar to anchor)
set positive [torch::tensorCreate -data {0.9 0.6 -0.1 0.7} -shape {1 4}]

# Create negative tensor (dissimilar to anchor)
set negative [torch::tensorCreate -data {-0.8 1.5 0.9 -1.2} -shape {1 4}]

# Compute triplet margin loss
set loss [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative]
```

#### With Custom Parameters
```tcl
# Use different margin and L1 norm
set loss [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -margin 0.5 -p 1.0 -reduction "sum"]
```

#### Using camelCase Alias
```tcl
# Equivalent using camelCase alias
set loss [torch::tripletMarginLoss -anchor $anchor -positive $positive -negative $negative -margin 2.0]
```

### Positional Syntax (Legacy)

#### Basic Usage
```tcl
# Basic usage with defaults (margin=1.0, p=2.0, reduction=mean)
set loss [torch::triplet_margin_loss $anchor $positive $negative]

# With custom margin
set loss [torch::triplet_margin_loss $anchor $positive $negative 0.5]

# With margin and p-norm
set loss [torch::triplet_margin_loss $anchor $positive $negative 1.0 1.0]

# With all parameters (reduction: 0=none, 1=mean, 2=sum)
set loss [torch::triplet_margin_loss $anchor $positive $negative 1.0 2.0 0]
```

## Use Cases

### Face Recognition
```tcl
# Face verification: same person vs different person
set person_a_photo1 [torch::tensorCreate -data {0.8 -0.2 0.5 -0.1 0.9 -0.3 0.7 0.4} -shape {1 8}]
set person_a_photo2 [torch::tensorCreate -data {0.7 -0.1 0.4 -0.2 0.8 -0.4 0.6 0.3} -shape {1 8}]
set person_b_photo [torch::tensorCreate -data {-0.3 0.9 -0.7 0.8 -0.2 0.6 -0.5 -0.9} -shape {1 8}]

set face_loss [torch::triplet_margin_loss -anchor $person_a_photo1 -positive $person_a_photo2 -negative $person_b_photo -margin 0.2]
```

### Word Embeddings
```tcl
# Word similarity learning: synonyms vs unrelated words
set word_main [torch::tensorCreate -data {0.5 -0.8 0.3 0.7 -0.4} -shape {1 5}]
set word_synonym [torch::tensorCreate -data {0.6 -0.7 0.2 0.8 -0.3} -shape {1 5}]
set word_unrelated [torch::tensorCreate -data {-0.9 0.4 -0.6 -0.2 0.8} -shape {1 5}]

set word_loss [torch::triplet_margin_loss -anchor $word_main -positive $word_synonym -negative $word_unrelated -p 1.0 -margin 0.5]
```

### Product Recommendation
```tcl
# Product similarity: user preferences
set user_profile [torch::tensorCreate -data {0.3 0.7 -0.2 0.8 0.1} -shape {1 5}]
set liked_product [torch::tensorCreate -data {0.4 0.6 -0.1 0.9 0.2} -shape {1 5}]
set disliked_product [torch::tensorCreate -data {-0.7 -0.3 0.8 -0.1 -0.9} -shape {1 5}]

set rec_loss [torch::triplet_margin_loss -anchor $user_profile -positive $liked_product -negative $disliked_product]
```

### Image Retrieval
```tcl
# Image similarity: query image vs relevant/irrelevant results
set query_image [torch::tensorCreate -data {0.2 0.8 -0.3 0.6 -0.1 0.9} -shape {1 6}]
set relevant_image [torch::tensorCreate -data {0.3 0.7 -0.2 0.7 0.0 0.8} -shape {1 6}]
set irrelevant_image [torch::tensorCreate -data {-0.8 0.1 0.9 -0.4 0.7 -0.6} -shape {1 6}]

set retrieval_loss [torch::triplet_margin_loss -anchor $query_image -positive $relevant_image -negative $irrelevant_image -margin 0.3]
```

## Advanced Examples

### Batch Processing
```tcl
# Process batch of triplets simultaneously
set batch_anchors [torch::tensorCreate -data {1.0 0.5 -0.2 0.8 0.3 -0.7 0.9 -0.1 -0.4 0.6 0.2 -0.5} -shape {3 4}]
set batch_positives [torch::tensorCreate -data {0.9 0.6 -0.1 0.7 0.4 -0.6 0.8 0.0 -0.3 0.7 0.3 -0.4} -shape {3 4}]
set batch_negatives [torch::tensorCreate -data {-0.8 1.5 0.9 -1.2 -0.9 0.8 -0.5 1.3 0.8 -0.9 -0.7 1.0} -shape {3 4}]

set batch_loss [torch::triplet_margin_loss -anchor $batch_anchors -positive $batch_positives -negative $batch_negatives -reduction "mean"]
```

### Hard Negative Mining
```tcl
# Simulate hard negative mining (challenging negatives)
set anchor [torch::tensorCreate -data {0.5 0.3 -0.1 0.7}]
set positive [torch::tensorCreate -data {0.6 0.4 0.0 0.8}]  # Very similar to anchor

# Easy negative (very different)
set easy_negative [torch::tensorCreate -data {-0.9 -0.8 0.9 -0.7}]
set easy_loss [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $easy_negative]

# Hard negative (somewhat similar to anchor, but should be pushed away)
set hard_negative [torch::tensorCreate -data {0.3 0.1 -0.3 0.5}]
set hard_loss [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $hard_negative]
```

### Different Distance Metrics
```tcl
# Compare L1 vs L2 distance metrics
set anchor [torch::tensorCreate -data {1.0 0.5 -0.2 0.8} -shape {1 4}]
set positive [torch::tensorCreate -data {0.9 0.6 -0.1 0.7} -shape {1 4}]
set negative [torch::tensorCreate -data {-0.8 1.5 0.9 -1.2} -shape {1 4}]

# L1 norm (Manhattan distance)
set l1_loss [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -p 1.0]

# L2 norm (Euclidean distance)  
set l2_loss [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -p 2.0]
```

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::triplet_margin_loss -anchor $anchor -positive $positive} error
# Error: Required parameters -anchor, -positive, and -negative must be provided

# Invalid tensor names
catch {torch::triplet_margin_loss -anchor "invalid" -positive $positive -negative $negative} error
# Error: Invalid anchor tensor name

# Unknown parameters
catch {torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -unknown value} error
# Error: Unknown parameter: -unknown
```

## Performance Notes

- **Computational Complexity**: O(n × d) where n is batch size and d is embedding dimension
- **Memory Usage**: Scales linearly with batch size and embedding dimensions
- **GPU Acceleration**: Fully supports CUDA tensors and GPU computation
- **Gradient Computation**: Supports automatic differentiation for training

## Mathematical Properties

### Loss Characteristics:
- **Non-negative**: Loss ≥ 0 always
- **Zero optimum**: Loss = 0 when margin constraint is satisfied
- **Piecewise linear**: Max operation creates non-smooth regions
- **Translation invariant**: Adding constants to all embeddings doesn't change loss

### Distance Metrics:
| p-value | Distance Type | Characteristics |
|---------|---------------|-----------------|
| **1.0** | Manhattan (L1) | Robust to outliers, sparse gradients |
| **2.0** | Euclidean (L2) | Standard choice, smooth gradients |

## Comparison with Other Loss Functions

| Loss Function | Use Case | Triplet Structure | Margin |
|---------------|----------|-------------------|--------|
| **Triplet Margin** | Metric learning | Yes (A,P,N) | Hard margin |
| **Contrastive** | Pairwise similarity | No (pairs only) | Soft margin |
| **Center Loss** | Class separation | No (class centers) | No margin |
| **ArcFace** | Face recognition | No (angular) | Angular margin |

## Training Tips

### Effective Margin Selection:
- **Small datasets**: Use larger margins (1.0-2.0)
- **Large datasets**: Use smaller margins (0.1-0.5)
- **High-dimensional embeddings**: Consider smaller margins
- **Low-dimensional embeddings**: Can use larger margins

### Triplet Selection Strategies:
1. **Random sampling**: Simple but may not converge well
2. **Hard negative mining**: Select challenging negatives
3. **Semi-hard mining**: Select moderately difficult examples
4. **Batch-all**: Use all valid triplets in batch

## Compatibility

- **Backward Compatible**: Original positional syntax remains fully supported
- **Thread Safe**: Can be used safely in multi-threaded environments
- **Device Agnostic**: Works with CPU and CUDA tensors
- **Data Types**: Supports float32, float64, and other floating-point types

## See Also

- [`torch::margin_ranking_loss`](margin_ranking_loss.md) - Ranking-based margin loss
- [`torch::cosine_embedding_loss`](cosine_embedding_loss.md) - Cosine similarity loss
- [`torch::contrastive_loss`](contrastive_loss.md) - Pairwise contrastive loss (if available)
- [`torch::center_loss`](center_loss.md) - Center-based loss (if available)

## Migration Guide

### From Positional to Named Parameters

```tcl
# OLD (Positional)
set loss [torch::triplet_margin_loss $anchor $positive $negative 0.5 1.0 1]

# NEW (Named Parameters)
set loss [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative -margin 0.5 -p 1.0 -reduction "mean"]

# NEW (camelCase)
set loss [torch::tripletMarginLoss -anchor $anchor -positive $positive -negative $negative -margin 0.5 -p 1.0 -reduction "mean"]
```

### From Other Similarity Loss Functions

#### From Contrastive Loss
```tcl
# OLD: Contrastive loss (pairwise)
# set loss [torch::contrastive_loss $embedding1 $embedding2 $label $margin]

# NEW: Triplet margin loss (requires explicit triplet)
set loss [torch::triplet_margin_loss -anchor $embedding1 -positive $embedding2 -negative $negative_embedding -margin $margin]
```

### Benefits of Named Parameters

1. **Self-Documenting**: Parameter names clarify the triplet structure
2. **Flexible Order**: Parameters can be specified in any order
3. **Less Error-Prone**: Clear distinction between anchor, positive, and negative
4. **Better Maintainability**: Code is easier to understand and modify
5. **IDE Support**: Better autocomplete and parameter hints

### Triplet Construction Guidelines

```tcl
# Helper function for triplet construction
proc create_triplet {anchor_data positive_data negative_data embedding_dim} {
    set anchor [torch::tensorCreate -data $anchor_data -shape [list 1 $embedding_dim]]
    set positive [torch::tensorCreate -data $positive_data -shape [list 1 $embedding_dim]]
    set negative [torch::tensorCreate -data $negative_data -shape [list 1 $embedding_dim]]
    
    return [list $anchor $positive $negative]
}

# Usage
set triplet [create_triplet {0.5 0.3 -0.1 0.7} {0.6 0.4 0.0 0.8} {-0.9 -0.8 0.9 -0.7} 4]
set anchor [lindex $triplet 0]
set positive [lindex $triplet 1] 
set negative [lindex $triplet 2]

set loss [torch::triplet_margin_loss -anchor $anchor -positive $positive -negative $negative]
```

## Real-World Applications

### 1. Face Recognition System
```tcl
# Training phase: Learn face embeddings
proc train_face_embeddings {face_batch_anchors face_batch_positives face_batch_negatives} {
    set loss [torch::triplet_margin_loss -anchor $face_batch_anchors -positive $face_batch_positives -negative $face_batch_negatives -margin 0.2 -reduction "mean"]
    return $loss
}
```

### 2. Recommendation Engine
```tcl
# Product similarity learning
proc learn_product_similarity {user_profiles liked_products disliked_products} {
    set loss [torch::triplet_margin_loss -anchor $user_profiles -positive $liked_products -negative $disliked_products -margin 1.0 -p 1.0]
    return $loss
}
```

### 3. Image Retrieval
```tcl
# Content-based image retrieval
proc train_image_retrieval {query_features relevant_features irrelevant_features} {
    set loss [torch::triplet_margin_loss -anchor $query_features -positive $relevant_features -negative $irrelevant_features -margin 0.5]
    return $loss
} 