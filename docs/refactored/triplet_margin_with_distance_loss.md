# torch::triplet_margin_with_distance_loss

**Triplet Margin Loss with Custom Distance Function**

Computes the triplet margin loss between anchor, positive, and negative tensors using a customizable distance function. This extends standard triplet margin loss by allowing different distance metrics (cosine, pairwise L2, Euclidean) for flexible similarity learning.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::triplet_margin_with_distance_loss -anchor tensor -positive tensor -negative tensor ?-distanceFunction string? ?-margin double? ?-reduction string?
torch::tripletMarginWithDistanceLoss -anchor tensor -positive tensor -negative tensor ?-distanceFunction string? ?-margin double? ?-reduction string?
```

### Positional Syntax (Legacy)
```tcl
torch::triplet_margin_with_distance_loss anchor positive negative ?distance_function? ?margin? ?reduction?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **anchor** | tensor | - | Anchor tensor (reference embeddings) |
| **positive** | tensor | - | Positive tensor (similar to anchor) |
| **negative** | tensor | - | Negative tensor (dissimilar to anchor) |
| **distanceFunction** | string | "euclidean" | Distance metric: "cosine", "pairwise", or "euclidean" |
| **margin** | double | 1.0 | Margin value for separation |
| **reduction** | string | "mean" | Reduction mode: "none", "mean", or "sum" |

### Distance Functions

| Function | Description | Best For |
|----------|-------------|----------|
| **"cosine"** | 1 - cosine_similarity | Normalized embeddings, face recognition |
| **"pairwise"** | L2 pairwise distance | General metric learning |
| **"euclidean"** | L2 norm difference | Fast computation, raw features |

## Examples

### Named Parameter Syntax
```tcl
# Create triplet tensors
set anchor [torch::tensorCreate -data {1.0 0.5 -0.2 0.8} -shape {1 4}]
set positive [torch::tensorCreate -data {0.9 0.6 -0.1 0.7} -shape {1 4}]
set negative [torch::tensorCreate -data {-0.8 1.5 0.9 -1.2} -shape {1 4}]

# Basic usage with default euclidean distance
set loss [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative]

# Face recognition with cosine distance
set loss_cosine [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative -distanceFunction "cosine" -margin 0.2]

# Using camelCase alias
set loss_camel [torch::tripletMarginWithDistanceLoss -anchor $anchor -positive $positive -negative $negative -distanceFunction "pairwise" -margin 0.5]
```

### Positional Syntax (Legacy)
```tcl
# Basic usage (distance_function: 0=cosine, 1=pairwise, 2=euclidean)
set loss [torch::triplet_margin_with_distance_loss $anchor $positive $negative]
set loss_cosine [torch::triplet_margin_with_distance_loss $anchor $positive $negative 0 0.2 1]
set loss_pairwise [torch::triplet_margin_with_distance_loss $anchor $positive $negative 1 0.5 1]
```

## Use Cases

### Face Recognition
```tcl
# Cosine distance for normalized face embeddings
set face_loss [torch::triplet_margin_with_distance_loss -anchor $person_a_photo1 -positive $person_a_photo2 -negative $person_b_photo -distanceFunction "cosine" -margin 0.1]
```

### Text Embeddings
```tcl
# Euclidean distance for word embeddings
set text_loss [torch::triplet_margin_with_distance_loss -anchor $word_main -positive $word_synonym -negative $word_unrelated -distanceFunction "euclidean" -margin 0.5]
```

### General Metric Learning
```tcl
# Pairwise distance for general embeddings
set metric_loss [torch::triplet_margin_with_distance_loss -anchor $user_profile -positive $liked_item -negative $disliked_item -distanceFunction "pairwise" -margin 1.0]
```

## Mathematical Details

The loss computes:
```
triplet_loss = max(0, distance(anchor, positive) - distance(anchor, negative) + margin)
```

Where distance functions are:
- **Cosine**: `1 - cosine_similarity(anchor, positive)`
- **Pairwise**: `||anchor - positive||_2` (with broadcasting)
- **Euclidean**: `||anchor - positive||_2` (direct norm)

## Error Handling

```tcl
# Missing parameters
catch {torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive} error
# Error: Required parameters -anchor, -positive, and -negative must be provided

# Invalid tensor
catch {torch::triplet_margin_with_distance_loss -anchor "invalid" -positive $positive -negative $negative} error
# Error: Invalid anchor tensor name

# Unknown parameter
catch {torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative -unknown value} error
# Error: Unknown parameter: -unknown
```

## Performance Notes

- **Cosine Distance**: Additional normalization computation
- **Pairwise Distance**: Standard L2 with broadcasting
- **Euclidean Distance**: Fastest option
- **GPU Support**: All distance functions support CUDA acceleration
- **Gradients**: All functions support automatic differentiation

## Distance Function Selection

### Cosine Distance
- **Use for**: Normalized embeddings, face recognition, text similarity
- **Margin range**: 0.1-0.3 (bounded to [0,2])
- **Properties**: Scale-invariant, measures angular separation

### Pairwise Distance
- **Use for**: General metric learning, standard embeddings
- **Margin range**: 0.5-2.0 (unbounded)
- **Properties**: Magnitude-sensitive, standard choice

### Euclidean Distance
- **Use for**: Fast computation, simple L2 similarity
- **Margin range**: 0.5-2.0 (unbounded)
- **Properties**: Direct norm computation, fastest

## Migration Guide

### From Positional to Named Parameters
```tcl
# OLD (Positional - cosine distance, margin 0.5, mean reduction)
set loss [torch::triplet_margin_with_distance_loss $anchor $positive $negative 0 0.5 1]

# NEW (Named Parameters)
set loss [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative -distanceFunction "cosine" -margin 0.5 -reduction "mean"]

# NEW (camelCase)
set loss [torch::tripletMarginWithDistanceLoss -anchor $anchor -positive $positive -negative $negative -distanceFunction "cosine" -margin 0.5 -reduction "mean"]
```

### From Standard Triplet Loss
```tcl
# OLD: Fixed L2 distance
# set loss [torch::triplet_margin_loss $anchor $positive $negative 1.0 2.0 1]

# NEW: Equivalent with euclidean distance
set loss [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative -distanceFunction "euclidean" -margin 1.0 -reduction "mean"]

# NEW: Better for normalized embeddings
set loss [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative -distanceFunction "cosine" -margin 0.2 -reduction "mean"]
```

## Compatibility

- **Backward Compatible**: Original positional syntax fully supported
- **Thread Safe**: Multi-threaded environment support
- **Device Agnostic**: CPU and CUDA tensor support
- **Data Types**: float32, float64, and other floating-point types
- **Gradient Support**: All distance functions support automatic differentiation

## See Also

- [`torch::triplet_margin_loss`](triplet_margin_loss.md) - Standard triplet margin loss
- [`torch::cosine_embedding_loss`](cosine_embedding_loss.md) - Cosine-based loss
- [`torch::margin_ranking_loss`](margin_ranking_loss.md) - Ranking-based loss
- [`torch::pairwise_distance`](pairwise_distance.md) - Pairwise distance computation (if available)

## Migration Guide

### From Positional to Named Parameters
```tcl
# OLD (Positional - cosine distance, margin 0.5, mean reduction)
set loss [torch::triplet_margin_with_distance_loss $anchor $positive $negative 0 0.5 1]

# NEW (Named Parameters)
set loss [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative -distanceFunction "cosine" -margin 0.5 -reduction "mean"]

# NEW (camelCase)
set loss [torch::tripletMarginWithDistanceLoss -anchor $anchor -positive $positive -negative $negative -distanceFunction "cosine" -margin 0.5 -reduction "mean"]
```

### From Standard Triplet Loss
```tcl
# OLD: Fixed L2 distance
# set loss [torch::triplet_margin_loss $anchor $positive $negative 1.0 2.0 1]

# NEW: Equivalent with euclidean distance
set loss [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative -distanceFunction "euclidean" -margin 1.0 -reduction "mean"]

# NEW: Better for normalized embeddings
set loss [torch::triplet_margin_with_distance_loss -anchor $anchor -positive $positive -negative $negative -distanceFunction "cosine" -margin 0.2 -reduction "mean"]
```

### Benefits of Named Parameters

1. **Self-Documenting**: Parameter names clarify the triplet structure and distance function
2. **Flexible Order**: Parameters can be specified in any order
3. **Distance Clarity**: Explicit distance function specification
4. **Better Maintainability**: Code is easier to understand and modify
5. **IDE Support**: Better autocomplete and parameter hints

## Real-World Applications

### 1. Face Recognition with Cosine Distance
```tcl
# Face embeddings normalized to unit sphere
proc train_face_recognition {face_anchors face_positives face_negatives} {
    set loss [torch::triplet_margin_with_distance_loss -anchor $face_anchors -positive $face_positives -negative $face_negatives -distanceFunction "cosine" -margin 0.2 -reduction "mean"]
    return $loss
}
```

### 2. Text Similarity with Multiple Distance Functions
```tcl
# Compare different distance functions for text embeddings
proc train_text_similarity {text_anchors text_positives text_negatives} {
    # Cosine for semantic similarity
    set cosine_loss [torch::triplet_margin_with_distance_loss -anchor $text_anchors -positive $text_positives -negative $text_negatives -distanceFunction "cosine" -margin 0.1]
    
    # Euclidean for syntactic similarity
    set euclidean_loss [torch::triplet_margin_with_distance_loss -anchor $text_anchors -positive $text_positives -negative $text_negatives -distanceFunction "euclidean" -margin 0.5]
    
    return [list $cosine_loss $euclidean_loss]
}
```

### 3. Multi-Modal Retrieval
```tcl
# Image-text matching with appropriate distance functions
proc train_multimodal_retrieval {image_features text_features matching_pairs non_matching_pairs} {
    # Use cosine distance for cross-modal similarity
    set cross_modal_loss [torch::triplet_margin_with_distance_loss -anchor $image_features -positive $text_features -negative $non_matching_pairs -distanceFunction "cosine" -margin 0.3]
    return $cross_modal_loss
}
``` 