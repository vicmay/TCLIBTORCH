# torch::soft_margin_loss

**Soft Margin Loss**

Computes the soft margin loss between input and target tensors. This loss function is commonly used for binary classification tasks where targets are represented as -1 (negative class) or +1 (positive class). It combines elements of hinge loss and logistic loss for robust binary classification.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::soft_margin_loss -input tensor -target tensor ?-reduction string?
torch::softMarginLoss -input tensor -target tensor ?-reduction string?
```

### Positional Syntax (Legacy)
```tcl
torch::soft_margin_loss input target ?reduction?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **input** | tensor | - | Input tensor containing raw predictions (any real values) |
| **target** | tensor | - | Target tensor containing binary labels (+1 or -1) |
| **reduction** | string | "mean" | Reduction mode: "none", "mean", or "sum" |

## Returns

Returns a tensor handle containing the computed soft margin loss.

- If `reduction="none"`: Returns tensor with same shape as input
- If `reduction="mean"`: Returns scalar tensor with mean loss
- If `reduction="sum"`: Returns scalar tensor with sum of losses

## Mathematical Details

The soft margin loss is computed as:

```
soft_margin_loss(x, y) = log(1 + exp(-y * x))
```

Where:
- `x` is the input (raw prediction)
- `y` is the target (+1 or -1)

### Key Properties:
- **Binary classification**: Designed for two-class problems
- **Margin-based**: Encourages correct classifications with confident margins
- **Smooth**: Differentiable everywhere (unlike hinge loss)
- **Probabilistic interpretation**: Related to logistic regression
- **Robust**: Less sensitive to outliers than squared loss

### Relationship to Other Loss Functions:
- **Logistic Loss**: `log(1 + exp(-y * x))` (identical formulation)
- **Hinge Loss**: `max(0, 1 - y * x)` (non-smooth version)
- **Sigmoid + BCE**: Equivalent to applying sigmoid then binary cross entropy

## Examples

### Named Parameter Syntax

#### Basic Usage
```tcl
# Create input tensor (raw predictions)
set input [torch::tensorCreate -data {0.5 -1.2 2.0 -0.8} -shape {2 2}]

# Create target tensor (binary labels: -1 or +1)
set target [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0} -shape {2 2}]

# Compute soft margin loss
set loss [torch::soft_margin_loss -input $input -target $target]
```

#### With Different Reduction Modes
```tcl
# No reduction - preserve input shape
set loss_none [torch::soft_margin_loss -input $input -target $target -reduction "none"]

# Mean reduction (default)
set loss_mean [torch::soft_margin_loss -input $input -target $target -reduction "mean"]

# Sum reduction
set loss_sum [torch::soft_margin_loss -input $input -target $target -reduction "sum"]
```

#### Using camelCase Alias
```tcl
# Equivalent using camelCase alias
set loss [torch::softMarginLoss -input $input -target $target -reduction "mean"]
```

### Positional Syntax (Legacy)

#### Basic Usage
```tcl
# Basic usage with defaults
set loss [torch::soft_margin_loss $input $target]

# With reduction parameter (0=none, 1=mean, 2=sum)
set loss [torch::soft_margin_loss $input $target 0]

# Mean reduction (explicit)
set loss [torch::soft_margin_loss $input $target 1]
```

### Binary Classification Examples

#### SVM-style Classification
```tcl
# SVM-style predictions (positive/negative scores)
set predictions [torch::tensorCreate -data {1.5 -0.8 0.3 -2.1}]
set labels [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0}]
set loss [torch::soft_margin_loss -input $predictions -target $labels]
```

#### Neural Network Output
```tcl
# Raw output from final layer (before sigmoid)
set logits [torch::linear $hidden_features $output_weights]
set binary_targets [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0 1.0}]
set loss [torch::soft_margin_loss -input $logits -target $binary_targets -reduction "mean"]
```

## Use Cases

### Binary Classification
```tcl
# Medical diagnosis (positive/negative)
set diagnostic_scores [torch::tensorCreate -data {0.8 -0.5 1.2 -0.9}]
set diagnosis_labels [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0}]  # 1=disease, -1=healthy
set loss [torch::soft_margin_loss -input $diagnostic_scores -target $diagnosis_labels]
```

### Sentiment Analysis
```tcl
# Sentiment classification (positive/negative)
set sentiment_scores [torch::linear $text_features $classifier_weights]
set sentiment_labels [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0 1.0}]  # 1=positive, -1=negative
set loss [torch::soft_margin_loss -input $sentiment_scores -target $sentiment_labels -reduction "mean"]
```

### Anomaly Detection
```tcl
# Anomaly detection (normal/anomalous)
set anomaly_scores [torch::tensorCreate -data {-1.2 0.8 -0.5 2.0}]
set anomaly_labels [torch::tensorCreate -data {-1.0 1.0 -1.0 1.0}]  # 1=anomaly, -1=normal
set loss [torch::soft_margin_loss -input $anomaly_scores -target $anomaly_labels]
```

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::soft_margin_loss -input $input} error
# Error: Required parameters -input and -target must be provided

# Invalid tensor names
catch {torch::soft_margin_loss -input "invalid" -target $target} error
# Error: Invalid input tensor name

# Unknown parameters
catch {torch::soft_margin_loss -input $input -target $target -unknown value} error
# Error: Unknown parameter: -unknown
```

## Performance Notes

- **Computational Efficiency**: Single log-exp operation per element
- **Memory Usage**: Scales linearly with input tensor size
- **GPU Acceleration**: Fully supports CUDA tensors and GPU computation
- **Numerical Stability**: Uses numerically stable log-sum-exp implementation

## Comparison with Other Binary Loss Functions

| Loss Function | Formula | Smoothness | Margin | Outlier Sensitivity |
|---------------|---------|------------|--------|-------------------|
| **Soft Margin** | `log(1 + exp(-yx))` | Smooth | Soft | Medium |
| **Hinge Loss** | `max(0, 1 - yx)` | Non-smooth | Hard | Low |
| **Squared Hinge** | `max(0, 1 - yx)²` | Smooth | Soft | High |
| **Logistic** | `log(1 + exp(-yx))` | Smooth | Soft | Medium |

## Target Label Requirements

**Important**: Targets must be in {-1, +1} format, not {0, 1}:

```tcl
# CORRECT: Use +1 and -1
set correct_targets [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0}]

# INCORRECT: Don't use 0 and 1
# set wrong_targets [torch::tensorCreate -data {1.0 0.0 1.0 0.0}]  # This will give wrong results
```

## Advanced Examples

### Batch Processing
```tcl
# Process batch of samples
set batch_predictions [torch::tensorCreate -data {0.5 -1.2 2.0 -0.8 1.0 -0.5} -shape {2 3}]  # 2 samples, 3 features each
set batch_labels [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0 1.0 -1.0} -shape {2 3}]
set batch_loss [torch::soft_margin_loss -input $batch_predictions -target $batch_labels -reduction "mean"]
```

### Weight Decay Integration
```tcl
# Combine with weight decay for regularization
set predictions [torch::linear $features $weights]
set classification_loss [torch::soft_margin_loss -input $predictions -target $labels -reduction "mean"]
set weight_penalty [torch::sum [torch::mul $weights $weights]]
set total_loss [torch::add $classification_loss [torch::mul $weight_penalty 0.01]]  # L2 regularization
```

### Confidence-based Weighting
```tcl
# Weight samples by prediction confidence
set raw_predictions [torch::tensorCreate -data {0.1 -2.5 3.0 -0.2}]  # Low to high confidence
set targets [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0}]
set loss_per_sample [torch::soft_margin_loss -input $raw_predictions -target $targets -reduction "none"]
# Apply confidence-based weighting here if needed
```

## Compatibility

- **Backward Compatible**: Original positional syntax remains fully supported
- **Thread Safe**: Can be used safely in multi-threaded environments
- **Device Agnostic**: Works with CPU and CUDA tensors
- **Data Types**: Supports float32, float64, and other floating-point types

## See Also

- [`torch::bce_loss`](bce_loss.md) - Binary Cross Entropy loss (for sigmoid outputs)
- [`torch::hinge_embedding_loss`](hinge_embedding_loss.md) - Hinge loss variant
- [`torch::margin_ranking_loss`](margin_ranking_loss.md) - Margin-based ranking loss
- [`torch::cross_entropy_loss`](cross_entropy_loss.md) - Multi-class classification loss

## Migration Guide

### From Positional to Named Parameters

```tcl
# OLD (Positional)
set loss [torch::soft_margin_loss $input $target 1]

# NEW (Named Parameters)
set loss [torch::soft_margin_loss -input $input -target $target -reduction "mean"]

# NEW (camelCase)
set loss [torch::softMarginLoss -input $input -target $target -reduction "mean"]
```

### From Other Binary Loss Functions

#### From Binary Cross Entropy
```tcl
# OLD: BCE with sigmoid
set probs [torch::sigmoid $logits]
set bce_loss [torch::bce_loss $probs $targets_01]  # targets in {0,1}

# NEW: Soft margin loss (no sigmoid needed)
set targets_pm1 [torch::mul [torch::sub [torch::mul $targets_01 2.0] 1.0] 1.0]  # Convert {0,1} to {-1,1}
set soft_margin_loss [torch::soft_margin_loss -input $logits -target $targets_pm1]
```

### Benefits of Named Parameters

1. **Self-Documenting**: Parameter names clarify intent
2. **Flexible Order**: Parameters can be specified in any order
3. **Less Error-Prone**: No need to remember parameter positions
4. **Better Maintainability**: Code is easier to understand and modify
5. **IDE Support**: Better autocomplete and parameter hints

### Target Format Conversion

```tcl
# Convert from {0,1} to {-1,+1} format
proc convert_binary_targets {targets_01} {
    # targets_01: tensor with values in {0, 1}
    # returns: tensor with values in {-1, +1}
    set doubled [torch::mul $targets_01 2.0]
    set shifted [torch::sub $doubled 1.0]
    return $shifted
}

# Usage
set targets_01 [torch::tensorCreate -data {0 1 1 0 1}]
set targets_pm1 [convert_binary_targets $targets_01]
set loss [torch::soft_margin_loss -input $predictions -target $targets_pm1]
``` 