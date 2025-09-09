# torch::gaussian_nll_loss

## Overview

Computes the Gaussian Negative Log-Likelihood (NLL) Loss between input predictions and target values, given a variance tensor. This loss function is particularly useful for regression tasks where you need to model both the mean and uncertainty (variance) of predictions.

The Gaussian NLL Loss measures how well the input predictions match the target values under a Gaussian distribution assumption, taking into account the predicted variance for each sample.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::gaussian_nll_loss -input tensor -target tensor -var tensor ?-full bool? ?-eps double? ?-reduction string?
torch::gaussianNllLoss -input tensor -target tensor -var tensor ?-full bool? ?-eps double? ?-reduction string?
```

### Positional Syntax (Legacy)
```tcl
torch::gaussian_nll_loss input target var ?full? ?eps? ?reduction?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **input** | tensor | required | Input tensor containing predictions (mean values) |
| **target** | tensor | required | Target tensor containing ground truth values |
| **var** | tensor | required | Variance tensor containing predicted variances (must be positive) |
| **full** | bool | false | Whether to include the constant term in the loss calculation |
| **eps** | double | 1e-6 | Small value to clamp variance to avoid numerical instability |
| **reduction** | string | "mean" | Specifies the reduction to apply: "none", "mean", or "sum" |

## Returns

Returns a tensor handle containing the computed Gaussian NLL loss:
- If reduction="none": tensor with same shape as input, containing loss for each element
- If reduction="mean": scalar tensor with mean loss
- If reduction="sum": scalar tensor with sum of losses

## Mathematical Formula

The Gaussian Negative Log-Likelihood Loss is computed as:

### Base Formula (full=false)
**loss = 0.5 × ((input - target)² / var + log(var))**

### Full Formula (full=true)  
**loss = 0.5 × ((input - target)² / var + log(var) + log(2π))**

Where:
- **input**: predicted mean values
- **target**: ground truth values  
- **var**: predicted variance (clamped to be ≥ eps)
- **eps**: small constant to prevent division by zero

The loss consists of two terms:
1. **Data fit term**: (input - target)² / var - measures how well predictions match targets
2. **Regularization term**: log(var) - penalizes large variances to prevent overconfidence

## Examples

### Basic Usage
```tcl
# Create input predictions (means)
set predictions [torch::tensor_create {1.0 2.0 0.5 1.5} -dtype float32]

# Create target values
set targets [torch::tensor_create {0.8 1.8 0.3 1.2} -dtype float32]

# Create variance predictions (must be positive)
set variances [torch::tensor_create {0.1 0.2 0.1 0.15} -dtype float32]

# Compute Gaussian NLL loss
set loss [torch::gaussian_nll_loss -input $predictions -target $targets -var $variances]
puts "Gaussian NLL Loss: [torch::tensor_item $loss]"
```

### Full Loss with Constant Term
```tcl
# Include the constant log(2π) term in loss calculation
set loss_full [torch::gaussian_nll_loss -input $predictions -target $targets -var $variances -full 1]
set loss_base [torch::gaussian_nll_loss -input $predictions -target $targets -var $variances -full 0]

puts "Full loss: [torch::tensor_item $loss_full]"
puts "Base loss: [torch::tensor_item $loss_base]"
# Full loss will be higher due to the constant term
```

### Custom Epsilon for Numerical Stability
```tcl
# Very small variances - use larger eps to prevent numerical issues
set small_vars [torch::tensor_create {1e-8 1e-7 1e-9} -dtype float32]
set preds [torch::tensor_create {1.0 2.0 0.5} -dtype float32]
set targs [torch::tensor_create {0.9 1.9 0.4} -dtype float32]

set loss [torch::gaussian_nll_loss -input $preds -target $targs -var $small_vars -eps 1e-4]
```

### Different Reduction Options
```tcl
# No reduction - returns per-element losses
set losses_none [torch::gaussian_nll_loss -input $predictions -target $targets -var $variances -reduction none]
set shape [torch::tensor_shape $losses_none]  ;# Same shape as input

# Sum reduction
set total_loss [torch::gaussian_nll_loss -input $predictions -target $targets -var $variances -reduction sum]

# Mean reduction (default)
set avg_loss [torch::gaussian_nll_loss -input $predictions -target $targets -var $variances -reduction mean]
```

## Use Cases

### 1. Uncertainty-Aware Regression
Train neural networks to predict both mean and variance for regression tasks.

```tcl
# Neural network outputs: [mean_pred, log_var_pred]
# Convert log variance to variance
set var_pred [torch::exp $log_var_pred]

# Compute loss with predicted uncertainty
set loss [torch::gaussianNllLoss -input $mean_pred -target $targets -var $var_pred]
```

### 2. Heteroscedastic Regression
Model scenarios where noise varies across different inputs.

```tcl
# Different noise levels for different samples
set high_noise_var [torch::tensor_create {1.0 1.2 0.8} -dtype float32]
set low_noise_var [torch::tensor_create {0.1 0.15 0.08} -dtype float32]

# Model will learn to predict appropriate variance for each case
set loss [torch::gaussian_nll_loss -input $predictions -target $targets -var $predicted_vars]
```

### 3. Bayesian Neural Networks
Estimate epistemic uncertainty in neural network predictions.

```tcl
# Multiple forward passes through dropout-enabled network
# Compute mean and variance of predictions
set loss [torch::gaussian_nll_loss -input $ensemble_mean -target $targets -var $ensemble_var -full 1]
```

### 4. Time Series with Uncertainty
Forecast time series with confidence intervals.

```tcl
# Predict next values with uncertainty bounds
set forecast_loss [torch::gaussian_nll_loss -input $forecast_means -target $actual_values -var $forecast_vars]
```

## Parameter Guidelines

### Epsilon (eps)
- **1e-6**: Default value, works well for most cases
- **1e-4**: Use for very small predicted variances
- **1e-8**: Use when variances are well-behaved and you need precision

### Full Parameter
- **false**: Use when you only need relative loss values (default)
- **true**: Use when you need the complete likelihood calculation

### Variance Constraints
- Variance values must be positive
- Consider using `torch::exp(log_var)` or `torch::softplus(raw_var)` to ensure positivity
- Very small variances (< eps) will be clamped to eps

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
torch::gaussian_nll_loss $input $target $var 1 1e-5 0

# New named syntax
torch::gaussian_nll_loss -input $input -target $target -var $var -full 1 -eps 1e-5 -reduction none
```

### Reduction Parameter Changes
```tcl
# Old integer reduction values
# 0 = none, 1 = mean, 2 = sum

# New string reduction values
-reduction none   # Per-element losses
-reduction mean   # Average loss
-reduction sum    # Total loss
```

## Performance Notes

- **Computational Cost**: Moderate due to log and division operations
- **Memory Usage**: Same as input tensors
- **Numerical Stability**: Uses eps clamping to prevent division by zero
- **Gradient Flow**: Provides gradients for both mean and variance predictions

## Error Handling

The command validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::gaussian_nll_loss -input $input -target $target} error
# Returns: "Required parameters -input, -target, and -var must be provided"

# Invalid reduction type
catch {torch::gaussian_nll_loss -input $input -target $target -var $var -reduction invalid} error
# Returns: "Invalid reduction type: invalid"

# Invalid tensor references
catch {torch::gaussian_nll_loss invalid_input $target $var} error
# Returns: "Invalid input tensor name"
```

## Mathematical Properties

### Loss Characteristics
- **Can be negative**: Unlike MSE, Gaussian NLL can have negative values
- **Uncertainty-aware**: Accounts for both prediction accuracy and confidence
- **Proper scoring rule**: Encourages both accurate predictions and calibrated uncertainty

### Variance Impact
- **Low variance + accurate prediction**: Very low loss
- **Low variance + inaccurate prediction**: High loss (overconfident)
- **High variance + inaccurate prediction**: Moderate loss (underconfident but aware)
- **High variance + accurate prediction**: Moderate loss (unnecessarily uncertain)

## Advantages and Limitations

### Advantages
- **Uncertainty Quantification**: Provides both point estimates and uncertainty
- **Proper Scoring**: Rewards well-calibrated predictions
- **Flexible**: Works with any differentiable variance parameterization
- **Principled**: Based on maximum likelihood estimation

### Limitations
- **Variance Prediction Required**: Need to predict/estimate variance values
- **Gaussian Assumption**: Assumes Gaussian noise distribution
- **Hyperparameter Sensitivity**: eps value can affect training dynamics
- **Complexity**: More complex than simple MSE loss

## See Also

- [`torch::mse_loss`](mse_loss.md) - Mean squared error loss
- [`torch::l1_loss`](l1_loss.md) - L1/Mean absolute error loss
- [`torch::huber_loss`](huber_loss.md) - Huber loss for robust regression
- [`torch::tensor_create`](../tensor_create.md) - Creating input tensors
- [`torch::tensor_exp`](../tensor_exp.md) - Exponential function for variance
- [`torch::tensor_log`](../tensor_log.md) - Logarithm function

## References

1. Murphy, K. P. (2012). "Machine Learning: A Probabilistic Perspective" - Chapter on Gaussian distributions
2. Gal, Y. (2016). "Uncertainty in Deep Learning" - Bayesian deep learning
3. Nix, D. A., & Weigend, A. S. (1994). "Estimating the mean and variance of the target probability distribution"
4. PyTorch GaussianNLLLoss documentation for reference 