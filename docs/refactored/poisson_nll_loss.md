# torch::poisson_nll_loss

**Poisson Negative Log Likelihood Loss**

Computes the Poisson negative log likelihood loss between input and target tensors. This loss function is commonly used for regression problems where the target represents count data that follows a Poisson distribution.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::poisson_nll_loss -input tensor -target tensor ?-logInput bool? ?-full bool? ?-reduction string?
torch::poissonNllLoss -input tensor -target tensor ?-logInput bool? ?-full bool? ?-reduction string?
```

### Positional Syntax (Legacy)
```tcl
torch::poisson_nll_loss input target ?log_input? ?full? ?reduction?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **input** | tensor | - | Input tensor containing log probabilities (if logInput=true) or raw values (if logInput=false) |
| **target** | tensor | - | Target tensor containing Poisson observations (count data) |
| **logInput** | bool | true | If true, input is treated as log probabilities; if false, as raw values |
| **full** | bool | false | Whether to include the Stirling approximation term in the loss calculation |
| **reduction** | string | "mean" | Reduction mode: "none", "mean", or "sum" |

## Returns

Returns a tensor handle containing the computed Poisson NLL loss.

- If `reduction="none"`: Returns tensor with same shape as input
- If `reduction="mean"`: Returns scalar tensor with mean loss
- If `reduction="sum"`: Returns scalar tensor with sum of losses

## Mathematical Details

The Poisson negative log likelihood loss is computed as:

### When logInput=true (default):
```
loss = exp(input) - target * input + log(target!)
```

### When logInput=false:
```
loss = input - target * log(input) + log(target!)
```

### When full=true:
Includes the Stirling approximation for the factorial term:
```
log(target!) ≈ target * log(target) - target + 0.5 * log(2π * target)
```

## Examples

### Named Parameter Syntax

#### Basic Usage
```tcl
# Create input tensor (log probabilities)
set input [torch::tensorCreate -data {-1.0 -2.0 -0.5 -3.0} -shape {2 2}]

# Create target tensor (count data)
set target [torch::tensorCreate -data {1.0 0.0 2.0 1.0} -shape {2 2}]

# Compute Poisson NLL loss
set loss [torch::poisson_nll_loss -input $input -target $target]
```

#### With Custom Parameters
```tcl
# Using raw values instead of log probabilities
set loss [torch::poisson_nll_loss -input $input -target $target -logInput false -reduction "sum"]

# Include full Stirling approximation
set loss [torch::poisson_nll_loss -input $input -target $target -full true -reduction "none"]
```

#### Using camelCase Alias
```tcl
# Equivalent using camelCase alias
set loss [torch::poissonNllLoss -input $input -target $target -logInput true -full false]
```

### Positional Syntax (Legacy)

#### Basic Usage
```tcl
# Basic usage with defaults
set loss [torch::poisson_nll_loss $input $target]

# With log_input flag
set loss [torch::poisson_nll_loss $input $target false]

# With all parameters
set loss [torch::poisson_nll_loss $input $target true true 1]
```

### Different Reduction Modes

```tcl
# No reduction - preserve input shape
set loss_none [torch::poisson_nll_loss -input $input -target $target -reduction "none"]

# Mean reduction (default)
set loss_mean [torch::poisson_nll_loss -input $input -target $target -reduction "mean"]

# Sum reduction
set loss_sum [torch::poisson_nll_loss -input $input -target $target -reduction "sum"]
```

## Use Cases

### Count Data Regression
```tcl
# Predicting number of events (e.g., customer arrivals, website clicks)
set predicted_log_rates [torch::linear $features $weights]
set actual_counts [torch::tensorCreate -data {3 1 4 2 0 5}]
set loss [torch::poisson_nll_loss -input $predicted_log_rates -target $actual_counts]
```

### Time Series Forecasting
```tcl
# Forecasting count-based time series data
set forecast_log_rates [torch::lstm_forward $time_series]
set true_counts [torch::tensorCreate -data {2 3 1 4 0}]
set loss [torch::poisson_nll_loss -input $forecast_log_rates -target $true_counts -reduction "mean"]
```

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::poisson_nll_loss -input $input} error
# Error: Required parameters -input and -target must be provided

# Invalid tensor names
catch {torch::poisson_nll_loss -input "invalid" -target $target} error
# Error: Invalid input tensor name

# Unknown parameters
catch {torch::poisson_nll_loss -input $input -target $target -unknown value} error
# Error: Unknown parameter: -unknown
```

## Performance Notes

- The function uses PyTorch's optimized C++ implementation
- Memory usage scales linearly with input tensor size
- GPU acceleration supported when tensors are on CUDA device
- Reduction modes affect output tensor size and memory usage

## Compatibility

- **Backward Compatible**: Original positional syntax remains fully supported
- **Thread Safe**: Can be used safely in multi-threaded environments
- **Device Agnostic**: Works with CPU and CUDA tensors
- **Data Types**: Supports float32, float64, and other floating-point types

## See Also

- [`torch::nll_loss`](nll_loss.md) - Negative log likelihood loss
- [`torch::cross_entropy_loss`](cross_entropy_loss.md) - Cross entropy loss
- [`torch::gaussian_nll_loss`](gaussian_nll_loss.md) - Gaussian negative log likelihood loss
- [`torch::bce_loss`](bce_loss.md) - Binary cross entropy loss

## Migration Guide

### From Positional to Named Parameters

```tcl
# OLD (Positional)
set loss [torch::poisson_nll_loss $input $target false true 2]

# NEW (Named Parameters)
set loss [torch::poisson_nll_loss -input $input -target $target -logInput false -full true -reduction "sum"]

# NEW (camelCase)
set loss [torch::poissonNllLoss -input $input -target $target -logInput false -full true -reduction "sum"]
```

### Benefits of Named Parameters

1. **Self-Documenting**: Parameter names make code more readable
2. **Flexible Order**: Parameters can be specified in any order
3. **Optional Parameters**: Only specify the parameters you need
4. **Less Error-Prone**: No need to remember parameter positions
5. **IDE Support**: Better autocomplete and parameter hints 