# torch::model_summary

Provides a summary of a PyTorch model, including parameter counts.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::model_summary model
```

### Named Parameters (New Syntax)
```tcl
torch::model_summary -model model_name
```

### CamelCase Alias
```tcl
torch::modelSummary -model model_name
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` / `-model` | string | Name of the model to summarize | Required |

## Returns

Returns a string containing the model summary with the following information:
- Total number of parameters
- Number of trainable parameters
- Number of non-trainable parameters

## Description

The `torch::model_summary` command provides a summary of a PyTorch model, including the total number of parameters, the number of trainable parameters (those with `requires_grad=true`), and the number of non-trainable parameters.

This command is useful for understanding the complexity of a model and verifying that parameters are properly set to be trainable or non-trainable as expected.

## Examples

### Basic Usage

```tcl
# Create a model
set model [torch::sequential]
torch::nn_add_linear $model 10 5
torch::nn_add_relu $model
torch::nn_add_linear $model 5 1

# Get model summary (positional syntax)
puts [torch::model_summary $model]

# Get model summary (named parameter syntax)
puts [torch::model_summary -model $model]

# Get model summary (camelCase alias)
puts [torch::modelSummary -model $model]
```

### Example Output

```
Model Summary:
Total parameters: 61
Trainable parameters: 61
Non-trainable parameters: 0
```

## Error Handling

The command will raise an error in the following cases:
- The model name is invalid or does not exist
- Required parameters are missing
- Unknown parameters are provided

## Migration from Old Syntax

If you have existing code using the positional syntax, it will continue to work unchanged:

```tcl
# Old syntax (still supported)
torch::model_summary $model

# New syntax
torch::model_summary -model $model

# CamelCase alias
torch::modelSummary -model $model
```

## See Also

- `torch::count_parameters` - Count the parameters in a model
- `torch::model_eval` - Set a model to evaluation mode
- `torch::model_train` - Set a model to training mode
