# torch::model_eval

Sets a PyTorch model to evaluation mode.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::model_eval model
```

### Named Parameters (New Syntax)
```tcl
torch::model_eval -model model_name
```

### CamelCase Alias
```tcl
torch::modelEval -model model_name
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` / `-model` | string | Name of the model to set to evaluation mode | Required |

## Returns

Returns the model name (as a string) to allow for command chaining.

## Description

The `torch::model_eval` command sets a PyTorch model to evaluation mode. This is equivalent to calling `model->eval()` in C++ or `model.eval()` in Python.

When a model is in evaluation mode:
- Batch normalization layers use running statistics rather than batch statistics
- Dropout layers are disabled (no neurons are dropped)
- Other layers may behave differently depending on their implementation

This is typically used during inference or validation when you don't want to update the model's running statistics or apply dropout.

## Examples

### Basic Usage

```tcl
# Create a model
set model [torch::nn_sequential]
torch::nn_add_linear $model 10 5
torch::nn_add_relu $model
torch::nn_add_linear $model 5 1

# Set model to evaluation mode (positional syntax)
torch::model_eval $model

# Set model to evaluation mode (named parameter syntax)
torch::model_eval -model $model

# Set model to evaluation mode (camelCase alias)
torch::modelEval -model $model
```

### Command Chaining

```tcl
# Create a model, set to evaluation mode, and perform inference in one chain
set output [torch::forward [torch::model_eval $model] $input]
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
torch::model_eval $model

# New syntax
torch::model_eval -model $model

# CamelCase alias
torch::modelEval -model $model
```

## See Also

- `torch::model_train` - Set a model to training mode
- `torch::forward` - Forward pass through a model
- `torch::nn_sequential` - Create a sequential model
