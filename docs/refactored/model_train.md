# torch::model_train

Sets a PyTorch model to training mode.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::model_train model
```

### Named Parameters (New Syntax)
```tcl
torch::model_train -model model_name
```

### CamelCase Alias
```tcl
torch::modelTrain -model model_name
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` / `-model` | string | Name of the model to set to training mode | Required |

## Returns

Returns the model name (for command chaining).

## Description

The `torch::model_train` command sets a PyTorch model to training mode by calling the `train()` method on the underlying C++ module. This enables features like dropout and batch normalization to behave differently during training compared to evaluation.

When a model is in training mode:
- Dropout layers will randomly zero some elements of the input tensor
- Batch normalization layers will update their running statistics based on the current batch

This command is typically used before starting a training loop.

## Examples

### Basic Usage

```tcl
# Create a model
set model [torch::sequential]
torch::nn_add_linear $model 10 5
torch::nn_add_dropout $model 0.5
torch::nn_add_linear $model 5 1

# Set model to training mode (positional syntax)
torch::model_train $model

# Set model to training mode (named parameter syntax)
torch::model_train -model $model

# Set model to training mode (camelCase alias)
torch::modelTrain -model $model
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
torch::model_train $model

# New syntax
torch::model_train -model $model

# CamelCase alias
torch::modelTrain -model $model
```

## See Also

- `torch::model_eval` - Set a model to evaluation mode
- `torch::model_summary` - Get a summary of a model's parameters
