# torch::save_state_dict

Save a model's state dictionary (parameters) to a file.

## Syntax

```tcl
# Positional syntax
torch::save_state_dict model filename

# Named parameter syntax
torch::save_state_dict -model model -filename filename
```

The command also supports a camelCase alias: `torch::saveStateDict`

## Arguments

* `model` (positional) or `-model model` (named): Name of the model to save
* `filename` (positional) or `-filename/-file filename` (named): Path to save the state dictionary file

## Return Value

Returns a message indicating the file where the state dictionary was saved.

## Examples

```tcl
# Create a model
set model [torch::linear 10 5]

# Using positional syntax
torch::save_state_dict $model "model_params.pt"

# Using named parameter syntax
torch::save_state_dict -model $model -filename "model_params.pt"

# Using named parameter syntax with -file
torch::save_state_dict -model $model -file "model_params.pt"

# Using camelCase alias
torch::saveStateDict -model $model -filename "model_params.pt"
```

## Error Conditions

* If required parameters (model, filename) are missing
* If the model name is not found in the module storage
* If an unknown parameter is provided in named syntax

## See Also

* `torch::load_state_dict` - Load a saved state dictionary
* `torch::save_state` - Save complete module state
* `torch::save_checkpoint` - Save complete model checkpoint 