# torch::save_state

Save a module's complete state to a file.

## Syntax

```tcl
# Positional syntax
torch::save_state module filename

# Named parameter syntax
torch::save_state -module module -filename filename
```

The command also supports a camelCase alias: `torch::saveState`

## Arguments

* `module` (positional) or `-module module` (named): Name of the module to save
* `filename` (positional) or `-filename/-file filename` (named): Path to save the state file

## Return Value

Returns "OK" if the state was saved successfully.

## Examples

```tcl
# Create a model
set model [torch::linear 10 5]

# Using positional syntax
torch::save_state $model "model_state.pt"

# Using named parameter syntax
torch::save_state -module $model -filename "model_state.pt"

# Using named parameter syntax with -file
torch::save_state -module $model -file "model_state.pt"

# Using camelCase alias
torch::saveState -module $model -filename "model_state.pt"
```

## Error Conditions

* If required parameters (module, filename) are missing
* If the module name is not found in the module storage
* If an unknown parameter is provided in named syntax

## See Also

* `torch::load_state` - Load a saved module state
* `torch::save_state_dict` - Save model state dictionary only
* `torch::save_checkpoint` - Save complete model checkpoint 