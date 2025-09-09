# torch::unfreeze_model

Re-enables gradient computation for all parameters of a model that was previously frozen.

## Syntax

### Positional syntax
```tcl
torch::unfreeze_model model
```

### Named-parameter syntax
```tcl
torch::unfreeze_model -model model
```

### camelCase alias
```tcl
torch::unfreezeModel model
torch::unfreezeModel -model model
```

## Parameters
| Name | Type | Required | Description |
|------|------|----------|-------------|
| model | string | yes | Model handle to unfreeze |

## Returns
String "Model parameters unfrozen" on success.

## Description
Calling this command sets `requires_grad=true` for every parameter in the specified model, allowing them to be updated during back-propagation.

Typical scenarios:
* Fine-tuning after initial freezing
* Switching from inference to training
* Gradual-unfreezing training schedules

## Examples
```tcl
set m [torch::linear 128 64]
# Freeze then unfreeze
torch::freeze_model $m
torch::unfreeze_model $m  ;# -> Model parameters unfrozen
```

Using named parameters:
```tcl
torch::unfreeze_model -model $m
```

Using camelCase alias:
```tcl
torch::unfreezeModel $m
```

## Error handling
* Wrong arg count → Tcl error
* Unknown / missing `-model` value → descriptive runtime error
* Non-existent model → "Model not found"

## Related
* `torch::freeze_model` – complementary command 