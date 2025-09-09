# torch::parameters_to / torch::parametersTo

Moves a list of parameter tensors to a specified device.

## Syntax

### Legacy Syntax (Positional Parameters)
```tcl
torch::parameters_to parameters ?device?
torch::parametersTo parameters ?device?  ;# camelCase alias
```

### Modern Syntax (Named Parameters)
```tcl
torch::parameters_to -parameters list ?-device device?
torch::parametersTo -parameters list ?-device device?  ;# camelCase alias
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| parameters/list | list | A list of tensor handles representing model parameters |
| device | string | Target device ('cpu' or 'cuda'). Defaults to 'cpu' if not specified |

## Description

The `parameters_to` command moves a list of parameter tensors to a specified device. This is commonly used in deep learning workflows to move model parameters between CPU and GPU memory.

If no device is specified, the parameters are moved to the CPU by default.

## Examples

### Moving Parameters to CPU
```tcl
# Get model parameters
set params [torch::layer_parameters mymodel]

# Move to CPU using positional syntax
torch::parameters_to $params cpu

# Move to CPU using named syntax
torch::parameters_to -parameters $params -device cpu

# Using camelCase alias
torch::parametersTo -parameters $params -device cpu
```

### Moving Parameters to GPU (if available)
```tcl
# Move to GPU using positional syntax
torch::parameters_to $params cuda

# Move to GPU using named syntax
torch::parameters_to -parameters $params -device cuda
```

### Using Default Device (CPU)
```tcl
# Parameters will be moved to CPU
torch::parameters_to $params
torch::parameters_to -parameters $params
```

## Error Handling

The command will raise an error if:
- The parameters list is missing or invalid
- The specified device is neither 'cpu' nor 'cuda'
- Any of the tensors in the parameters list is invalid

## See Also

- `torch::layer_parameters` - Get trainable parameters from a layer
- `torch::layer_to` - Move a layer to a specific device 