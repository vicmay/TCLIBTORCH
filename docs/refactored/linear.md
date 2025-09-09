# torch::linear / torch::linearLayer

Creates a linear (fully connected) neural network layer.

## Syntax

```tcl
# Positional syntax (backward compatibility)
torch::linear in_features out_features ?bias?

# Named parameter syntax
torch::linear -inFeatures value -outFeatures value ?-bias value?

# camelCase alias
torch::linearLayer -inFeatures value -outFeatures value ?-bias value?
```

## Parameters

| Parameter    | Type    | Default | Description                                  |
|--------------|---------|---------|----------------------------------------------|
| inFeatures   | integer | -       | Size of each input sample                    |
| outFeatures  | integer | -       | Size of each output sample                   |
| bias         | boolean | 1       | If true, adds a learnable bias to the output |

## Return Value

Returns a handle to the created linear layer module.

## Description

The `torch::linear` command creates a linear transformation module that applies a linear transformation to the incoming data: y = xA^T + b.

The command supports both positional and named parameter syntax for backward compatibility.

## Examples

### Positional Syntax

```tcl
# Create a linear layer with 10 input features and 5 output features
set linear [torch::linear 10 5]

# Create a linear layer without bias
set linear_no_bias [torch::linear 10 5 0]

# Use the linear layer in forward pass
set input_tensor [torch::ones [list 2 10]]
set output_tensor [torch::layer_forward $linear $input_tensor]
```

### Named Parameter Syntax

```tcl
# Create a linear layer with named parameters
set linear [torch::linear -inFeatures 10 -outFeatures 5]

# Create a linear layer without bias
set linear_no_bias [torch::linear -inFeatures 10 -outFeatures 5 -bias 0]

# Parameters can be in any order
set linear [torch::linear -bias 1 -outFeatures 5 -inFeatures 10]
```

### camelCase Alias

```tcl
# Use the camelCase alias
set linear [torch::linearLayer -inFeatures 10 -outFeatures 5]
```

## Error Handling

The command will throw an error if:
- Required parameters are missing
- Parameters have invalid values (e.g., negative or zero values for inFeatures or outFeatures)
- Unknown parameters are provided
- A parameter value is missing

## Migration from Positional to Named Syntax

To migrate from the positional to the named parameter syntax:

```tcl
# Old syntax
set linear [torch::linear 10 5]

# New syntax
set linear [torch::linear -inFeatures 10 -outFeatures 5]
```

## See Also

- `torch::layer_forward` - Forward pass through a layer
- `torch::sequential` - Create a sequential container for layers 