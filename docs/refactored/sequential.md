# torch::sequential / torch::sequential

Creates a sequential container for neural network modules.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::sequential ?module_list?
```

### Named Parameter Syntax
```tcl
torch::sequential ?-modules module_list?
torch::sequential ?-modules module_list?  ;# camelCase alias
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| module_list | list | Optional list of module handles to add to the sequential container |

## Return Value

Returns a handle to the created sequential container.

## Description

The `sequential` command creates a sequential container that holds a list of neural network modules. The modules are executed in sequence during the forward pass, with each module's output becoming the input to the next module.

The command:
1. Creates a new sequential container
2. If modules are provided, adds them to the container in order
3. Returns a handle to the container

This is useful when you want to:
- Create a linear sequence of neural network layers
- Group multiple operations into a single module
- Build complex neural networks in a modular way

## Examples

### Basic Usage
```tcl
# Create an empty sequential container
set model [torch::sequential]

# Create a sequential container with modules
set linear1 [torch::linear -inFeatures 784 -outFeatures 128]
set relu [torch::relu]
set linear2 [torch::linear -inFeatures 128 -outFeatures 10]
set model [torch::sequential [list $linear1 $relu $linear2]]
```

### Named Parameter Syntax
```tcl
# Create a sequential container with modules using named parameters
set conv1 [torch::conv2d -inChannels 1 -outChannels 32 -kernelSize 3]
set bn1 [torch::batchnorm2d -numFeatures 32]
set pool1 [torch::maxpool2d -kernelSize 2]
set model [torch::sequential -modules [list $conv1 $bn1 $pool1]]
```

### Forward Pass
```tcl
# Create a sequential model
set linear1 [torch::linear -inFeatures 10 -outFeatures 20]
set relu [torch::relu]
set linear2 [torch::linear -inFeatures 20 -outFeatures 5]
set model [torch::sequential -modules [list $linear1 $relu $linear2]]

# Forward pass through the model
set input [torch::tensor_create {1 2 3 4 5 6 7 8 9 10} float32]
set output [torch::layer_forward $model $input]
```

## Common Use Cases

1. **Simple Feed-Forward Networks**
   ```tcl
   set model [torch::sequential -modules [list \
       [torch::linear -inFeatures 784 -outFeatures 128] \
       [torch::relu] \
       [torch::linear -inFeatures 128 -outFeatures 10] \
   ]]
   ```

2. **CNN Feature Extractors**
   ```tcl
   set feature_extractor [torch::sequential -modules [list \
       [torch::conv2d -inChannels 3 -outChannels 64 -kernelSize 3] \
       [torch::batchnorm2d -numFeatures 64] \
       [torch::relu] \
       [torch::maxpool2d -kernelSize 2] \
   ]]
   ```

3. **Multi-Layer Networks with Dropout**
   ```tcl
   set classifier [torch::sequential -modules [list \
       [torch::linear -inFeatures 512 -outFeatures 256] \
       [torch::relu] \
       [torch::dropout -p 0.5] \
       [torch::linear -inFeatures 256 -outFeatures 10] \
   ]]
   ```

## See Also

- [torch::layer_forward](layer_forward.md) - Forward pass through a layer or container
- [torch::linear](linear.md) - Linear (fully connected) layer
- [torch::conv2d](conv2d.md) - 2D convolution layer
- [torch::relu](relu.md) - ReLU activation function 